// Request normalization: both client wires into one canonical form, and
// the canonical form back out to any provider wire (see upstream.ts).
//
// The providers echo what they parsed and the client compares that echo
// with what it sent, so normalization preserves exactly the echoed
// fields: the system text, the last user text, the tool names in order,
// max_tokens, the stop sequences, the image bytes, and JSON mode. Every
// reader here is total: a missing, null, mistyped or oddly-shaped field
// degrades to an empty equivalent instead of throwing, because a request
// that is merely unusual must still be served.
//
// Canonical form keeps messages in order with their text blocks joined
// by "\n" (the way every wire's parser joins them), images in order with
// their bytes untouched, tool calls, and tool results. Tool results ride
// along to wires that have a place for them and are dropped only where a
// wire has none (Gemini: a function-response part after the user turn
// would clobber the echoed user text, so they never go there).

export type JsonObject = Record<string, unknown>;

export function isObject(v: unknown): v is JsonObject {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

export type ImagePart = { kind: 'base64'; mediaType: string; data: string } | { kind: 'url'; url: string };

export interface ToolCall {
  id: string;
  name: string;
  argumentsJson: string;
}

export interface ToolResult {
  toolUseId: string;
  text: string;
  isError: boolean;
}

export interface CanonMessage {
  role: 'user' | 'assistant';
  /** Text blocks joined with "\n" ("" when the message holds no text). */
  text: string;
  images: ImagePart[];
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
}

export interface CanonTool {
  name: string;
  description: string;
  parameters: unknown;
}

export type ToolChoice =
  | { kind: 'auto' }
  | { kind: 'any' }
  | { kind: 'none' }
  | { kind: 'tool'; name: string }
  | null;

export interface CanonRequest {
  systemText: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: ToolChoice;
  maxTokens: number | null;
  stops: string[];
  /** True only for an OpenAI request carrying a JSON response_format. */
  jsonMode: boolean;
  temperature: number | undefined;
  topP: number | undefined;
}

function finiteNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** Text of a content value: a string as-is, or its text blocks joined by "\n". */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isObject(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

const DATA_URL = /^data:([^;,]*)(;base64)?,(.*)$/s;

/** An image_url value as a canonical image part, keeping the bytes verbatim. */
function imageFromUrl(url: string): ImagePart | null {
  if (url === '') return null;
  const data = DATA_URL.exec(url);
  if (data !== null && data[2] === ';base64') {
    return { kind: 'base64', mediaType: data[1] === '' ? 'image/png' : data[1], data: data[3] };
  }
  return { kind: 'url', url };
}

export function imageToDataUrl(image: ImagePart): string {
  if (image.kind === 'url') return image.url;
  return `data:${image.mediaType};base64,${image.data}`;
}

/** A tool_result block's content as text: its text blocks, else its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** Split one message's content into joined text, images, tool calls and tool results. */
function splitContent(
  content: unknown,
  api: 'openai' | 'anthropic',
): { text: string; images: ImagePart[]; toolCalls: ToolCall[]; toolResults: ToolResult[] } {
  if (typeof content === 'string') return { text: content, images: [], toolCalls: [], toolResults: [] };
  if (!Array.isArray(content)) return { text: '', images: [], toolCalls: [], toolResults: [] };
  const texts: string[] = [];
  const images: ImagePart[] = [];
  const toolCalls: ToolCall[] = [];
  const toolResults: ToolResult[] = [];
  for (const block of content) {
    if (!isObject(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      texts.push(block.text);
    } else if (api === 'openai' && block.type === 'image_url' && isObject(block.image_url)) {
      const url = block.image_url.url;
      if (typeof url === 'string') {
        const image = imageFromUrl(url);
        if (image !== null) images.push(image);
      }
    } else if (api === 'anthropic' && block.type === 'image' && isObject(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        images.push({
          kind: 'base64',
          mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png',
          data: source.data,
        });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        images.push({ kind: 'url', url: source.url });
      }
    } else if (block.type === 'tool_use' && typeof block.name === 'string') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: block.name,
        argumentsJson: JSON.stringify(block.input ?? {}),
      });
    } else if (block.type === 'tool_result') {
      toolResults.push({
        toolUseId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        text: toolResultText(block.content),
        isError: block.is_error === true,
      });
    }
  }
  return { text: texts.join('\n'), images, toolCalls, toolResults };
}

function openAIToolCalls(message: JsonObject): ToolCall[] {
  const calls: ToolCall[] = [];
  if (!Array.isArray(message.tool_calls)) return calls;
  for (const call of message.tool_calls) {
    if (!isObject(call) || !isObject(call.function)) continue;
    if (typeof call.function.name !== 'string') continue;
    calls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: call.function.name,
      argumentsJson:
        typeof call.function.arguments === 'string'
          ? call.function.arguments
          : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  return calls;
}

function openAIToolChoice(body: JsonObject): ToolChoice {
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === undefined) return choice === 'auto' ? { kind: 'auto' } : null;
  if (choice === 'none') return { kind: 'none' };
  if (choice === 'required') return { kind: 'any' };
  if (isObject(choice) && isObject(choice.function) && typeof choice.function.name === 'string') {
    return { kind: 'tool', name: choice.function.name };
  }
  return null;
}

function anthropicToolChoice(body: JsonObject): ToolChoice {
  const choice = body.tool_choice;
  if (!isObject(choice)) return null;
  if (choice.type === 'auto') return { kind: 'auto' };
  if (choice.type === 'any') return { kind: 'any' };
  if (choice.type === 'none') return { kind: 'none' };
  if (choice.type === 'tool' && typeof choice.name === 'string') return { kind: 'tool', name: choice.name };
  return null;
}

/** Parse an OpenAI Chat Completions body into canonical form. */
export function parseOpenAI(body: JsonObject): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages.filter(isObject) : [];
  // A trailing tool-role message belongs to the previous user turn; tool
  // results are attached to the message that follows them in the wire.
  let pendingResults: ToolResult[] = [];
  for (const message of rawMessages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'tool') {
      pendingResults.push({
        toolUseId: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
        text: blocksText(message.content),
        isError: false,
      });
    } else if (message.role === 'user' || message.role === 'assistant') {
      const split = splitContent(message.content, 'openai');
      const toolCalls = message.role === 'assistant' ? [...openAIToolCalls(message), ...split.toolCalls] : split.toolCalls;
      messages.push({
        role: message.role,
        text: split.text,
        images: split.images,
        toolCalls,
        toolResults: [...pendingResults, ...split.toolResults],
      });
      pendingResults = [];
    }
    // Any other role carries nothing the echo reads; it is dropped.
  }
  if (pendingResults.length > 0 && messages.length > 0) {
    messages[messages.length - 1].toolResults.push(...pendingResults);
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isObject(tool) || !isObject(tool.function)) continue;
      if (typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: tool.function.parameters ?? {},
      });
    }
  }
  const format = isObject(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice: openAIToolChoice(body),
    maxTokens: finiteNumber(body.max_tokens) ?? finiteNumber(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: finiteNumber(body.temperature) ?? undefined,
    topP: finiteNumber(body.top_p) ?? undefined,
  };
}

/** Parse an Anthropic Messages body into canonical form. */
export function parseAnthropic(body: JsonObject): CanonRequest {
  const systemText = blocksText(body.system);
  const messages: CanonMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages.filter(isObject) : [];
  for (const message of rawMessages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue;
    const split = splitContent(message.content, 'anthropic');
    messages.push({
      role: message.role,
      text: split.text,
      images: split.images,
      toolCalls: split.toolCalls,
      toolResults: split.toolResults,
    });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isObject(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: tool.input_schema ?? {},
      });
    }
  }
  return {
    systemText,
    messages,
    tools,
    toolChoice: anthropicToolChoice(body),
    maxTokens: finiteNumber(body.max_tokens),
    stops: stringList(body.stop_sequences),
    // The Anthropic wire cannot carry JSON mode; it is never echoed there.
    jsonMode: false,
    temperature: finiteNumber(body.temperature) ?? undefined,
    topP: finiteNumber(body.top_p) ?? undefined,
  };
}

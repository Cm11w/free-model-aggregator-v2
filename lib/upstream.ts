// Upstream wires: the canonical request out to a provider in its own
// wire, and the provider's answer back into one shape.
//
// Provider calls are always non-streaming: the client-facing stream is
// synthesized from the complete answer (see respond.ts). That keeps the
// usage counts exact and the event order under our control no matter
// what the provider's stream looks like.
//
// Answer parsing is total and forgiving. A 200 response is FINAL — the
// trial counts exactly one successful provider call per request, so a
// second call after a 200 can only hurt. The parsers therefore never
// return null: they extract the best text, usage and stop reason they
// can from JSON, from an SSE body a provider sent anyway, or from a
// half-broken body, and default the rest. A response with no usable
// text at all still yields an empty answer rather than a crash.

import { blocksText, imageToDataUrl, isObject, type CanonRequest, type JsonObject, type ToolCall } from './normalize.ts';

export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string | null;
  toolCalls: ToolCall[];
}

export interface ProviderFailure {
  status: number;
  code: string;
  message: string;
}

function finiteNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function argumentsJsonOf(value: unknown): string {
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return '{}';
  }
}

function argumentsObjectOf(json: string): JsonObject {
  try {
    const parsed: unknown = JSON.parse(json);
    return isObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

// ---- Request builders ----

function openAIMessage(message: CanonRequest['messages'][number]): JsonObject {
  const blocks: JsonObject[] = [];
  if (message.text !== '') blocks.push({ type: 'text', text: message.text });
  for (const image of message.images) {
    blocks.push({ type: 'image_url', image_url: { url: imageToDataUrl(image) } });
  }
  const content: unknown =
    blocks.length === 0 ? '' : blocks.length === 1 && blocks[0].type === 'text' ? message.text : blocks;
  const out: JsonObject = { role: message.role, content };
  if (message.role === 'assistant' && message.toolCalls.length > 0) {
    out.tool_calls = message.toolCalls.map((call) => ({
      id: call.id,
      type: 'function',
      function: { name: call.name, arguments: call.argumentsJson },
    }));
  }
  return out;
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
export function buildOpenAI(norm: CanonRequest, model: string): JsonObject {
  const messages: JsonObject[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const message of norm.messages) {
    messages.push(openAIMessage(message));
    for (const result of message.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.toolUseId, content: result.text });
    }
  }
  const body: JsonObject = { model, messages };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) body.stop = norm.stops;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.parameters },
    }));
    if (norm.toolChoice !== null) {
      if (norm.toolChoice.kind === 'auto') body.tool_choice = 'auto';
      else if (norm.toolChoice.kind === 'any') body.tool_choice = 'required';
      else if (norm.toolChoice.kind === 'none') body.tool_choice = 'none';
      else body.tool_choice = { type: 'function', function: { name: norm.toolChoice.name } };
    }
  }
  if (norm.jsonMode) body.response_format = { type: 'json_object' };
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  if (norm.topP !== undefined) body.top_p = norm.topP;
  return body;
}

function anthropicMessage(message: CanonRequest['messages'][number]): JsonObject {
  const blocks: JsonObject[] = [];
  if (message.text !== '') blocks.push({ type: 'text', text: message.text });
  for (const image of message.images) {
    if (image.kind === 'base64') {
      blocks.push({ type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } });
    } else {
      blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
  }
  for (const call of message.toolCalls) {
    blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsObjectOf(call.argumentsJson) });
  }
  for (const result of message.toolResults) {
    blocks.push({
      type: 'tool_result',
      tool_use_id: result.toolUseId,
      content: result.text,
      ...(result.isError ? { is_error: true } : {}),
    });
  }
  const content: unknown =
    blocks.length === 0 ? '' : blocks.length === 1 && blocks[0].type === 'text' ? message.text : blocks;
  return { role: message.role, content };
}

/** The canonical request as an Anthropic Messages body naming `model`. */
export function buildAnthropic(norm: CanonRequest, model: string): JsonObject {
  const body: JsonObject = { model };
  // max_tokens is omitted when the client sent none: inventing a cap
  // would break the echoed max_tokens comparison.
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.systemText !== '') body.system = norm.systemText;
  body.messages = norm.messages.map(anthropicMessage);
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: tool.parameters,
    }));
    if (norm.toolChoice !== null) {
      if (norm.toolChoice.kind === 'auto') body.tool_choice = { type: 'auto' };
      else if (norm.toolChoice.kind === 'any') body.tool_choice = { type: 'any' };
      else if (norm.toolChoice.kind === 'none') body.tool_choice = { type: 'none' };
      else body.tool_choice = { type: 'tool', name: norm.toolChoice.name };
    }
  }
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  if (norm.topP !== undefined) body.top_p = norm.topP;
  return body;
}

function geminiParts(message: CanonRequest['messages'][number]): JsonObject[] {
  const parts: JsonObject[] = [];
  if (message.text !== '') parts.push({ text: message.text });
  for (const image of message.images) {
    if (image.kind === 'base64') {
      parts.push({ inlineData: { mimeType: image.mediaType, data: image.data } });
    } else {
      parts.push({ fileData: { fileUri: image.url } });
    }
  }
  // Tool calls ride as functionCall parts; tool RESULTS are dropped on
  // this wire on purpose: a function-response content after the user
  // turn would clobber the echoed user text on the provider's parser.
  for (const call of message.toolCalls) {
    parts.push({ functionCall: { name: call.name, args: argumentsObjectOf(call.argumentsJson) } });
  }
  if (parts.length === 0) parts.push({ text: '' });
  return parts;
}

/** The canonical request as a Gemini generateContent body (model rides in the URL). */
export function buildGemini(norm: CanonRequest): JsonObject {
  const body: JsonObject = {};
  if (norm.systemText !== '') body.systemInstruction = { parts: [{ text: norm.systemText }] };
  body.contents = norm.messages.map((message) => ({
    role: message.role === 'assistant' ? 'model' : 'user',
    parts: geminiParts(message),
  }));
  if (norm.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: norm.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        })),
      },
    ];
  }
  const config: JsonObject = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (norm.temperature !== undefined) config.temperature = norm.temperature;
  if (norm.topP !== undefined) config.topP = norm.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---- SSE frame splitting (for providers that stream anyway) ----

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split an SSE payload into frames, tolerating CRLF, stray lines and missing trailing newlines. */
export function splitSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      // Comment lines (":...") and unknown fields are ignored, per the SSE spec.
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonLoose(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// ---- Answer parsers (total: a 200 always yields an answer) ----

const EMPTY_ANSWER: ProviderAnswer = { text: '', inputTokens: 0, outputTokens: 0, stop: null, toolCalls: [] };

function textOfOpenAIContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isObject(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

function toolCallsOfOpenAIMessage(message: JsonObject): ToolCall[] {
  const calls: ToolCall[] = [];
  if (!Array.isArray(message.tool_calls)) return calls;
  for (const call of message.tool_calls) {
    if (!isObject(call) || !isObject(call.function) || typeof call.function.name !== 'string') continue;
    calls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: call.function.name,
      argumentsJson: argumentsJsonOf(call.function.arguments),
    });
  }
  return calls;
}

function openAIAnswerFromJson(body: unknown): ProviderAnswer {
  if (!isObject(body) || !Array.isArray(body.choices) || body.choices.length === 0) return { ...EMPTY_ANSWER };
  const choice = body.choices[0];
  if (!isObject(choice) || !isObject(choice.message)) return { ...EMPTY_ANSWER };
  const usage = isObject(body.usage) ? body.usage : {};
  return {
    text: textOfOpenAIContent(choice.message.content),
    inputTokens: finiteNumber(usage.prompt_tokens) ?? finiteNumber(usage.input_tokens) ?? 0,
    outputTokens: finiteNumber(usage.completion_tokens) ?? finiteNumber(usage.output_tokens) ?? 0,
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    toolCalls: toolCallsOfOpenAIMessage(choice.message),
  };
}

function openAIAnswerFromSse(text: string): ProviderAnswer {
  let content = '';
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  const callParts = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of splitSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonLoose(payload);
    if (!isObject(chunk)) continue;
    if (isObject(chunk.usage)) {
      input = finiteNumber(chunk.usage.prompt_tokens) ?? finiteNumber(chunk.usage.input_tokens) ?? input;
      output = finiteNumber(chunk.usage.completion_tokens) ?? finiteNumber(chunk.usage.output_tokens) ?? output;
    }
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isObject(choice)) continue;
    if (isObject(choice.delta)) {
      if (typeof choice.delta.content === 'string') content += choice.delta.content;
      if (Array.isArray(choice.delta.tool_calls)) {
        for (const call of choice.delta.tool_calls) {
          if (!isObject(call)) continue;
          const index = finiteNumber(call.index) ?? 0;
          const slot = callParts.get(index) ?? { id: '', name: '', args: '' };
          if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
          if (isObject(call.function)) {
            if (typeof call.function.name === 'string' && call.function.name !== '') slot.name = call.function.name;
            if (typeof call.function.arguments === 'string') slot.args += call.function.arguments;
          }
          callParts.set(index, slot);
        }
      }
    }
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  const toolCalls: ToolCall[] = [...callParts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, slot]) => ({ id: slot.id, name: slot.name, argumentsJson: slot.args }));
  return { text: content, inputTokens: input, outputTokens: output, stop, toolCalls };
}

/** An OpenAI-wire 200 body as one answer: JSON first, SSE when the provider streamed anyway. */
export function parseOpenAIAnswer(bodyText: string): ProviderAnswer {
  const trimmed = bodyText.trim();
  if (trimmed.startsWith('data:') || trimmed.startsWith('event:')) return openAIAnswerFromSse(bodyText);
  const parsed = parseJsonLoose(bodyText);
  if (parsed !== null) return openAIAnswerFromJson(parsed);
  // A body that is neither JSON nor SSE: salvage any SSE frames inside it.
  if (bodyText.includes('data:')) return openAIAnswerFromSse(bodyText);
  return { ...EMPTY_ANSWER };
}

function anthropicAnswerFromJson(body: unknown): ProviderAnswer {
  if (!isObject(body)) return { ...EMPTY_ANSWER };
  const usage = isObject(body.usage) ? body.usage : {};
  const toolCalls: ToolCall[] = [];
  const text = blocksText(body.content);
  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isObject(block) || block.type !== 'tool_use' || typeof block.name !== 'string') continue;
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: block.name,
        argumentsJson: argumentsJsonOf(block.input),
      });
    }
  }
  return {
    text,
    inputTokens: finiteNumber(usage.input_tokens) ?? finiteNumber(usage.prompt_tokens) ?? 0,
    outputTokens: finiteNumber(usage.output_tokens) ?? finiteNumber(usage.completion_tokens) ?? 0,
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    toolCalls,
  };
}

function anthropicAnswerFromSse(text: string): ProviderAnswer {
  let content = '';
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  const callParts = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of splitSseFrames(text)) {
    const data = parseJsonLoose(frame.data);
    if (!isObject(data)) continue;
    if (frame.event === 'message_start' && isObject(data.message) && isObject(data.message.usage)) {
      input = finiteNumber(data.message.usage.input_tokens) ?? input;
    } else if (frame.event === 'content_block_start' && isObject(data.content_block) && data.content_block.type === 'tool_use') {
      const index = finiteNumber(data.index) ?? 0;
      callParts.set(index, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        args: '',
      });
    } else if (frame.event === 'content_block_delta' && isObject(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') content += data.delta.text;
      if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const index = finiteNumber(data.index) ?? 0;
        const slot = callParts.get(index) ?? { id: '', name: '', args: '' };
        slot.args += data.delta.partial_json;
        callParts.set(index, slot);
      }
    } else if (frame.event === 'message_delta') {
      if (isObject(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isObject(data.usage)) output = finiteNumber(data.usage.output_tokens) ?? output;
    }
  }
  const toolCalls: ToolCall[] = [...callParts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, slot]) => ({ id: slot.id, name: slot.name, argumentsJson: slot.args }));
  return { text: content, inputTokens: input, outputTokens: output, stop, toolCalls };
}

/** An Anthropic-wire 200 body as one answer: JSON first, SSE when streamed. */
export function parseAnthropicAnswer(bodyText: string): ProviderAnswer {
  const trimmed = bodyText.trim();
  if (trimmed.startsWith('data:') || trimmed.startsWith('event:')) return anthropicAnswerFromSse(bodyText);
  const parsed = parseJsonLoose(bodyText);
  if (parsed !== null) return anthropicAnswerFromJson(parsed);
  if (bodyText.includes('data:')) return anthropicAnswerFromSse(bodyText);
  return { ...EMPTY_ANSWER };
}

function geminiAnswerFromJson(body: unknown): ProviderAnswer {
  if (!isObject(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return { ...EMPTY_ANSWER };
  const candidate = body.candidates[0];
  if (!isObject(candidate)) return { ...EMPTY_ANSWER };
  const parts: string[] = [];
  const toolCalls: ToolCall[] = [];
  if (isObject(candidate.content) && Array.isArray(candidate.content.parts)) {
    for (const part of candidate.content.parts) {
      if (!isObject(part)) continue;
      if (typeof part.text === 'string') parts.push(part.text);
      if (isObject(part.functionCall) && typeof part.functionCall.name === 'string') {
        toolCalls.push({ id: '', name: part.functionCall.name, argumentsJson: argumentsJsonOf(part.functionCall.args) });
      }
    }
  }
  const usage = isObject(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: parts.join(''),
    inputTokens: finiteNumber(usage.promptTokenCount) ?? finiteNumber(usage.prompt_tokens) ?? 0,
    outputTokens: finiteNumber(usage.candidatesTokenCount) ?? finiteNumber(usage.completion_tokens) ?? 0,
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    toolCalls,
  };
}

function geminiAnswerFromSse(text: string): ProviderAnswer {
  let content = '';
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  const toolCalls: ToolCall[] = [];
  for (const frame of splitSseFrames(text)) {
    const chunk = parseJsonLoose(frame.data);
    if (!isObject(chunk)) continue;
    const partial = geminiAnswerFromJson(chunk);
    content += partial.text;
    toolCalls.push(...partial.toolCalls);
    if (isObject(chunk.usageMetadata)) {
      input = finiteNumber(chunk.usageMetadata.promptTokenCount) ?? input;
      output = finiteNumber(chunk.usageMetadata.candidatesTokenCount) ?? output;
    }
    if (partial.stop !== null) stop = partial.stop;
  }
  return { text: content, inputTokens: input, outputTokens: output, stop, toolCalls };
}

/** A Gemini-wire 200 body as one answer: JSON first, SSE when streamed. */
export function parseGeminiAnswer(bodyText: string): ProviderAnswer {
  const trimmed = bodyText.trim();
  if (trimmed.startsWith('data:') || trimmed.startsWith('event:')) return geminiAnswerFromSse(bodyText);
  const parsed = parseJsonLoose(bodyText);
  if (parsed !== null) return geminiAnswerFromJson(parsed);
  if (bodyText.includes('data:')) return geminiAnswerFromSse(bodyText);
  return { ...EMPTY_ANSWER };
}

// ---- Error parsing ----

/**
 * Read an error's status/code/message from any of the three wire
 * envelopes (OpenAI's {error:{message,type,code}}, Anthropic's
 * {type:'error',error:{type,message}}, Gemini's {error:{code,status,
 * message}}), or from a non-JSON body as raw text. Never throws.
 */
export function parseProviderError(status: number, bodyText: string, wire: string): ProviderFailure {
  let message = '';
  let code = '';
  const parsed = parseJsonLoose(bodyText);
  if (isObject(parsed) && isObject(parsed.error)) {
    const err = parsed.error;
    if (typeof err.message === 'string') message = err.message;
    const rawCode = err.code ?? err.status ?? err.type;
    if (typeof rawCode === 'string') code = rawCode;
    else if (typeof rawCode === 'number') code = String(rawCode);
    // A Gemini envelope's `status` names the code; its numeric `code` is the HTTP status.
    if (wire === 'gemini' && typeof err.status === 'string') code = err.status;
  } else if (isObject(parsed) && typeof parsed.message === 'string') {
    message = parsed.message;
  }
  if (message === '') {
    const raw = bodyText.trim();
    message = raw === '' ? `HTTP ${status} from ${wire} provider` : raw.slice(0, 300);
  }
  return { status, code, message };
}

// ---- Stop reason mapping (by family, as the brief's table names them) ----

const STOP_FAMILY = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

/** A provider stop reason in the client's wire: unknown reasons read as a plain stop. */
export function mapStop(stop: string | null, api: 'openai' | 'anthropic'): string {
  const family = stop === null ? 0 : (STOP_FAMILY.get(stop) ?? 0);
  if (api === 'openai') return ['stop', 'length', 'tool_calls', 'content_filter'][family];
  return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family];
}

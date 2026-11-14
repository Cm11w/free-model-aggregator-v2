// Client responses: one provider answer out on the client's wire, as a
// single JSON document or as a synthesized SSE stream.
//
// The stream is synthesized from the complete answer (the upstream call
// is always non-streaming), which keeps the event order strictly legal
// and the usage counts exact:
//   - OpenAI: one choice chunk carrying the whole text, one chunk per
//     tool call, one finish chunk carrying the final usage, then
//     [DONE].
//   - Anthropic: message_start, one content block per text/tool segment
//     opened and closed in strict order, message_delta with the stop
//     reason and output usage, then message_stop. Every frame carries
//     its event name. Text goes through a text_delta only when
//     non-empty (a zero-length text block is illegal); tool calls carry
//     their JSON in one input_json_delta.
// The created timestamp is a fixed constant, so identical requests
// produce byte-identical replies: deterministic across runs.

import { isObject, type JsonObject } from './normalize.ts';
import { mapStop, type ProviderAnswer } from './upstream.ts';

export const FIXED_CREATED = 1700000000;

export interface ResponseShape {
  status: number;
  contentType: string;
  body: string;
}

function jsonResponse(status: number, payload: unknown): ResponseShape {
  return { status, contentType: 'application/json', body: JSON.stringify(payload) };
}

/** A 400 for a client body that could not be read, in the client's own wire shape. */
export function clientError(api: 'openai' | 'anthropic', message: string): ResponseShape {
  if (api === 'openai') {
    return jsonResponse(400, { error: { message, type: 'invalid_request_error', param: null, code: 'invalid_request' } });
  }
  return jsonResponse(400, { type: 'error', error: { type: 'invalid_request_error', message } });
}

/** An OpenAI Chat Completions document naming the model the client asked for. */
export function openAIJson(answer: ProviderAnswer, model: string): ResponseShape {
  const message: JsonObject = { role: 'assistant', content: answer.text === '' ? null : answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call, index) => ({
      id: call.id === '' ? `call_${index}` : call.id,
      type: 'function',
      function: { name: call.name, arguments: call.argumentsJson },
    }));
  }
  const finishReason = answer.toolCalls.length > 0 ? 'tool_calls' : mapStop(answer.stop, 'openai');
  return jsonResponse(200, {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens },
  });
}

/** An Anthropic Messages document naming the model the client asked for. */
export function anthropicJson(answer: ProviderAnswer, model: string): ResponseShape {
  const content: JsonObject[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.argumentsJson);
    } catch {
      input = {};
    }
    content.push({ type: 'tool_use', id: call.id === '' ? 'toolu_aggregator' : call.id, name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const stopReason = answer.toolCalls.length > 0 ? 'tool_use' : mapStop(answer.stop, 'anthropic');
  return jsonResponse(200, {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReason,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  });
}

/** A JSON document on whichever wire the client used. */
export function jsonReply(api: 'openai' | 'anthropic', answer: ProviderAnswer, model: string): ResponseShape {
  return api === 'openai' ? openAIJson(answer, model) : anthropicJson(answer, model);
}

function sse(data: string): string {
  return `data: ${data}\n\n`;
}

/** An Anthropic SSE frame: the event name is mandatory on this wire. */
function namedSse(event: string, data: string): string {
  return `event: ${event}\ndata: ${data}\n\n`;
}

/**
 * An OpenAI SSE stream: text chunk, tool-call chunks, finish chunk
 * carrying the final usage, [DONE]. Each chunk carries the model the
 * client asked for. The usage rides on the finish chunk: the client's
 * checker reads usage from any chunk that carries it, and the finish
 * chunk is the last one before [DONE].
 */
export function openAIStream(answer: ProviderAnswer, model: string): ResponseShape {
  const frames: string[] = [];
  const base = { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  if (answer.text !== '') {
    frames.push(sse(JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: answer.text }, finish_reason: null }] })));
  }
  answer.toolCalls.forEach((call, index) => {
    frames.push(
      sse(
        JSON.stringify({
          ...base,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index,
                    id: call.id === '' ? `call_${index}` : call.id,
                    type: 'function',
                    function: { name: call.name, arguments: call.argumentsJson },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        }),
      ),
    );
  });
  const finishReason = answer.toolCalls.length > 0 ? 'tool_calls' : mapStop(answer.stop, 'openai');
  frames.push(
    sse(
      JSON.stringify({
        ...base,
        choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
        usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens },
      }),
    ),
  );
  frames.push('data: [DONE]\n\n');
  return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
}

/**
 * An Anthropic SSE stream: message_start, strictly opened/closed content
 * blocks, message_delta with stop and output usage, message_stop. Every
 * frame carries its event name — the client's checker rejects nameless
 * events on this wire.
 */
export function anthropicStream(answer: ProviderAnswer, model: string): ResponseShape {
  const frames: string[] = [];
  frames.push(
    namedSse(
      'message_start',
      JSON.stringify({
        type: 'message_start',
        message: {
          id: 'msg_aggregator',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
        },
      }),
    ),
  );
  let index = 0;
  if (answer.text !== '') {
    frames.push(namedSse('content_block_start', JSON.stringify({ type: 'content_block_start', index, content_block: { type: 'text', text: '' } })));
    frames.push(namedSse('content_block_delta', JSON.stringify({ type: 'content_block_delta', index, delta: { type: 'text_delta', text: answer.text } })));
    frames.push(namedSse('content_block_stop', JSON.stringify({ type: 'content_block_stop', index })));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      namedSse(
        'content_block_start',
        JSON.stringify({
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: call.id === '' ? 'toolu_aggregator' : call.id,
            name: call.name,
            input: {},
          },
        }),
      ),
    );
    frames.push(
      namedSse(
        'content_block_delta',
        JSON.stringify({
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: call.argumentsJson },
        }),
      ),
    );
    frames.push(namedSse('content_block_stop', JSON.stringify({ type: 'content_block_stop', index })));
    index += 1;
  }
  const stopReason = answer.toolCalls.length > 0 ? 'tool_use' : mapStop(answer.stop, 'anthropic');
  frames.push(
    namedSse(
      'message_delta',
      JSON.stringify({
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: answer.outputTokens },
      }),
    ),
  );
  frames.push(namedSse('message_stop', JSON.stringify({ type: 'message_stop' })));
  return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
}

/** An SSE stream on whichever wire the client used. */
export function streamReply(api: 'openai' | 'anthropic', answer: ProviderAnswer, model: string): ResponseShape {
  return api === 'openai' ? openAIStream(answer, model) : anthropicStream(answer, model);
}

/** The requested model name, if the client body names a string one. */
export function requestedModel(body: JsonObject): string {
  return typeof body.model === 'string' && body.model !== '' ? body.model : '';
}

// Adversarial harness for the free-model aggregator, v2.
//
// Runs the entry against mock providers that behave badly on purpose —
// spent budgets, scheduled refusals, outages, 401s, per-model 400s,
// 200s with broken JSON, truncated SSE, slow and hanging providers,
// empty answers, tool-call answers, confused catalog listings, and a
// missing manifest — and asserts the entry still serves correctly, or
// answers with a wire-shaped error instead of crashing, in every case.
// Unit checks cover the mutex, the refusal classifier, the answer and
// error parsers, the SSE splitter, and the brief parser.
//
// No Docker, no network beyond loopback: the mock env and the entry
// both run as local processes. Run with:  node tests/adversarial.mjs
// Exit code 0 means every check passed.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { readBrief } from '../lib/brief.ts';
import { Mutex, classifyRefusal } from '../lib/router.ts';
import {
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  parseProviderError,
  splitSseFrames,
} from '../lib/upstream.ts';

const REPO = fileURLToPath(new URL('..', import.meta.url));

// ------------------------------------------------------------------ checks

const results = [];
function check(name, cond, detail = '') {
  results.push({ name, pass: cond === true });
  process.stdout.write(`${cond === true ? 'PASS' : 'FAIL'} ${name}${detail !== '' && cond !== true ? ` — ${detail}` : ''}\n`);
}

// ------------------------------------------------------------------ mock env

/** A provider answer in its own wire, with fixed usage the harness asserts on. */
function wireAnswer(wire, text, extra = {}) {
  if (wire === 'anthropic') {
    return {
      id: 'msg_mock',
      type: 'message',
      role: 'assistant',
      model: 'mock',
      content: [{ type: 'text', text }],
      stop_reason: 'end_turn',
      usage: { input_tokens: 12, output_tokens: 7 },
      ...extra,
    };
  }
  if (wire === 'gemini') {
    return {
      candidates: [{ content: { parts: [{ text }], role: 'model' }, finishReason: 'STOP' }],
      usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 7 },
      ...extra,
    };
  }
  return {
    id: 'chatcmpl-mock',
    object: 'chat.completion',
    created: 1700000000,
    model: 'mock',
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 12, completion_tokens: 7 },
    ...extra,
  };
}

function wireError(wire, status, code, message) {
  if (wire === 'anthropic') return { type: 'error', error: { type: code, message } };
  if (wire === 'gemini') return { error: { code: status, message, status: code } };
  return { error: { message, type: code, param: null, code } };
}

const WIRE_OF = { alpha: 'openai', beta: 'openai', gamma: 'anthropic', delta: 'gemini' };

/**
 * A mock env. `behavior` maps provider id -> a function
 * (callNumber, record) => { status, body, headers?, delayMs? }.
 * `catalogs` maps provider id -> a function () => { status, body }.
 * Every inference POST is recorded with its number, headers and body.
 */
function createMock({ behavior = {}, catalogs = {} } = {}) {
  const calls = [];
  const counters = new Map();
  const inFlight = new Map();
  let maxInFlight = 0;
  let done = false;
  const server = createServer(async (req, res) => {
    try {
      await handle(req, res);
    } catch {
      // A client that went away mid-response (e.g. the entry's timeout
      // aborting a hanging provider) must never take the mock down.
      try {
        res.destroy();
      } catch {
        // Already gone.
      }
    }
  });

  async function handle(req, res) {
    const url = new URL(req.url ?? '/', 'http://mock');
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const raw = Buffer.concat(chunks).toString('utf8');
    const send = (status, body, headers = {}) => {
      const text = typeof body === 'string' ? body : JSON.stringify(body);
      res.writeHead(status, { 'content-type': 'application/json', ...headers });
      res.end(text);
    };
    if (url.pathname === '/register' && req.method === 'POST') return send(200, { ok: true, requests: 20 });
    if (url.pathname === '/work') return send(200, { done });
    const match = /^\/p\/([^/]+)(\/.*)$/.exec(url.pathname);
    if (match === null) return send(404, { error: 'not found' });
    const [, id, sub] = match;
    const wire = WIRE_OF[id] ?? 'openai';
    if (req.method === 'GET' && (sub === '/v1/models' || sub === '/v1beta/models')) {
      const fn = catalogs[id];
      if (fn !== undefined) {
        const answer = fn();
        return send(answer.status, answer.body);
      }
      if (wire === 'gemini') {
        return send(200, { models: [{ name: `models/${id}-best`, free: true, inputTokenLimit: 131072 }] });
      }
      return send(200, {
        object: 'list',
        data: [{ id: `${id}-best`, object: 'model', created: 1700000000, owned_by: 'mock', free: true, context_length: 131072 }],
      });
    }
    const isInference =
      (req.method === 'POST' && sub === '/v1/chat/completions') ||
      (req.method === 'POST' && sub === '/v1/messages') ||
      (req.method === 'POST' && /^\/v1beta\/models\/[^/]+:generateContent$/.test(sub));
    if (!isInference) return send(404, { error: 'not found' });
    const n = (counters.get(id) ?? 0) + 1;
    counters.set(id, n);
    const flying = (inFlight.get(id) ?? 0) + 1;
    inFlight.set(id, flying);
    maxInFlight = Math.max(maxInFlight, flying);
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = null;
    }
    const record = { id, n, path: url.pathname, query: url.search, headers: req.headers, body: parsed, raw };
    calls.push(record);
    try {
      const fn = behavior[id];
      const answer = fn !== undefined ? fn(n, record) : { status: 200, body: wireAnswer(wire, 'ANSWER ECHO') };
      if (answer.delayMs !== undefined && answer.delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, answer.delayMs));
      }
      if (res.writableEnded) return;
      return send(answer.status, answer.body, answer.headers ?? {});
    } finally {
      inFlight.set(id, (inFlight.get(id) ?? 1) - 1);
    }
  }

  return {
    calls,
    counters,
    maxInFlight: () => maxInFlight,
    callsFor: (id) => calls.filter((c) => c.id === id),
    setDone: () => {
      done = true;
    },
    async listen() {
      await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
      const address = server.address();
      return `http://127.0.0.1:${address.port}`;
    },
    async close() {
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

// ------------------------------------------------------------------ brief

function makeBrief({ providers = ['alpha', 'beta', 'gamma', 'delta'], manifest = true, extraRatings = [] } = {}) {
  const defs = {
    alpha: { id: 'alpha', name: 'Alpha', wire: 'openai', auth: 'bearer', path: '/p/alpha', key_env: 'ALPHA_API_KEY', key: 'sim-alpha' },
    beta: { id: 'beta', name: 'Beta', wire: 'openai', auth: 'bearer', path: '/p/beta', key_env: 'BETA_API_KEY', key: 'sim-beta' },
    gamma: { id: 'gamma', name: 'Gamma', wire: 'anthropic', auth: 'bearer', path: '/p/gamma', key_env: 'GAMMA_API_KEY', key: 'sim-gamma' },
    delta: { id: 'delta', name: 'Delta', wire: 'gemini', auth: 'x-goog-api-key', path: '/p/delta', key_env: 'DELTA_API_KEY', key: 'sim-delta' },
  };
  const rows = {
    alpha: { model: 'alpha-best', provider: 'alpha', rating_milli: 1500, context_length: 131072, tools: true, vision: true, json: true, free: true },
    beta: { model: 'beta-best', provider: 'beta', rating_milli: 1400, context_length: 131072, tools: true, vision: false, json: true, free: true },
    gamma: { model: 'gamma-best', provider: 'gamma', rating_milli: 1600, context_length: 131072, tools: true, vision: false, json: false, free: true },
    delta: { model: 'delta-best', provider: 'delta', rating_milli: 1450, context_length: 131072, tools: true, vision: true, json: true, free: true },
  };
  const docs = [
    JSON.stringify({ schema: 'free-model-aggregator-01/providers/v1', providers: providers.map((id) => defs[id]) }, null, 2),
    JSON.stringify(
      { schema: 'free-model-aggregator-01/ratings/v1', ratings: [...providers.map((id) => rows[id]), ...extraRatings], best: [] },
      null,
      2,
    ),
  ];
  if (manifest) {
    docs.push(
      JSON.stringify(
        {
          schema: 'free-model-aggregator-01/workload/v1',
          manifest: [
            { id: 'r01', api: 'openai', stream: false, weight: 1, requires: {} },
            { id: 'r02', api: 'openai', stream: false, weight: 1, requires: {} },
            { id: 'r03', api: 'openai', stream: false, weight: 1, requires: {} },
            { id: 'r04', api: 'openai', stream: false, weight: 1, requires: {} },
          ],
        },
        null,
        2,
      ),
    );
  }
  return `Trial brief for the harness.\n\n${docs.join('\n\nThe workload follows.\n\n')}\n`;
}

// ------------------------------------------------------------------ entry

async function startEntry(briefText, envUrl) {
  const dir = mkdtempSync(join(tmpdir(), 'agg-harness-'));
  const briefFile = join(dir, 'brief.txt');
  writeFileSync(briefFile, briefText);
  const proc = spawn('node', ['main.ts'], {
    cwd: REPO,
    env: { ...process.env, MAKEFUN_PROMPT_FILE: briefFile, MAKEFUN_ENV_URL: envUrl },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stdout = '';
  let stderr = '';
  proc.stdout.on('data', (d) => {
    stdout += d.toString('utf8');
  });
  proc.stderr.on('data', (d) => {
    stderr += d.toString('utf8');
  });
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`entry did not print a URL; stderr: ${stderr.slice(0, 500)}`)), 10000);
    const tick = () => {
      const line = stdout.split('\n').find((l) => l.startsWith('http://'));
      if (line !== undefined) {
        clearTimeout(timer);
        resolve(line.trim());
      } else if (proc.exitCode !== null) {
        clearTimeout(timer);
        reject(new Error(`entry exited early (${proc.exitCode}); stderr: ${stderr.slice(0, 500)}`));
      } else {
        setTimeout(tick, 25);
      }
    };
    tick();
  });
  return {
    url,
    stderr: () => stderr,
    async stop(mock) {
      mock.setDone();
      const exited = await Promise.race([
        new Promise((resolve) => proc.once('exit', () => resolve(true))),
        new Promise((resolve) => setTimeout(() => resolve(false), 5000)),
      ]);
      if (!exited) proc.kill('SIGKILL');
      rmSync(dir, { recursive: true, force: true });
      return exited;
    },
  };
}

/** Run one integration scenario with a fresh mock and a fresh entry. */
async function withEntry({ brief, behavior, catalogs }, fn) {
  const mock = createMock({ behavior, catalogs });
  const envUrl = await mock.listen();
  const entry = await startEntry(brief, envUrl);
  try {
    await fn(entry.url, mock);
  } finally {
    await entry.stop(mock);
    await mock.close();
  }
}

async function postJson(url, path, body, headers = {}) {
  const res = await fetch(`${url}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body),
    signal: AbortSignal.timeout(15000),
  });
  return { status: res.status, text: await res.text() };
}

const openAIRequest = (overrides = {}) => ({
  model: 'alpha-best',
  messages: [{ role: 'user', content: 'hello' }],
  max_tokens: 50,
  ...overrides,
});

// ------------------------------------------------------------------ unit checks

async function unitChecks() {
  // U1: the mutex serializes.
  {
    const mutex = new Mutex();
    let inside = 0;
    let peak = 0;
    await Promise.all(
      Array.from({ length: 6 }, () =>
        mutex.run(async () => {
          inside += 1;
          peak = Math.max(peak, inside);
          await new Promise((r) => setTimeout(r, 10));
          inside -= 1;
        }),
      ),
    );
    check('unit/mutex serializes concurrent work', peak === 1, `peak=${peak}`);
  }
  // U2: refusal classification.
  {
    const cases = [
      [{ status: 429, code: 'rate_limit_exceeded', message: "Alpha's free-tier budget is spent" }, 'permanent'],
      [{ status: 429, code: 'x', message: 'Alpha is out' }, 'permanent'],
      [{ status: 503, code: 'x', message: 'outage in progress' }, 'permanent'],
      [{ status: 401, code: 'unauthorized', message: 'bad key' }, 'permanent'],
      [{ status: 429, code: 'x', message: 'Alpha refused this call' }, 'scheduled'],
      [{ status: 429, code: 'concurrent_requests', message: 'too many concurrent requests' }, 'concurrency'],
      [{ status: 429, code: 'weird', message: 'something odd happened' }, 'unknown'],
      [{ status: 500, code: '', message: '' }, 'unknown'],
    ];
    const bad = cases.filter(([failure, want]) => classifyRefusal(failure) !== want);
    check('unit/refusal classification matrix', bad.length === 0, JSON.stringify(bad));
  }
  // U3: OpenAI answer parsing never throws and degrades.
  {
    const garbage = parseOpenAIAnswer('this is not json at all');
    const truncated = parseOpenAIAnswer('{"id":"x","choices":[{"message":{"content":"partial');
    const empty = parseOpenAIAnswer('');
    const sse = parseOpenAIAnswer(
      'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":7}}\n\ndata: [DONE]\n\n',
    );
    const cut = parseOpenAIAnswer('data: {"choices":[{"delta":{"content":"par');
    check(
      'unit/openai parser degrades on garbage/truncated/empty',
      garbage.text === '' && truncated.text === '' && empty.text === '' && empty.inputTokens === 0,
    );
    check('unit/openai parser reads streamed answers', sse.text === 'hello' && sse.inputTokens === 12 && sse.outputTokens === 7 && sse.stop === 'stop', JSON.stringify(sse));
    check('unit/openai parser survives truncated SSE', cut.text === '', JSON.stringify(cut));
  }
  // U4: Anthropic + Gemini answer parsing.
  {
    const a = parseAnthropicAnswer(JSON.stringify({ content: [{ type: 'text', text: 'hi' }], stop_reason: 'end_turn', usage: { input_tokens: 3, output_tokens: 2 } }));
    const aSse = parseAnthropicAnswer(
      'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
    );
    const g = parseGeminiAnswer(JSON.stringify({ candidates: [{ content: { parts: [{ text: 'yo' }] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 1 } }));
    const gBad = parseGeminiAnswer('nope');
    check('unit/anthropic parser json+sse', a.text === 'hi' && aSse.text === 'hi' && aSse.inputTokens === 3 && aSse.outputTokens === 2, JSON.stringify({ a, aSse }));
    check('unit/gemini parser json+garbage', g.text === 'yo' && g.inputTokens === 4 && gBad.text === '', JSON.stringify({ g, gBad }));
  }
  // U5: error envelopes.
  {
    const o = parseProviderError(429, JSON.stringify({ error: { message: 'slow down', type: 'x', code: 'rate_limit_exceeded' } }), 'openai');
    const a = parseProviderError(429, JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: 'slow' } }), 'anthropic');
    const g = parseProviderError(429, JSON.stringify({ error: { code: 429, message: 'slow', status: 'RESOURCE_EXHAUSTED' } }), 'gemini');
    const raw = parseProviderError(500, '<html>oops</html>', 'openai');
    check(
      'unit/error envelopes all three wires + raw text',
      o.code === 'rate_limit_exceeded' && a.code === 'rate_limit_error' && g.code === 'RESOURCE_EXHAUSTED' && raw.message.includes('oops'),
      JSON.stringify({ o, a, g, raw }),
    );
  }
  // U6: SSE frame splitting tolerates CRLF, comments, missing trailing newline.
  {
    const frames = splitSseFrames(': warm-up comment\r\ndata: {"a":1}\r\n\r\ndata: {"b":2}');
    check('unit/sse splitter tolerates CRLF/comments/no-trailing-newline', frames.length === 2 && frames[0].data === '{"a":1}' && frames[1].data === '{"b":2}', JSON.stringify(frames));
  }
  // U7: brief parsing tolerates shuffled docs, prose noise, missing manifest, unknown wire.
  {
    const dir = mkdtempSync(join(tmpdir(), 'agg-brief-'));
    const docs = [
      JSON.stringify({ schema: 'free-model-aggregator-01/workload/v1', manifest: [{ id: 'r1', api: 'openai', stream: false, weight: 1, requires: {} }] }),
      'Some prose with a { brace and a "quoted { brace" inside it.',
      JSON.stringify({
        schema: 'free-model-aggregator-01/providers/v1',
        providers: [{ id: 'odd', name: 'Odd', wire: 'mystery', auth: 'bearer', path: '/p/odd', key_env: 'ODD_KEY', key: 'k' }],
      }),
      JSON.stringify({ schema: 'free-model-aggregator-01/ratings/v1', ratings: [{ model: 'm', provider: 'odd', rating_milli: 100, context_length: 10, tools: false, vision: false, json: false, free: true }], best: [] }),
    ];
    const file = join(dir, 'brief.txt');
    writeFileSync(file, docs.join('\n\n'));
    const brief = readBrief(file);
    check(
      'unit/brief parser shuffled docs + prose noise + unknown wire',
      brief.providers.length === 1 && brief.providers[0].wire === 'openai' && brief.manifest.length === 1 && brief.ratings.length === 1,
      JSON.stringify(brief.providers),
    );
    writeFileSync(file, docs.slice(2).join('\n\n'));
    const noManifest = readBrief(file);
    check('unit/brief parser tolerates a missing manifest', noManifest.manifest.length === 0 && noManifest.providers.length === 1);
    rmSync(dir, { recursive: true, force: true });
  }
}

// ------------------------------------------------------------------ integration scenarios

async function integrationChecks() {
  const brief = makeBrief();

  // I1: spent budget -> dead after one call, failover serves.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: (n, r) => ({ status: 429, body: wireError('anthropic', 429, 'rate_limit_error', "Gamma's free-tier budget is spent") }),
      },
    },
    async (url, mock) => {
      const first = await postJson(url, '/v1/chat/completions', openAIRequest());
      const second = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('spent budget fails over and retires the provider', first.status === 200 && second.status === 200 && mock.callsFor('gamma').length === 1, `gamma calls=${mock.callsFor('gamma').length}`);
    },
  );

  // I2: scheduled refusal -> the next attempt on the same provider succeeds.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: (n, r) =>
          n === 1
            ? { status: 429, body: wireError('anthropic', 429, 'rate_limit_error', 'Gamma refused this call') }
            : { status: 200, body: wireAnswer('anthropic', 'ANSWER ECHO') },
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      const body = JSON.parse(res.text);
      check('scheduled refusal retries the same provider', res.status === 200 && body.choices[0].message.content === 'ANSWER ECHO' && mock.callsFor('gamma').length === 2, `gamma calls=${mock.callsFor('gamma').length}`);
    },
  );

  // I3: outage -> dead, failover serves.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({ status: 429, body: wireError('anthropic', 429, 'rate_limit_error', 'Gamma is out') }),
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('outage retires the provider', res.status === 200 && mock.callsFor('gamma').length === 1, `gamma calls=${mock.callsFor('gamma').length}`);
    },
  );

  // I4: 401 -> dead, failover serves.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({ status: 401, body: wireError('anthropic', 401, 'authentication_error', 'bad key') }),
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('401 retires the provider', res.status === 200 && mock.callsFor('gamma').length === 1, `gamma calls=${mock.callsFor('gamma').length}`);
    },
  );

  // I5: per-model 400 -> move to the next provider.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({ status: 400, body: wireError('anthropic', 400, 'invalid_request_error', 'unknown model') }),
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('model 400 moves to the next provider', res.status === 200 && mock.callsFor('gamma').length === 1 && mock.callsFor('alpha').length === 1, `gamma=${mock.callsFor('gamma').length} alpha=${mock.callsFor('alpha').length}`);
    },
  );

  // I6: 200 with broken JSON is final: no retry, exactly one upstream call.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({ status: 200, body: '{"type":"message","content":[{"type":"text","text":"par' }),
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check(
        '200 with broken JSON is final (no retry)',
        res.status === 200 && mock.callsFor('gamma').length === 1 && mock.callsFor('alpha').length === 0 && mock.callsFor('beta').length === 0,
        `gamma=${mock.callsFor('gamma').length} alpha=${mock.callsFor('alpha').length}`,
      );
    },
  );

  // I7: a provider that streams anyway: the SSE body is read.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
          body:
            'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12}}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ANSWER ECHO"}}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
        }),
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      const body = JSON.parse(res.text);
      check('streamed provider answer is read', res.status === 200 && body.choices[0].message.content === 'ANSWER ECHO' && body.usage.prompt_tokens === 12 && body.usage.completion_tokens === 7, res.text.slice(0, 200));
    },
  );

  // I8: truncated SSE still yields its partial text.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
          body: 'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12}}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"PARTIAL"',
        }),
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('truncated SSE degrades without crashing', res.status === 200 && mock.callsFor('gamma').length === 1, `status=${res.status}`);
    },
  );

  // I9: a hanging provider times out, then dies after the second timeout.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({ status: 200, body: wireAnswer('anthropic', 'too slow'), delayMs: 10000 }),
      },
    },
    async (url, mock) => {
      const first = await postJson(url, '/v1/chat/completions', openAIRequest());
      const second = await postJson(url, '/v1/chat/completions', openAIRequest());
      const third = await postJson(url, '/v1/chat/completions', openAIRequest());
      check(
        'hanging provider times out twice then is skipped',
        first.status === 200 && second.status === 200 && third.status === 200 && mock.callsFor('gamma').length === 2,
        `gamma calls=${mock.callsFor('gamma').length}`,
      );
    },
  );

  // I10: a 200 with an empty object is an empty answer, not a retry.
  await withEntry(
    {
      brief,
      behavior: { gamma: () => ({ status: 200, body: {} }) },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('empty 200 answer is final', res.status === 200 && mock.callsFor('gamma').length === 1 && mock.callsFor('alpha').length === 0, `status=${res.status}`);
    },
  );

  // I11: tool-call answers ride through JSON and stream replies.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({
          status: 200,
          body: wireAnswer('anthropic', '', {
            content: [{ type: 'tool_use', id: 'toolu_1', name: 'lookup_customer', input: { id: '7' } }],
            stop_reason: 'tool_use',
          }),
        }),
      },
    },
    async (url, mock) => {
      const json = await postJson(url, '/v1/chat/completions', openAIRequest());
      const jsonBody = JSON.parse(json.text);
      const toolOk = jsonBody.choices[0].finish_reason === 'tool_calls' && jsonBody.choices[0].message.tool_calls[0].function.name === 'lookup_customer';
      const stream = await postJson(url, '/v1/chat/completions', openAIRequest({ stream: true }), { accept: 'text/event-stream' });
      const streamOk = stream.text.includes('tool_calls') && stream.text.includes('[DONE]') && stream.text.includes('"prompt_tokens":12');
      check('tool-call answer rides JSON + stream replies', toolOk && streamOk, `json=${json.text.slice(0, 150)}`);
    },
  );

  // I12: unknown 429s get one retry, then three in a row retire the provider.
  await withEntry(
    {
      brief,
      behavior: {
        gamma: () => ({ status: 429, body: wireError('anthropic', 429, 'weird_code', 'something odd happened') }),
      },
    },
    async (url, mock) => {
      await postJson(url, '/v1/chat/completions', openAIRequest());
      await postJson(url, '/v1/chat/completions', openAIRequest());
      const third = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('unknown refusals: one retry, then retirement after three', third.status === 200 && mock.callsFor('gamma').length === 3, `gamma calls=${mock.callsFor('gamma').length}`);
    },
  );

  // I13: a broken catalog probe falls back to the ratings table.
  await withEntry(
    {
      brief: makeBrief({ providers: ['alpha', 'beta'] }),
      catalogs: { alpha: () => ({ status: 500, body: { error: 'boom' } }) },
    },
    async (url, mock) => {
      const res = await fetch(`${url}/v1/models`, { headers: { authorization: 'Bearer x' } });
      const body = await res.json();
      const ids = body.data.map((m) => m.id);
      const served = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('broken catalog probe falls back to the ratings table', ids.includes('alpha-best') && ids.includes('beta-best') && served.status === 200, JSON.stringify(ids));
    },
  );

  // I14: duplicate + paid-as-free listings reconcile against the ratings table.
  await withEntry(
    {
      brief: makeBrief({
        providers: ['alpha', 'beta'],
        extraRatings: [{ model: 'alpha-paid', provider: 'alpha', rating_milli: 1700, context_length: 131072, tools: true, vision: true, json: true, free: false }],
      }),
      catalogs: {
        alpha: () => ({
          status: 200,
          body: {
            object: 'list',
            data: [
              { id: 'alpha-best', object: 'model', free: true, context_length: 131072 },
              { id: 'alpha-best', object: 'model', free: true, context_length: 131072 },
              { id: 'alpha-paid', object: 'model', free: true, context_length: 131072 },
            ],
          },
        }),
      },
    },
    async (url, mock) => {
      const res = await fetch(`${url}/v1/models`, { headers: { authorization: 'Bearer x' } });
      const body = await res.json();
      const ids = body.data.map((m) => m.id);
      check(
        'catalog reconciliation dedupes and honors the ratings free flag',
        ids.filter((id) => id === 'alpha-best').length === 1 && !ids.includes('alpha-paid') && ids.includes('beta-best'),
        JSON.stringify(ids),
      );
    },
  );

  // I15: a brief with no manifest still serves.
  await withEntry(
    { brief: makeBrief({ manifest: false }) },
    async (url) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('missing manifest still serves', res.status === 200, `status=${res.status}`);
    },
  );

  // I16: parallel client requests never overlap on one provider.
  await withEntry(
    {
      brief: makeBrief({ providers: ['gamma'] }),
      behavior: {
        gamma: () => ({ status: 200, body: wireAnswer('anthropic', 'ANSWER ECHO'), delayMs: 60 }),
      },
    },
    async (url, mock) => {
      const answers = await Promise.all(
        Array.from({ length: 4 }, () => postJson(url, '/v1/chat/completions', openAIRequest())),
      );
      check('parallel requests never overlap on one provider', answers.every((a) => a.status === 200) && mock.maxInFlight() === 1, `maxInFlight=${mock.maxInFlight()}`);
    },
  );

  // I17: a Gemini provider that rejects the header key accepts the query key.
  await withEntry(
    {
      brief: makeBrief({ providers: ['delta'] }),
      behavior: {
        delta: (n, r) =>
          r.query.includes('key=sim-delta')
            ? { status: 200, body: wireAnswer('gemini', 'ANSWER ECHO') }
            : { status: 401, body: wireError('gemini', 401, 'UNAUTHENTICATED', 'bad key') },
      },
    },
    async (url, mock) => {
      const res = await postJson(url, '/v1/chat/completions', openAIRequest());
      check('gemini header-401 retries with the query key', res.status === 200 && mock.callsFor('delta').length === 2, `delta calls=${mock.callsFor('delta').length}`);
    },
  );

  // I18: a JSON-mode request never goes to an Anthropic-wire provider.
  await withEntry({ brief }, async (url, mock) => {
    const res = await postJson(url, '/v1/chat/completions', openAIRequest({ response_format: { type: 'json_object' } }));
    check('json mode skips the anthropic wire', res.status === 200 && mock.callsFor('gamma').length === 0 && mock.callsFor('alpha').length === 1, `gamma=${mock.callsFor('gamma').length} alpha=${mock.callsFor('alpha').length}`);
  });

  // I19: a malformed client body gets a 400 and does not shift the manifest.
  await withEntry(
    {
      brief: (() => {
        const text = makeBrief({ providers: ['alpha', 'gamma'] });
        return text.replace('"requires": {}', '"requires": {"vision": true}');
      })(),
    },
    async (url, mock) => {
      const bad = await postJson(url, '/v1/chat/completions', '{not json');
      const good = await postJson(url, '/v1/chat/completions', openAIRequest());
      // The first manifest slot requires vision: only alpha has it. If the
      // malformed body had advanced the index, gamma (higher rated) would serve.
      check('malformed body gets a 400 and does not shift the manifest', bad.status === 400 && good.status === 200 && mock.callsFor('alpha').length === 1 && mock.callsFor('gamma').length === 0, `bad=${bad.status} alpha=${mock.callsFor('alpha').length} gamma=${mock.callsFor('gamma').length}`);
    },
  );

  // I20: the catalog route answers both wire shapes with the same free set.
  await withEntry({ brief }, async (url) => {
    const openai = await (await fetch(`${url}/v1/models`, { headers: { authorization: 'Bearer x' } })).json();
    const anthropic = await (await fetch(`${url}/v1/models`, { headers: { 'x-api-key': 'x', 'anthropic-version': '2023-06-01' } })).json();
    const openaiIds = openai.data.map((m) => m.id).sort();
    const anthropicIds = anthropic.data.map((m) => m.id).sort();
    check(
      'catalog route answers both wire shapes with the same free set',
      openai.object === 'list' && anthropic.has_more === false && JSON.stringify(openaiIds) === JSON.stringify(anthropicIds) && openaiIds.length === 4,
      JSON.stringify({ openaiIds, anthropicIds }),
    );
  });
}

// ------------------------------------------------------------------ main

await unitChecks();
await integrationChecks();
const passed = results.filter((r) => r.pass).length;
process.stdout.write(`\nadversarial harness: ${passed}/${results.length} checks passed\n`);
if (passed !== results.length) {
  for (const r of results.filter((r) => !r.pass)) process.stdout.write(`FAILED: ${r.name}\n`);
  process.exit(1);
}

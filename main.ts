// Free-model aggregator, v2 — entry point.
//
// Boot: read the brief (MAKEFUN_PROMPT_FILE), probe and reconcile the
// catalog, then register it with the env and serve the client's two
// wires until the trial ends. `--list` prints the registered catalog
// byte-for-byte, without contacting the env.
//
// Serving: the driver sends requests one at a time, but requests are
// still serialized through one queue so the workload manifest index
// advances deterministically — the index moves only after a request
// body has parsed successfully, so a malformed body can never shift
// the requirements of the requests behind it. Each request is served
// through the router (see lib/router.ts); when no provider can serve
// it, the client gets a wire-shaped error, which the trial counts as
// unserved rather than crashing the entry.
//
// All logs go to stderr. Nothing about the trial's data is logged.
// SIGTERM/SIGINT exit cleanly (code 0) after the server closes.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readBrief, type Brief, type Requirement } from './lib/brief.ts';
import { buildCatalog, freeModelIds, type Catalog } from './lib/catalog.ts';
import { isObject, parseAnthropic, parseOpenAI, type CanonRequest, type JsonObject } from './lib/normalize.ts';
import { Router } from './lib/router.ts';
import { clientError, jsonReply, requestedModel, streamReply, type ResponseShape } from './lib/respond.ts';

const FIXED_HOST = '127.0.0.1';
const BODY_LIMIT_BYTES = 2 * 1024 * 1024;
const BODY_TIMEOUT_MS = 5000;
const REGISTER_ATTEMPTS = 20;
const REGISTER_DELAY_MS = 200;
const WORK_POLL_MS = 100;

function log(message: string): void {
  process.stderr.write(`[aggregator-v2] ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Register the catalog with the env, retrying while the env warms up. */
async function registerCatalog(envUrl: string, port: number, catalog: Catalog): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < REGISTER_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(`${envUrl.replace(/\/+$/, '')}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.status >= 200 && res.status < 300) return;
      lastError = new Error(`HTTP ${res.status}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(REGISTER_DELAY_MS);
  }
  throw new Error(`catalog registration failed: ${String(lastError)}`);
}

/** Read a request body with a size cap and a timeout; null on any failure. */
async function readBody(request: IncomingMessage): Promise<string | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      request.destroy();
      resolve(null);
    }, BODY_TIMEOUT_MS);
    request.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > BODY_LIMIT_BYTES) {
        clearTimeout(timer);
        request.destroy();
        resolve(null);
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => {
      clearTimeout(timer);
      resolve(Buffer.concat(chunks).toString('utf8'));
    });
    request.on('error', () => {
      clearTimeout(timer);
      resolve(null);
    });
  });
}

function send(response: ServerResponse, shape: ResponseShape): void {
  response.writeHead(shape.status, {
    'content-type': shape.contentType,
    'cache-control': 'no-store',
  });
  response.end(shape.body);
}

interface Served {
  brief: Brief;
  catalog: Catalog;
  router: Router;
  manifestIndex: number;
  /** The request queue tail: handlers chain onto it, one at a time. */
  queueTail: Promise<void>;
}

function nextRequirement(served: Served, api: 'openai' | 'anthropic'): Requirement {
  const entry = served.brief.manifest[served.manifestIndex];
  if (entry !== undefined && entry.api === api) return entry.requires;
  // The manifest is exhausted, or this request does not match the next
  // manifest slot: fall back to what the body itself declares.
  return {};
}

async function handleCompletion(
  served: Served,
  api: 'openai' | 'anthropic',
  bodyText: string,
): Promise<ResponseShape> {
  let body: unknown = null;
  try {
    body = JSON.parse(bodyText);
  } catch {
    return clientError(api, 'request body is not valid JSON');
  }
  if (!isObject(body)) return clientError(api, 'request body must be a JSON object');
  let norm: CanonRequest;
  try {
    norm = api === 'openai' ? parseOpenAI(body) : parseAnthropic(body);
  } catch (error) {
    return clientError(api, `request could not be normalized: ${String(error)}`);
  }
  const requirement = nextRequirement(served, api);
  // The manifest index advances only now that the body has parsed.
  served.manifestIndex += 1;
  const model = requestedModel(body);
  const answer = await served.router.serve(norm, requirement);
  if (answer === null) {
    return clientError(api, 'no free provider could serve this request');
  }
  return body.stream === true ? streamReply(api, answer, model) : jsonReply(api, answer, model);
}

async function main(): Promise<void> {
  const briefPath = process.env.MAKEFUN_PROMPT_FILE ?? '';
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  if (briefPath === '') throw new Error('MAKEFUN_PROMPT_FILE is not set');
  const brief = readBrief(briefPath);
  for (const note of brief.notes) log(`brief: ${note}`);
  const catalog = await buildCatalog(envUrl, brief);
  const modelIds = new Set(freeModelIds(catalog));
  log(
    `catalog: ${catalog.providers.length} providers, ${modelIds.size} free models` +
      (catalog.providers.length === 0 ? ' (empty catalog)' : ''),
  );

  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog, null, 2)}\n`);
    return;
  }

  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is not set');

  const router = new Router(envUrl, brief, catalog);
  const served: Served = {
    brief,
    catalog,
    router,
    manifestIndex: 0,
    queueTail: Promise.resolve(),
  };

  const server = createServer((request, response) => {
    // The catalog route is answered immediately, outside the request
    // queue: the warm-up asks it before the workload, and it never
    // touches the manifest. Both wires list exactly the registered
    // catalog's free ids; the wire shape follows the auth header the
    // warm-up used (x-api-key means the Anthropic shape).
    const earlyUrl = new URL(request.url ?? '/', `http://${FIXED_HOST}`);
    if (request.method === 'GET' && earlyUrl.pathname === '/v1/models') {
      const ids = freeModelIds(catalog);
      const anthropicShape = typeof request.headers['x-api-key'] === 'string';
      const payload = anthropicShape
        ? {
            data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })),
            has_more: false,
            first_id: ids[0] ?? null,
            last_id: ids[ids.length - 1] ?? null,
          }
        : {
            object: 'list',
            data: ids.map((id) => ({ id, object: 'model', created: 1700000000, owned_by: 'aggregator' })),
          };
      response.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
      response.end(JSON.stringify(payload));
      return;
    }
    // Serialize handlers through the queue; the response is written
    // only when this request reaches the head of the queue.
    let release!: () => void;
    const lock = new Promise<void>((resolve) => {
      release = resolve;
    });
    const previous = served.queueTail;
    served.queueTail = served.queueTail.then(() => lock);
    void previous
      .then(async () => {
        try {
          const url = new URL(request.url ?? '/', `http://${FIXED_HOST}`);
          const api: 'openai' | 'anthropic' | null =
            url.pathname === '/v1/chat/completions'
              ? 'openai'
              : url.pathname === '/v1/messages'
                ? 'anthropic'
                : null;
          if (request.method !== 'POST' || api === null) {
            response.writeHead(404, { 'content-type': 'application/json' });
            response.end(JSON.stringify({ error: 'not found' }));
            return;
          }
          const bodyText = await readBody(request);
          if (bodyText === null) {
            send(response, clientError(api, 'request body could not be read'));
            return;
          }
          const shape = await handleCompletion(served, api, bodyText);
          send(response, shape);
        } catch (error) {
          // A handler must never take the server down with it.
          log(`handler error: ${String(error)}`);
          if (!response.headersSent) {
            response.writeHead(500, { 'content-type': 'application/json' });
            response.end(JSON.stringify({ error: 'internal error' }));
          } else {
            response.end();
          }
        } finally {
          release();
        }
      })
      .catch(() => {
        // The queue itself must never reject.
      });
  });

  await new Promise<void>((resolve) => server.listen(0, FIXED_HOST, resolve));
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('server did not bind');
  const port = address.port;
  process.stdout.write(`http://${FIXED_HOST}:${port}\n`);

  await registerCatalog(envUrl, port, catalog);
  log('registered catalog with env');

  let shuttingDown = false;
  const shutdown = (): void => {
    if (shuttingDown) return;
    shuttingDown = true;
    log(`shutting down; dead providers: ${router.deadProviders().join(', ') || 'none'}`);
    server.close(() => process.exit(0));
    // Never hang the trial's teardown: force exit after a short grace.
    setTimeout(() => process.exit(0), 2000).unref();
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);

  // Wait until the env reports the workload is done, then exit so the
  // driver can collect the state export.
  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      if (res.status >= 200 && res.status < 300) {
        const work = (await res.json()) as { done?: unknown };
        if (work.done === true) break;
      }
    } catch {
      // The env may restart or lag; keep polling.
    }
    await sleep(WORK_POLL_MS);
  }
  shutdown();
}

main().catch((error: unknown) => {
  process.stderr.write(`[aggregator-v2] fatal: ${String(error)}\n`);
  process.exit(1);
});

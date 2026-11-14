// Router: choose providers, walk them, and fail over.
// ---------------------------------------------------------------------------
// Ranking: the brief's quality table, restricted to free models that
// satisfy the request's requirements and the wire's own limits (a JSON
// request never goes to an Anthropic-wire provider — that wire cannot
// echo JSON mode). Within a provider, models rank by rating; providers
// rank by their best model. When the table is missing or names nothing
// usable, the probed catalog stands in (free models, unrated, in id
// order), so routing still works on a brief that is prose-only.
//
// State: each provider carries a small state machine. A provider that
// says its budget is spent, that it is out, or that answers 401 is
// DEAD — calling it again spends nothing and can never succeed, so it
// is never called again. A scheduled refusal ("refused this call") is
// transient: the provider is still alive, and the very next attempt on
// the same provider succeeds once the scheduled call number has passed.
// Unknown failures are retried once per provider, and three in a row
// retire the provider. A per-provider mutex serializes inference calls
// (a provider with a one-request concurrency limit answers 429 to a
// second simultaneous call without spending budget; the mutex makes
// that race impossible), and call attempts are capped per request so a
// pathological trial can never burn the wall clock.
//
// The 200 rule: an HTTP 200 is FINAL. Its body is parsed best-effort
// and never retried — a second successful call would break the trial's
// exactly-one-successful-call check and spend budget for nothing.

import { authHeaders, joinBase, type Catalog } from './catalog.ts';
import {
  meetsRequirements,
  ratingRowFor,
  type Brief,
  type ProviderEntry,
  type Requirement,
} from './brief.ts';
import type { CanonRequest } from './normalize.ts';
import {
  buildAnthropic,
  buildGemini,
  buildOpenAI,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  parseProviderError,
  type ProviderAnswer,
  type ProviderFailure,
} from './upstream.ts';

export interface Candidate {
  provider: ProviderEntry;
  model: string;
  ratingMilli: number;
}

const CALL_TIMEOUT_MS = 3000;
const MAX_CALLS_PER_REQUEST = 50;
const PASS2_MAX_ATTEMPTS = 1;

/** A tiny per-provider mutex: callers chain onto a promise tail. */
export class Mutex {
  private tail: Promise<void> = Promise.resolve();

  async run<T>(task: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const lock = new Promise<void>((resolve) => {
      release = resolve;
    });
    const previous = this.tail;
    this.tail = this.tail.then(() => lock);
    await previous;
    try {
      return await task();
    } finally {
      release();
    }
  }
}

export interface ProviderState {
  dead: boolean;
  timeouts: number;
  unknownStreak: number;
}

export function freshState(): ProviderState {
  return { dead: false, timeouts: 0, unknownStreak: 0 };
}

export type RefusalClass = 'permanent' | 'scheduled' | 'concurrency' | 'unknown';

export type CallOutcome =
  | { kind: 'answer'; answer: ProviderAnswer }
  | { kind: 'failure'; failure: ProviderFailure; refusal: RefusalClass }
  | { kind: 'timeout' }
  | { kind: 'network' }
  | { kind: 'model400' };

/**
 * Classify a provider refusal by its semantics, not its status code
 * alone: the sealed providers all refuse with 429/402, and only the
 * message tells a spent budget from a scheduled hiccup.
 */
export function classifyRefusal(failure: ProviderFailure): RefusalClass {
  const message = failure.message.toLowerCase();
  if (message.includes('budget is spent')) return 'permanent';
  if (/\bis out\b/.test(message) || message.endsWith('is out')) return 'permanent';
  if (message.includes('outage')) return 'permanent';
  if (failure.status === 401) return 'permanent';
  if (message.includes('refused this call')) return 'scheduled';
  if (message.includes('concurrent') || message.includes('too many concurrent')) return 'concurrency';
  return 'unknown';
}

/**
 * The candidate list for one request, best first, with no provider
 * named twice (its best eligible model) and no dead provider.
 */
export function rankCandidates(
  brief: Brief,
  catalog: Catalog,
  req: Requirement,
  states: Map<string, ProviderState>,
  jsonMode: boolean,
): Candidate[] {
  const catalogByProvider = new Map(catalog.providers.map((p) => [p.id, p]));
  const out: Candidate[] = [];
  for (const entry of brief.providers) {
    if (states.get(entry.id)?.dead) continue;
    if (jsonMode && entry.wire === 'anthropic') continue;
    const listed = catalogByProvider.get(entry.id);
    const catalogIds = listed === undefined ? null : new Set(listed.models.filter((m) => m.free).map((m) => m.id));
    let best: Candidate | null = null;
    for (const row of brief.ratings) {
      if (row.provider !== entry.id || !row.free) continue;
      if (catalogIds !== null && !catalogIds.has(row.model)) continue;
      if (!meetsRequirements(row, req)) continue;
      if (best === null || row.ratingMilli > best.ratingMilli) {
        best = { provider: entry, model: row.model, ratingMilli: row.ratingMilli };
      }
    }
    if (best === null && listed !== undefined) {
      // No rated model qualifies: fall back to the probed catalog's
      // free models, unrated, lowest id first. Their capability flags
      // are unknown, so requirements cannot be checked here.
      for (const model of listed.models) {
        if (!model.free) continue;
        const row = ratingRowFor(brief, entry.id, model.id);
        if (row !== null && !meetsRequirements(row, req)) continue;
        best = { provider: entry, model: model.id, ratingMilli: -1 };
        break;
      }
    }
    if (best !== null) out.push(best);
  }
  out.sort((a, b) => b.ratingMilli - a.ratingMilli || a.provider.id.localeCompare(b.provider.id));
  return out;
}

/** The inference endpoint URL for one provider call (a Gemini model rides in the path). */
export function callUrl(envUrl: string, entry: ProviderEntry, model: string): string {
  const base = joinBase(envUrl, entry.base);
  if (entry.wire === 'gemini') return `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  if (entry.wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1/chat/completions`;
}

/** Parse a 200 body on the provider's own wire (total: never throws, never null). */
export function parseAnswer(wire: ProviderEntry['wire'], bodyText: string): ProviderAnswer {
  if (wire === 'anthropic') return parseAnthropicAnswer(bodyText);
  if (wire === 'gemini') return parseGeminiAnswer(bodyText);
  return parseOpenAIAnswer(bodyText);
}

export class Router {
  private readonly envUrl: string;
  private readonly brief: Brief;
  private readonly catalog: Catalog;
  private readonly mutexes = new Map<string, Mutex>();
  private readonly states = new Map<string, ProviderState>();

  constructor(envUrl: string, brief: Brief, catalog: Catalog) {
    this.envUrl = envUrl;
    this.brief = brief;
    this.catalog = catalog;
  }

  private stateFor(id: string): ProviderState {
    let state = this.states.get(id);
    if (state === undefined) {
      state = freshState();
      this.states.set(id, state);
    }
    return state;
  }

  private mutexFor(id: string): Mutex {
    let mutex = this.mutexes.get(id);
    if (mutex === undefined) {
      mutex = new Mutex();
      this.mutexes.set(id, mutex);
    }
    return mutex;
  }

  isDead(id: string): boolean {
    return this.states.get(id)?.dead ?? false;
  }

  /** The provider states, for the stderr diagnostics line at shutdown. */
  deadProviders(): string[] {
    return [...this.states.entries()].filter(([, s]) => s.dead).map(([id]) => id);
  }

  /** One provider call, serialized by the provider's mutex, with a bounded timeout. */
  async callProvider(entry: ProviderEntry, model: string, norm: CanonRequest): Promise<CallOutcome> {
    return this.mutexFor(entry.id).run(async () => {
      const body = entry.wire === 'anthropic' ? buildAnthropic(norm, model) : entry.wire === 'gemini' ? buildGemini(norm) : buildOpenAI(norm, model);
      const headers: Record<string, string> = {
        'content-type': 'application/json',
        accept: 'application/json',
        ...authHeaders(entry),
      };
      if (entry.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
      const urls = [callUrl(this.envUrl, entry, model)];
      if (entry.wire === 'gemini' && entry.auth === 'x-goog-api-key' && entry.key !== '') {
        urls.push(`${urls[0]}?key=${encodeURIComponent(entry.key)}`);
      }
      let lastOutcome: CallOutcome = { kind: 'network' };
      for (const url of urls) {
        let res: Response;
        try {
          res = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
          });
        } catch (error) {
          if (error instanceof DOMException && error.name === 'TimeoutError') return { kind: 'timeout' };
          lastOutcome = { kind: 'network' };
          continue;
        }
        const text = await res.text().catch(() => '');
        if (res.status === 200) {
          // FINAL: a 200 is never retried, whatever its body looks like.
          return { kind: 'answer', answer: parseAnswer(entry.wire, text) };
        }
        if (res.status === 400) return { kind: 'model400' };
        if (res.status === 401 && urls.length > 1 && url === urls[0]) {
          // A Gemini key carried in the header may be rejected where
          // the same key in the query string is accepted: one retry.
          continue;
        }
        const failure = parseProviderError(res.status, text, entry.wire);
        return { kind: 'failure', failure, refusal: classifyRefusal(failure) };
      }
      return lastOutcome;
    });
  }

  /**
   * Serve one request: walk the candidates best-first, retrying only
   * transient failures on the same provider, then one second pass over
   * the survivors. Returns the first answer found, or null when no
   * provider can serve the request (the client then gets a wire-shaped
   * error, which the trial counts as unserved rather than crashing).
   */
  async serve(norm: CanonRequest, req: Requirement): Promise<ProviderAnswer | null> {
    const counter = { calls: 0, budget: MAX_CALLS_PER_REQUEST };
    const skipped400 = new Set<string>();
    const first = rankCandidates(this.brief, this.catalog, req, this.states, norm.jsonMode);
    const answer = await this.walk(first, norm, req, skipped400, 3, counter);
    if (answer !== null) return answer;
    const second = rankCandidates(this.brief, this.catalog, req, this.states, norm.jsonMode);
    if (second.length === 0) return null;
    return this.walk(second, norm, req, skipped400, PASS2_MAX_ATTEMPTS, counter);
  }

  /** One walk over a candidate list; returns the first answer, or null. */
  private async walk(
    candidates: Candidate[],
    norm: CanonRequest,
    req: Requirement,
    skipped400: Set<string>,
    maxAttemptsPerCandidate: number,
    counter: { calls: number; budget: number },
  ): Promise<ProviderAnswer | null> {
    for (const candidate of candidates) {
      const state = this.stateFor(candidate.provider.id);
      if (state.dead || skipped400.has(candidate.provider.id)) continue;
      let model400s = 0;
      for (let attempt = 0; attempt < maxAttemptsPerCandidate; attempt += 1) {
        if (counter.calls >= counter.budget) return null;
        counter.calls += 1;
        const outcome = await this.callProvider(candidate.provider, candidate.model, norm);
        if (outcome.kind === 'answer') {
          state.unknownStreak = 0;
          return outcome.answer;
        }
        if (outcome.kind === 'model400') {
          // The model name is rejected; a second 400 on the same
          // provider within this request means the body is suspect —
          // skip the provider for this request only.
          model400s += 1;
          state.unknownStreak = 0;
          if (model400s >= 2) skipped400.add(candidate.provider.id);
          break;
        }
        if (outcome.kind === 'timeout') {
          state.timeouts += 1;
          if (state.timeouts >= 2) state.dead = true;
          break;
        }
        if (outcome.kind === 'network') {
          state.unknownStreak += 1;
          if (state.unknownStreak >= 3) state.dead = true;
          break;
        }
        // A classified refusal.
        if (outcome.refusal === 'permanent') {
          state.dead = true;
          break;
        }
        if (outcome.refusal === 'concurrency') {
          await sleep(25);
          continue;
        }
        if (outcome.refusal === 'scheduled') {
          // The scheduled refusal spends one call number; the next
          // attempt is past it. Retry briefly, never longer than the
          // per-candidate attempt cap.
          await sleep(10);
          continue;
        }
        // Unknown refusals: one retry here, then move on; a provider
        // that fails three times in a row is retired.
        state.unknownStreak += 1;
        if (state.unknownStreak >= 3) {
          state.dead = true;
          break;
        }
        if (attempt + 1 >= 2) break;
        await sleep(10);
      }
    }
    return null;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

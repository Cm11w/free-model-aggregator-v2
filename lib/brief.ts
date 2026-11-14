// Brief parsing for the free-model aggregator, v2.
//
// The trial brief (MAKEFUN_PROMPT_FILE) is prose with three embedded JSON
// documents: the provider directory, the quality table, and the workload
// manifest. This module finds each document by its `schema` member with a
// brace scanner that respects strings, so nothing about the prose layout
// (line breaks, ordering, extra text) matters. Every field reader is
// total: a missing, mistyped or oddly-shaped field degrades to a safe
// default instead of throwing, because a brief that is merely unusual
// must never take the whole trial down.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type JsonObject = Record<string, unknown>;

export function isObject(v: unknown): v is JsonObject {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  keyEnv: string;
  key: string;
  /** Base path such as /p/groq (or an absolute URL); the catalog reports it verbatim. */
  base: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  /** Human-readable notes about anything odd found while parsing; stderr only. */
  notes: string[];
}

/** Scan `text` for balanced top-level JSON objects, respecting string escapes. */
function scanObjects(text: string): unknown[] {
  const found: unknown[] = [];
  let i = 0;
  while (i < text.length) {
    if (text[i] !== '{') {
      i += 1;
      continue;
    }
    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;
    for (let j = i; j < text.length; j += 1) {
      const c = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (c === '\\') escaped = true;
        else if (c === '"') inString = false;
        continue;
      }
      if (c === '"') inString = true;
      else if (c === '{') depth += 1;
      else if (c === '}') {
        depth -= 1;
        if (depth === 0) {
          end = j;
          break;
        }
      }
    }
    if (end === -1) {
      // An unbalanced brace (prose, or an unclosed quote): step past it
      // and keep scanning — a valid document may still follow.
      i += 1;
      continue;
    }
    try {
      found.push(JSON.parse(text.slice(i, end + 1)));
      i = end + 1;
    } catch {
      // A brace-balanced span that is not JSON: retry from just past
      // this brace instead of skipping the whole span, so a document
      // nested inside prose is never swallowed.
      i += 1;
    }
  }
  return found;
}

function asString(v: unknown, fallback = ''): string {
  return typeof v === 'string' ? v : fallback;
}

function asFinite(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asWire(v: unknown, notes: string[], id: string): Wire {
  if (v === 'anthropic' || v === 'gemini' || v === 'openai') return v;
  notes.push(`provider ${id}: unknown wire ${JSON.stringify(v)}, treating as openai`);
  return 'openai';
}

function asAuth(v: unknown): AuthScheme {
  if (v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none') return v;
  // 'bearer' and anything unrecognized: the bearer header is the common case.
  return 'bearer';
}

/** The provider's base: `path` first, then base_path / base_url / base spellings. */
function baseOf(raw: JsonObject): string {
  for (const key of ['path', 'base_path', 'base_url', 'base']) {
    const value = asString(raw[key]);
    if (value !== '') return value;
  }
  return '';
}

function parseProviders(doc: JsonObject, notes: string[]): ProviderEntry[] {
  const list = Array.isArray(doc.providers) ? doc.providers : [];
  const out: ProviderEntry[] = [];
  const seen = new Set<string>();
  for (const raw of list) {
    if (!isObject(raw)) continue;
    const id = asString(raw.id);
    if (id === '' || seen.has(id)) continue;
    seen.add(id);
    out.push({
      id,
      name: asString(raw.name, id),
      wire: asWire(raw.wire, notes, id),
      auth: asAuth(raw.auth),
      keyEnv: asString(raw.key_env),
      key: asString(raw.key),
      base: baseOf(raw),
    });
  }
  return out;
}

function parseRatings(doc: JsonObject): { ratings: RatingRow[]; best: Map<string, number> } {
  const rows: RatingRow[] = [];
  const list = Array.isArray(doc.ratings) ? doc.ratings : [];
  const seen = new Set<string>();
  for (const raw of list) {
    if (!isObject(raw)) continue;
    const model = asString(raw.model);
    const provider = asString(raw.provider);
    if (model === '' || provider === '') continue;
    const key = `${provider}\n${model}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({
      model,
      provider,
      ratingMilli: asFinite(raw.rating_milli) ?? 0,
      contextLength: asFinite(raw.context_length),
      tools: raw.tools === true,
      vision: raw.vision === true,
      json: raw.json === true,
      free: raw.free === true,
    });
  }
  const best = new Map<string, number>();
  const bestList = Array.isArray(doc.best) ? doc.best : [];
  for (const raw of bestList) {
    if (!isObject(raw)) continue;
    const id = asString(raw.id);
    const rating = asFinite(raw.rating_milli);
    if (id !== '' && rating !== null) best.set(id, rating);
  }
  return { ratings: rows, best };
}

function parseRequirement(raw: unknown): Requirement {
  if (!isObject(raw)) return {};
  const req: Requirement = {};
  if (raw.tools === true) req.tools = true;
  if (raw.vision === true) req.vision = true;
  if (raw.json === true) req.json = true;
  const minContext = asFinite(raw.min_context);
  if (minContext !== null) req.minContext = minContext;
  return req;
}

function parseManifest(doc: JsonObject): WorkloadRequest[] {
  const list = Array.isArray(doc.manifest) ? doc.manifest : [];
  const out: WorkloadRequest[] = [];
  for (const raw of list) {
    if (!isObject(raw)) continue;
    const id = asString(raw.id);
    if (id === '') continue;
    out.push({
      id,
      api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
      stream: raw.stream === true,
      weight: raw.weight === 2 ? 2 : 1,
      requires: parseRequirement(raw.requires),
    });
  }
  return out;
}

/**
 * Read and parse the brief file. Throws only when the file cannot be read
 * at all or holds no usable document; a brief missing one document still
 * yields a usable Brief (the missing part degrades: no ratings means
 * routing falls back to probe data, no manifest means requirements come
 * from the request bodies).
 */
export function readBrief(path: string): Brief {
  const notes: string[] = [];
  const text = readFileSync(path, 'utf8');
  const docs = scanObjects(text);
  let providers: ProviderEntry[] = [];
  let ratings: RatingRow[] = [];
  let best = new Map<string, number>();
  let manifest: WorkloadRequest[] = [];
  for (const doc of docs) {
    if (!isObject(doc)) continue;
    if (doc.schema === PROVIDERS_SCHEMA) providers = parseProviders(doc, notes);
    else if (doc.schema === RATINGS_SCHEMA) {
      const parsed = parseRatings(doc);
      ratings = parsed.ratings;
      best = parsed.best;
    } else if (doc.schema === WORKLOAD_SCHEMA) manifest = parseManifest(doc);
  }
  if (providers.length === 0 && ratings.length === 0 && manifest.length === 0) {
    throw new Error(`no brief documents found in ${path}`);
  }
  if (providers.length === 0) notes.push('brief holds no provider directory');
  if (ratings.length === 0) notes.push('brief holds no quality table');
  if (manifest.length === 0) notes.push('brief holds no workload manifest');
  return { providers, ratings, best, manifest, notes };
}

/** Does a rating row satisfy a requirement set? Mirrors the env's own rule. */
export function meetsRequirements(row: RatingRow, req: Requirement): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.minContext !== undefined) {
    if (row.contextLength === null || row.contextLength < req.minContext) return false;
  }
  return true;
}

/** The rating row for one provider's model, when the quality table has one. */
export function ratingRowFor(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

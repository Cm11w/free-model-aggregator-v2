// Catalog: probe every provider's own catalog route and reconcile what
// it says with the brief's quality table, into the one catalog document
// that is both registered with the env and printed by --list.
//
// The probe is free (a catalog call spends no inference budget), so it is
// retried; a provider whose probe fails still appears in the catalog from
// the quality table. Reconciliation rule: the quality table is the sealed
// truth for the models it names, so a model the table names takes the
// table's own free flag whatever a listing says; a model only the probe
// saw keeps the probe's own free flag. The union of both sources is what
// makes the free set exact even when a listing is partial or confused.

import { ratingRowFor, type Brief, type ProviderEntry } from './brief.ts';

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: ProviderEntry['wire'];
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const PROBE_TIMEOUT_MS = 3000;
const PROBE_ATTEMPTS = 3;

type JsonObject = Record<string, unknown>;

function isObject(v: unknown): v is JsonObject {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/** A base path joined to the env URL; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The auth header a provider's scheme names, or none for scheme `none`. */
export function authHeaders(entry: ProviderEntry): Record<string, string> {
  if (entry.key === '') return {};
  if (entry.auth === 'bearer') return { authorization: `Bearer ${entry.key}` };
  if (entry.auth === 'x-api-key') return { 'x-api-key': entry.key };
  if (entry.auth === 'x-goog-api-key') return { 'x-goog-api-key': entry.key };
  return {};
}

function catalogPath(entry: ProviderEntry): string {
  return entry.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
}

function finiteOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/**
 * One listed model, in whichever wire's spelling. Accepts the OpenAI
 * `data` shape, the Anthropic `data` shape, the Gemini `models` shape
 * (with the `models/` prefix), and a bare top-level array. A listing that
 * prices models instead of flagging them reads as free when every price
 * is zero; a listing that says nothing about price leaves free unknown.
 */
function listedModel(wire: ProviderEntry['wire'], raw: unknown): { id: string; free: boolean | null; context: number | null } | null {
  if (!isObject(raw)) return null;
  const id =
    wire === 'gemini'
      ? (typeof raw.name === 'string' ? raw.name : '').replace(/^models\//, '')
      : typeof raw.id === 'string'
        ? raw.id
        : '';
  if (id === '') return null;
  let free: boolean | null = typeof raw.free === 'boolean' ? raw.free : null;
  if (free === null && isObject(raw.pricing)) {
    const values = Object.values(raw.pricing).flatMap((x) => {
      if (typeof x === 'number') return [x];
      if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
      return [];
    });
    if (values.length > 0) free = values.every((n) => Number.isFinite(n) && n === 0);
  }
  const context = finiteOrNull(raw.context_length) ?? finiteOrNull(raw.context_window) ?? finiteOrNull(raw.inputTokenLimit);
  return { id, free, context };
}

function modelsFromProbe(entry: ProviderEntry, body: unknown): { id: string; free: boolean | null; context: number | null }[] | null {
  if (Array.isArray(body)) {
    const out: { id: string; free: boolean | null; context: number | null }[] = [];
    for (const raw of body) {
      const m = listedModel(entry.wire, raw);
      if (m !== null) out.push(m);
    }
    return out;
  }
  if (!isObject(body)) return null;
  const list = entry.wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return null;
  const out: { id: string; free: boolean | null; context: number | null }[] = [];
  for (const raw of list) {
    const m = listedModel(entry.wire, raw);
    if (m !== null) out.push(m);
  }
  return out;
}

/** Probe one provider's catalog route, retrying transient failures. */
async function probeProvider(
  envUrl: string,
  entry: ProviderEntry,
): Promise<{ id: string; free: boolean | null; context: number | null }[] | null> {
  const url = `${joinBase(envUrl, entry.base)}${catalogPath(entry)}`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(entry) };
  if (entry.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  for (let attempt = 0; attempt < PROBE_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(url, { method: 'GET', headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (res.status < 200 || res.status >= 300) return null;
      const text = await res.text();
      let body: unknown = null;
      try {
        body = JSON.parse(text);
      } catch {
        return null;
      }
      const models = modelsFromProbe(entry, body);
      if (models !== null && models.length > 0) return models;
      return null;
    } catch {
      // A refused connection or a timeout: retry; the fallback covers a dead probe.
    }
  }
  return null;
}

/**
 * Build the catalog: probe every provider (in parallel — catalog calls
 * never touch the one-call-at-a-time inference rule), reconcile each
 * listing with the quality table, and keep the brief's provider order
 * with models sorted by id, so the document is deterministic.
 */
export async function buildCatalog(envUrl: string, brief: Brief): Promise<Catalog> {
  const probed = await Promise.all(
    brief.providers.map(async (entry) => (envUrl === '' ? null : probeProvider(envUrl, entry))),
  );
  const providers: CatalogProvider[] = [];
  for (let i = 0; i < brief.providers.length; i += 1) {
    const entry = brief.providers[i];
    const probe = probed[i] ?? [];
    const byId = new Map<string, { free: boolean | null; context: number | null }>();
    for (const m of probe) {
      if (!byId.has(m.id)) byId.set(m.id, { free: m.free, context: m.context });
    }
    // The quality table's free models for this provider join the union,
    // even when the probe missed them.
    for (const row of brief.ratings) {
      if (row.provider !== entry.id || !row.free) continue;
      if (!byId.has(row.model)) byId.set(row.model, { free: true, context: row.contextLength });
    }
    const models: CatalogModel[] = [];
    for (const id of [...byId.keys()].sort()) {
      const seen = byId.get(id);
      const row = ratingRowFor(brief, entry.id, id);
      models.push({
        id,
        // The quality table's own free flag wins when it names the
        // model (it can list paid models too); otherwise the probe's.
        free: row !== null ? row.free : (seen?.free ?? false),
        context_length: row?.contextLength ?? seen?.context ?? null,
      });
    }
    providers.push({ id: entry.id, base_url: entry.base, key_env: entry.keyEnv, wire: entry.wire, models });
  }
  return { providers };
}

/** The union of free model ids in the catalog, each once, in first-seen order. */
export function freeModelIds(catalog: Catalog): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !seen.has(model.id)) {
        seen.add(model.id);
        ids.push(model.id);
      }
    }
  }
  return ids;
}

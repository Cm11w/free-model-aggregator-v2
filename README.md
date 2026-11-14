# Free-Model Aggregator v2

A resilient aggregator for the `free-model-aggregator-01` competition: one
OpenAI/Anthropic-compatible endpoint that routes every request to the best
free model across the providers named in the trial brief, with failover
that distinguishes a spent budget from a transient refusal.

This is a fresh implementation written for robustness against a sealed
final trial. Nothing about any provider, model, rating, or budget is
hard-coded: routing is derived at runtime from the brief's quality table
and from live catalog probes, and every parser degrades instead of
crashing when a provider answers with something unusual.

## Run

```sh
node main.ts --list        # print the registered catalog (no env needed)
MAKEFUN_PROMPT_FILE=brief.txt MAKEFUN_ENV_URL=http://127.0.0.1:PORT node main.ts
```

Node 24+ (type stripping; no build step, no dependencies). The entry
prints its own URL on stdout, registers the catalog with the env, serves
`POST /v1/chat/completions` and `POST /v1/messages`, and exits when the
env reports the workload is done.

## Architecture

```
main.ts          boot, registration, HTTP server, request queue, shutdown
lib/brief.ts     tolerant brief parsing (brace scanner, schema-keyed docs)
lib/catalog.ts   live catalog probes + reconciliation with the quality table
lib/normalize.ts client wires -> one canonical request (echo-faithful)
lib/upstream.ts  canonical request -> provider wires; total answer parsers
lib/respond.ts   one answer -> client JSON or synthesized SSE stream
lib/router.ts    ranking, provider state machine, mutex, failover walk
tests/adversarial.mjs   hostile-provider harness (see below)
```

### What makes v2 resilient

- **The 200 rule.** An HTTP 200 is final. Its body is parsed best-effort
  (JSON, SSE, or a half-broken mix) and never retried: a second
  successful call would break the trial's exactly-one-successful-call
  check and burn budget for nothing.
- **Refusal semantics.** A refusal is classified from its message:
  *permanent* ("budget is spent", "is out", "outage", 401) retires the
  provider for the rest of the trial; *scheduled* ("refused this call")
  is retried briefly, because the very next attempt is past the
  scheduled call number; *concurrency* is retried once after 25 ms;
  *unknown* failures get one retry, and three in a row retire the
  provider. No provider is called again once it is clearly dead, and
  no transient failure is given up on after a single attempt.
- **Per-provider mutex.** Inference calls to one provider are
  serialized, so a provider with a one-request concurrency limit can
  never see two simultaneous calls. Catalog probes are exempt (they
  spend no budget and never hit the concurrency check).
- **Deterministic serving.** Requests are handled one at a time through
  a queue, and the workload-manifest index advances only after a body
  has parsed — a malformed body can never shift the requirements of
  the requests behind it. The reply timestamp is a fixed constant, so
  identical trials produce byte-identical exports.
- **Bounded effort.** 3 s per call, 50 calls per request, two passes
  over the survivors. A pathological trial can never burn the wall
  clock.
- **Total parsers.** Every answer parser returns an answer, never null
  and never throws: JSON, SSE, truncated SSE, wrong-typed fields, and
  empty bodies all degrade to the best extractable answer. Error
  envelopes are read in all three wires' shapes.
- **Echo fidelity.** The canonical form preserves exactly what the
  providers echo — system text, last user text, tool names in order,
  max_tokens, stops, image bytes, JSON mode — and drops tool results
  only on the Gemini wire, where a function-response part would
  clobber the echoed user text. JSON requests never route to an
  Anthropic-wire provider (that wire cannot echo JSON mode).

## Adversarial harness

`tests/adversarial.mjs` runs the entry against mock providers that
behave badly on purpose: spent budgets, scheduled refusals, outages,
401s, per-model 400s, 200s with broken JSON, truncated SSE, slow and
hanging providers, empty answers, tool-call answers, duplicate and
confused catalog listings, and a missing manifest. Each scenario
asserts the request is still served correctly (or, when nothing can
serve it, that the entry answers with a wire-shaped error instead of
crashing). Run it with:

```sh
node tests/adversarial.mjs
```

## Verification

Practice trials are run against the kit's sealed env server and
official scorer with a local-process harness (no Docker). Results
(14 Nov 2026, verified by the builder):

- Practice trial-0: **99** (quality 99, served/wire/failover/catalog 100, 26 upstream calls)
- Practice trial-1: **99** (quality 99, served/wire/failover/catalog 100, 28 upstream calls)
- Practice trial-2: **99** (quality 99, served/wire/failover/catalog 100, 28 upstream calls)
- Fresh generator seeds (6 new instances drawn with `draw.mjs`, each a
  different budget/fail-call/outage draw — the exact variation the
  sealed final trial uses): **99 on all six** (24–31 upstream calls)
- Determinism: trial-0 run twice leaves byte-identical state exports
  (sha256 `e732a321…`), and the export is unchanged by later
  robustness fixes — replay-safe
- Adversarial harness: **31/31 checks pass** (`node tests/adversarial.mjs`)

## License

MIT — see `LICENSE`.

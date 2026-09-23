# SatQuery AI (SIH26167) — Full Repository Audit

**Audit date:** 2026-08-28
**Scope:** whole repository, as-is. **No source file was modified to produce this audit.**
**Verdict in one line:** the analysis core (`app/agents`, `app/services`, `app/geospatial`) is real,
coherent and largely honest; the API / persistence / reporting shell and the entire evaluation
presentation layer are broken and over-claiming. The correct remedy is **surgical repair, not a
rewrite** (per §59 "reuse working implementations").

---

## 0. How this audit was produced

Findings below are separated into two classes, and the class is stated on every finding:

| Marker | Meaning |
| :--- | :--- |
| **[MEASURED]** | Reproduced by executing the code in this session; the observed output is quoted. |
| **[READ]** | Established by reading the source; not executed end-to-end. |

Verification gate, re-run this session (this supersedes any earlier "all green" claim):

```
backend$ ../.venv/Scripts/python.exe -m pytest -q --no-header
  1 failed, 1078 passed, 1 warning in 58.91s          -> EXIT 1
backend$ ../.venv/Scripts/python.exe -m ruff check app tests
  Found 44 errors.                                    -> EXIT 1
backend$ ../.venv/Scripts/python.exe -m mypy app
  Found 7 errors in 1 file (checked 49 source files)  -> EXIT 1
```

**The gate is RED on all three tools.** Any statement that this project is complete and verified is
false as of this date.

---

## 1. Current architecture (as built)

```
frontend/src                     React 19 + Vite + TS + Tailwind
  services/api.ts                <- API client. ALSO SYNTHESISES DATA THE BACKEND DID NOT SEND (§4.6)
  services/fixtures.ts           <- offline demo replay of recorded runs
  pages/evaluation-page.tsx      <- benchmark numbers HARDCODED IN SOURCE (§4.7)
  lib/report-model.ts (686 L)    <- report builder (works; do not rebuild, §1/§32)

backend/app
  api/routes/analysis.py  (498 L)  <- EPICENTRE OF DEFECTS
  api/routes/upload.py    (132 L)  <- security gaps (§4.4)
  schemas/analysis.py     (124 L)  <- contract drift, lying defaults
  db/{models,session}.py           <- §7 fields absent; status hardcoded
  main.py                          <- CORS *, storage/ publicly mounted

  agents/     classifier -> router -> registry -> 7 tools     HONEST, WIRED
  services/   landcover change sar fusion grounding vqa
              captioning indices features vocabulary
              confidence.py (477 L, documented gate formula)  HONEST, WIRED
  geospatial/ raster validate align measure vectorize viz     HONEST, WIRED

ml/           adaptation/ evaluation/ datasets/ inference/  checkpoints/
data/demo/    optical/ sar/ temporal/ + *_test.json fixtures
```

Intended pipeline (§5): classify -> **normalize canonical mode** -> route -> validate -> execute ->
validate evidence -> integrate.

**Actually built pipeline [READ + MEASURED]:**

```
classify(query) -> route(query, rasters) -> registry.dispatch -> tool.validate_input (INSIDE run)
                                                              -> tool.run  -> confidence
derive_input_mode(rasters)  ------ computed, then DISCARDED ------> only written to the DB row
```

There is **no orchestrator module and no evidence-validator module**. `InputMode` exists in
`core/types.py`, is imported at `analysis.py:21`, and is **never used** (ruff F401) — mode is
cosmetic in the backend while the *frontend* depends on it heavily.

---

## 2. What is real vs fixture vs mock vs hardcoded

| Component | Classification | Evidence |
| :--- | :--- | :--- |
| Land-cover / VQA / captioning / grounding / change / SAR / fusion services | **REAL classical CV** on real pixels | [MEASURED] T1 returned `bare soil 69%… 26.21 km²`; T5 returned `10.8% of valid area (2.82 km²)`, derived from actual rasters |
| `app/geospatial/*` (raster IO, CRS, alignment, area measurement, vectorize) | **REAL** | [MEASURED] ground areas in km² are computed, so pixel-area maths and georeferencing are live |
| `services/confidence.py` | **REAL**, documented formula `min(weighted_mean(factors), min(gates))` | [READ] full factor/gate model with `to_dict()` |
| Classifier + router | **REAL** deterministic weighted-keyword matcher (NOT an LLM) | [MEASURED] T1–T6 all matched with reported keyword lists |
| Registry tier fallback | **REAL** but never exercised — all 7 tools are `CLASSICAL` tier | [READ] |
| `GET /api/models` | **100% HARDCODED FABRICATION** | [MEASURED] 7 invented entries, all `available:true status:ready`; names match no registered tool |
| Evaluation page benchmark table | **HARDCODED IN TYPESCRIPT SOURCE** | [READ] `BENCHMARKS` const, 5 rows, all `status:'evaluated'` |
| Confidence `reasons` + `factors` shown in the UI | **FABRICATED IN THE FRONTEND** | [READ] `api.ts:231,419` invent three fixed reason strings |
| `reports/benchmark_evaluation_report.json` | **MIXED**: change-detection is a sound synthetic test; VQA is circular; fusion is incoherent | see §3 |
| `data/demo/{rsvqa,vrsbench,cdvqa}_test.json` | **LOCALLY AUTHORED FILES NAMED AFTER PUBLIC BENCHMARKS** (2–4 KB, 10 items) | [READ] rsvqa "answers" are verbatim output of this system's own answer template |
| `ml/datasets/processed/dataset_manifest.json` | **SYNTHETIC, and says so** | `"dataset":"BigEarthNet-Synthetic","is_synthetic":true`, 300 samples, Gaussian-noise spectra |
| `ml/checkpoints/satquery_adapted` | **REAL 35.5 s CPU training run** of a from-scratch 6-band CNN; **mislabelled as LoRA** | `total_parameters == trainable_parameters == 299859`; `best_val_f1 0.1176` |
| Frontend offline fixtures (`SAT-2026-0001xx.json`) | **Recorded real runs**, labelled `fixture:true` | [READ] honest design |

---

## 3. Classical CV vs real ML inference vs UI simulation

**Every answer the running system produces today comes from classical CV.** All 7 registered tools
are `ToolTier.CLASSICAL`. `enable_learned_models` defaults to `False`, and `/api/health` confirms
`"learned_models_enabled": false` [MEASURED]. Correct §49 naming for what exists: *classical
spectral land-cover decomposition*, *change vector analysis*, *decision-level optical/SAR fusion*,
*template-based answer generation over measured statistics*.

**The one real ML artefact.** `ml/checkpoints/satquery_adapted/adapter_config.json` records a
genuine 15-epoch, 35.48 s CPU training run — `training_completed: true`, `is_simulated: false`, full
per-epoch history, and two real `.pt` files (~1.2 MB). Training genuinely happened. But:

1. `total_parameters (299859) == trainable_parameters (299859)` — 100% of parameters are trainable,
   so this is **ordinary full training, not LoRA**, despite `adaptation_method: "LoRA"`,
   `lora_rank: 8`, `lora_alpha: 16.0`.
2. `base_model: "rs-cnn-6band"` is an **in-repo from-scratch CNN**, not a pretrained vision-language
   model. There is therefore **no remote-sensing *adaptation of a VLM*** — §0 forbids claiming
   adaptation that did not occur.
3. The model does not work: `best_val_f1 0.1176`, final `val_recall 0.0375`, `val_f1 0.0723`. The
   headline `val_accuracy 0.9099` is the standard multi-label all-negatives artefact.
4. Its training data is synthetic (`BigEarthNet-Synthetic`); **BigEarthNet is not present**.
5. It is **not in the serving path**, yet `/api/models` advertises it as
   `RemoteSensingLoRAAdapter, tier: "preferred", available: true, status: "ready"` [MEASURED] —
   which contradicts `/api/health` in the same API surface.

**Benchmark numbers, examined individually:**

| Metric | Reported | Assessment |
| :--- | :--- | :--- |
| VQA exact-match / similarity | `1.0` / `1.0` | **Circular.** Its own note admits "Fixture-only samples use self-match baseline". Measures determinism, not accuracy. Cannot be reported as a benchmark score. |
| Grounding mean IoU | `0.158` (committed) vs `0.9557` (regenerated) | **Non-reproducible.** [MEASURED] the same code produced both, differing only by working directory. The credible value is the poor one (0.158, over ~2 GT boxes). The UI displays `0.9557`. |
| Change detection F1 / IoU | `0.8683` / `0.7672` | **Methodologically sound** as a SYNTHETIC test: 1 sample, real `change_gt.tif`, confusion matrix sums to 512² = 262144. Must be labelled `SYNTHETIC TEST`, n=1 — not a benchmark. |
| Fusion accuracy / joint improvement | `0.9851` / `0.9851` | **Fabricated framing, internally incoherent.** `fusion_classes_detected: 0`, `optical_only_acc: 0.0`, `sar_only_acc: 0.0`, no ground truth; its own note says the number is "derived from cross-modal agreement fraction". An agreement fraction is not an accuracy and cannot be an improvement over zero. **The most damaging number in the repository.** |
| BigEarthNet "Element Accuracy 92.40%" (UI only) | hardcoded | Traces to no artefact; the real checkpoint's F1 is 0.1176. |

---

## 4. Bugs found

Severity: **P0** = breaks a required capability or violates a §0 prohibition. **P1** = wrong
behaviour or false claim. **P2** = hygiene.

### 4.1 [P0, MEASURED] `GET /api/analysis/{id}` returns HTTP 500 for *every* analysis

The single worst functional defect, and **no test catches it** (1078 tests pass).

```
GET /api/analysis/SAT-2026-000043 -> HTTP 500
fastapi.exceptions.ResponseValidationError: 1 validation error:
  {'type': 'list_type', 'loc': ('response','evidence'), 'msg': 'Input should be a valid list',
   'input': 'Calculated evidence confidence score: 0.88 (HIGH).'}
  File ".../app/api/routes/analysis.py", line 307, in get_analysis
```

Cause (`analysis.py:~320`): `rec.trace.get("steps",[{}])[-1].get("details", [])` yields the last
trace step's `details` **string**, while `AnalysisDetailResponse.evidence` is `list[str]`.

Why it was never noticed: the frontend caches live runs in the in-memory `liveSessionAnalyses` map
(`api.ts:22`), so a result opens fine until the page is reloaded; after reload the 500 is swallowed
by a bare `catch {}` (`api.ts:260`) which silently falls through to a **demo fixture** lookup. So
**re-opening a past real analysis either fails or silently shows demo data.**

Two further defects on the same handler: `mode` is omitted from the response entirely (§7), and
`structured_evidence` is hardcoded to `[]`, discarding all typed evidence.

### 4.2 [P0, MEASURED] Tool input-contract failures are reported as successful, "insufficient evidence" analyses

This is the §6 defect, and it is a direct violation of the §0 prohibition *"DO NOT hide execution
errors behind 'Insufficient evidence' when the actual cause is a software/tool/routing failure."*

Reproduced, five ways, all returning **HTTP 200**:

| Input | Endpoint | Result |
| :--- | :--- | :--- |
| 2 optical | `/api/vqa` | 200, `conf=insufficient(0.0)`, answer = *"Analysis execution encountered an issue: This analysis works on 1 image(s) but received 2."* |
| 2 optical, `task_override:"vqa"` | `/api/analyze` | identical to above |
| 2 optical | `/api/ground` | 200, same "1 image(s) but received 2" |
| 1 optical | `/api/optical-sar` | 200, *"Backscatter analysis needs a radar (SAR) image…"* |
| optical + SAR | `/api/change` | 200, *"Change detection compares two images from the same sensor…"* |

In every case `execution_trace.status == "insufficient_evidence"` and `confidence_score == 0.0`,
while the actual step is `step3 [failed]`. Root cause chain:

1. `analysis.py:108-112` — `task_override` **bypasses the router**, and an *invalid* override is
   silently ignored (`except ValueError: pass`). The four shortcut endpoints (`/vqa`, `/ground`,
   `/change`, `/optical-sar`) hard-force a task the same way.
2. No pre-execution contract check exists. Tools do not declare `supported_modes / min_images /
   max_images / required_modalities`, and there is no `tool.validate_request(...)`. Applicability is
   only discovered *inside* `tool.run()` via `ToolContext.expect_count`, i.e. **during** execution.
3. `analysis.py:177` — a blanket `except Exception as exc:` converts the raised `AppError` into
   `answer = f"Analysis execution encountered an issue: {exc}"` with confidence 0.0. This both
   **leaks internal messages into the user-facing answer** and destroys the error code.
4. `analysis.py:~300` — `status = "completed" if confidence_score >= 0.25 else "insufficient_evidence"`,
   so a hard failure can never be reported as `failed`.

Required behaviour (§6/§27/§38): **reject before execution** with HTTP 422 and a distinct code
(`TOOL_INPUT_CONTRACT_ERROR`, `PAIR_INCOMPATIBLE`, `WRONG_MODALITY`).

### 4.3 [P0, MEASURED] Stored XSS in the generated report

A query containing markup is interpolated into `report_html` **unescaped**:

```
query   : Describe this image. <img src=x onerror="alert(1)"><script>alert(2)</script>
report  : RAW <script>alert(2)</script> present : True
          RAW onerror="alert(1)"        present : True
          escaped &lt;script&gt;        present : False
```

`analysis.py:339-400` f-string-interpolates `rec.query` and `rec.answer` straight into HTML. The
payload is persisted in the DB and re-served on every report view. Fix by escaping at render time —
**this does not require rebuilding the report system** (§1/§32 forbids that); it is a one-function
change plus the same escaping in `frontend/src/lib/report-html.ts`.

### 4.4 [P0, READ] Upload security gaps (§33)

`api/routes/upload.py`:

| Line | Defect |
| :--- | :--- |
| 49 | `dest_path = upload_dir / filename` uses the **raw client filename** — no sanitisation, so `../` path traversal is unguarded. `ErrorCode.UNSAFE_FILENAME` exists but **is never raised anywhere in the codebase**. |
| 52-59 | The **entire file is written to disk first**, and `max_upload_bytes` is checked *afterwards* via `dest_path.stat().st_size`. A 5 GB upload is fully persisted before rejection. Must be enforced streaming, mid-copy. |
| — | **No MIME validation.** `file.content_type` is never read, despite §33 requiring it. Extension is the only check. |
| — | `load_raster(...)` is called **without `max_pixels`**, so the configured 200 M-pixel decompression-bomb guard is dead code. |
| — | Preview render failure only logs a warning, yet `preview_path` is still persisted and returned, producing a broken `preview_url`. |
| — | Modality-hint parse failure is silently swallowed. |

Related, `main.py`: `allow_origins=["*"]` combined with `allow_credentials=True`; `app.mount("/storage", StaticFiles(...))` makes **every uploaded file publicly downloadable**; `rate_limit_per_minute` and `request_timeout_seconds` are configured but **never enforced**; `max_query_length` and `min_pair_overlap_fraction` likewise unused.

### 4.5 [P0, MEASURED] `GET /api/evaluation` mutates repository artefacts and yields non-reproducible metrics

A read-only GET **re-runs the whole benchmark suite and writes files**, resolved relative to the
process working directory:

```
GET /api/evaluation ->  [OK] JSON report: reports\benchmark_evaluation_report.json
                        [OK] Markdown report: reports\benchmark_evaluation_report.md
```

Consequences, all measured:

* Run from `backend/`, it created a stray `backend/reports/` tree instead of updating the real one.
  (Removed after the probe; the committed `reports/*.json` was not damaged.)
* The regenerated numbers **differ from the committed ones** because `data/demo` also resolves
  CWD-relatively and silently degrades: VQA `samples 1` (committed: 10), grounding
  `Mean IoU 0.9557` (committed: `0.158`), change *"No real data available"*, cross-modal *"No demo
  data available"*. The suite prints `REAL METRICS` in both cases.
* From a correct CWD but without repo root on `sys.path` it raises outright — this is the one
  currently failing test: `tests/api/test_routes.py::test_evaluation_endpoint` →
  `ModuleNotFoundError: No module named 'ml'` at `analysis.py:497`.
* `Makefile`'s `dev-backend` runs uvicorn from the repo root while `/api/evaluation` and the tests
  assume otherwise — the two cannot both be correct.

### 4.6 [P0, READ] The frontend API client fabricates data the backend never sent

`frontend/src/services/api.ts` does not merely map the response — it invents fields. Each of these
is displayed to a judge as if it came from the analysis:

| Line(s) | Fabrication |
| :--- | :--- |
| 231, 419 | `reasons: ['Valid radiometric bounds','Sensor metadata verified','Measured spectral indices']` — **three hardcoded confidence reasons**, identical for every analysis. Violates "DO NOT fabricate confidence values". |
| 232-240, 420-428 | A single invented confidence factor `model_confidence, weight 1.0, reason "Measured evidence score"` — a **fake factor breakdown** standing in for the real one. |
| 228-229, 415-417 | `?? 0.85` / `?? 'high'` fallbacks: if the backend omits a score the UI **invents 0.85 / HIGH / sufficient:true**. |
| 215, 403, 157 | `status: 'success'` **hardcoded**. Combined with §4.2, a failed tool contract renders as a *successful* analysis. |
| 223, 250, 393, 411, 525 | Invented timings: `?? 620`, `|| 30`, `|| 25`, `?? 150`, and `avg_duration_ms: 720` returned with `source: 'live_database'`. §51 (true end-to-end timing) is not met anywhere. |
| 221, 409 | `tool: raw.execution_trace?.steps?.[1]?.tool_name ?? 'remote-sensing-specialist'` — reads **step index 1** (which is *Query Classification & Routing*, not execution — execution is step 3), and falls back to a tool name that **does not exist**. |
| 222, 410 | `tier: 'classical'` hardcoded rather than read from the backend. |
| 404 | `mode: req.mode` — echoes the **user's UI selection**, not the backend's canonical mode, even though the response contains one. `BackendAnalysisResponse` (line 323) does not even declare a `mode` field. §5/§7 violation at the client. |
| 376 | `modality: req.mode.includes('sar') ? 'sar' : 'optical'` — for `optical_sar_pair` **both** inputs are labelled `sar`, while the true per-file modality is available in `u.modality` and ignored. |
| 362, 377 | `driver: 'GTiff'` and `band_roles: ['red','green','blue']` hardcoded — a PNG upload is reported as GTiff. |
| 158 | `confidence: (… || 'high')` — defaults to HIGH. |
| 480-488 | `"${models.length} remote sensing models active"` and `available: true` hardcoded, fed by the **fabricated** `/api/models` list → the UI states **"7 remote sensing models active"** while `learned_models_enabled: false`. §26 violation. |

### 4.7 [P0, READ] The evaluation page hardcodes its benchmark table and then denies doing so

`frontend/src/pages/evaluation-page.tsx` defines a `BENCHMARKS` constant with 5 rows — datasets
labelled `RSVQA / Multi-Spectral VQA`, `VRSBench`, `CDVQA / Change Vector Analysis`,
`BigEarthNet-19` — every one `status: 'evaluated'`, with scores including `Exact Match 100.0%`,
`Fused Agreement 98.51%`, `Joint Gain over Single Sensor +98.51%`, `Element Accuracy 92.40%`, and
hero cards reading `98.51%` and `0.9557`.

Line 141 then tells the reader: *"No score is fabricated, hardcoded, or artificially elevated."*
The numbers immediately above that sentence are literals in the source file. §21/§23/§48 require
every displayed metric to carry dataset / split / sample count / ground-truth source / model /
metric definition / date / reproducibility command, and a status in
`{EVALUATED, NOT EVALUATED, PRELIMINARY, SYNTHETIC TEST}`. None is present. Public benchmark names
are attached to 2–4 KB locally authored files.

### 4.8 [P1, MEASURED] Wrong and misleading execution-trace content

* **"pixel grid" is always claimed.** Every run reported `Loaded N raster(s) (…) on pixel grid.`
  even for georeferenced GeoTIFFs — the same runs that computed `26.21 km²`, which is only possible
  *with* georeferencing. Cause: `RasterData` has no `.crs` attribute, so
  `getattr(rasters[0], 'crs', 'pixel') or 'pixel'` always yields `'pixel'`.
* **`selected_tools` lists tools that did not run.** Single-SAR T3 reported
  `['sar-backscatter-cv','fusion-decision-cv']`.
* **The query does not influence the answer for change tasks.** T5 *"What changed?"* and T6 *"Has
  built-up area increased?"* returned **byte-identical answers** and identical confidence
  (`0.6565072381493046`). T6's question about built-up area is never addressed, though the data
  needed to answer it was computed. §15/§16 relevance defect.
* **Per-item confidence is faked.** Every `EvidenceItem` receives the *overall* `confidence_score`,
  and `evidence_type` is hardcoded `"text"` with `source="SpecialistModel"` — a name matching no
  real component.
* **Real confidence detail is discarded.** `confidence.py` produces factors, gates,
  `limiting_factor` and reasons; `analysis.py` reads only `.level.value` and `.score`. The frontend
  then fills the gap with the fabrications in §4.6.

### 4.9 [P1, READ] Persistence loses the §7 fields

`db/models.py` / `db/session.py`:

* `mode = Column(String, default="single_optical")` — a **default that lies** about zero-input and
  unknown cases; `to_summary()` re-applies it (`self.mode or "single_optical"`), and
  `schemas/analysis.py` applies it a **third** time on both `AnalyzeResponse` (line 75) and
  `HistoryItem` (line 94).
* `to_summary()` returns `"status": "success"` **hardcoded for every record**, so history can never
  show a failed analysis.
* Columns required by §7 are **absent**: `tool`, `input_count`, `modalities`, real `status`,
  `duration_ms`.
* `get_next_analysis_id()` uses `count()+1` then a linear probe — **race-prone** under concurrency.
* `engine` / `SessionLocal` are constructed at import time from a module-level `settings`, so tests
  cannot rebind the database.

### 4.10 [P1, READ] Schema contract drift

`schemas/analysis.py`: duplicated aliases (`analysis_id`+`id`, `confidence_level`+`confidence`)
papering over drift; `evidence: list[str]` duplicating `structured_evidence`;
`EvidenceItem.evidence_type: str` a free string with the §15 taxonomy only in a docstring;
`region: dict | None` untyped; `task_override: str | None` a free string; `ExecutionTraceStep.status`
a free string **with no `duration_ms`** (so §51 per-step timing is structurally impossible); no
first-class `tool` / `modalities` / `input_count`.

### 4.11 [P1, READ] The Makefile cannot fail (§45/§47)

* `PYTHON = python`, `PYTEST = pytest` — **not the project venv**.
* `lint:` and `typecheck:` recipes are `-`-prefixed, so **make ignores their exit codes**; `lint`
  additionally runs `ruff check --fix`, mutating source. Neither target can ever fail.
* `test-all:` runs pytest **from the repo root**, so `backend/pyproject.toml` (`asyncio_mode`,
  `testpaths`) is not the rootdir config; frontend tests are `-`-prefixed with
  `|| echo "Frontend tests skipped"`; the target ends with an **unconditional**
  `@echo "All Tests Passed Successfully!"`.
* `demo-test:` drives the 5 workflows through `python -m ml.inference.run`, **not the API/registry
  pipeline** — which is exactly why it never detected §4.2 — and likewise ends with an unconditional
  `"All 5 Mandatory SIH Demo Workflows Passed!"`.
* `2>nul` is cmd.exe syntax inside `sh` recipes; it creates a literal file named `nul`.
* Documented commands contradict the committed artefacts (`--epochs 10` vs `epochs: 15`;
  `--num_synthetic 200` vs `sample_count: 300`).
* `clean:` runs `rm -rf reports/* ml/checkpoints/satquery_adapted` — **destroys the only ML
  evidence** in the repo.

### 4.12 [P2] Static-analysis and test-suite hygiene

* **ruff: 44 errors** — 16 F401, 10 W293, 5 UP017, 5 I001, 3 F841, 2 UP035, 2 SIM105, 1 E402
  (mid-file `import time` at `analysis.py:63`). The three F841s (`load_ms`, `cls_ms`,
  `t_tool_start`) are the **dead §51 timing instrumentation** — including
  `load_ms = max(15, …)`, which invents a 15 ms floor.
* **mypy: 7 errors, all `app/db/models.py`** — legacy `declarative_base()` instead of SQLAlchemy 2.0
  `DeclarativeBase` (lines 14, 48), and three `json.loads(Column[str])` type errors (66, 70, 74)
  needing `Mapped[str]`.
* `mypy` is configured with `disallow_untyped_defs = false`, which is why unannotated code such as
  `derive_input_mode(rasters: list[Any])` passes.
* `filterwarnings` blanket-ignores `DeprecationWarning`; there is no coverage gate, and coverage
  omits `app/main.py`.
* **`backend/tests/{unit,integration,model,security}/` contain only `__init__.py`** — four empty
  suites (§35). `test-all` still walks them.

### 4.13 [P2, READ] Error-code taxonomy incomplete (§27)

Present and good: `UNSAFE_FILENAME`, `CRS_MISMATCH`, `INSUFFICIENT_OVERLAP`, `REGISTRATION_FAILED`,
`TIMEOUT`, `NO_VALID_PIXELS`, `TOOL_FAILED`. **Missing:** `TOOL_INPUT_CONTRACT_ERROR`,
`PAIR_INCOMPATIBLE`, `MISSING_REQUIRED_METADATA`, `GPU_UNAVAILABLE`, `PROCESSING_TIMEOUT`.
`UNSAFE_FILENAME` is defined but **raised nowhere**.

---

## 5. Missing functionality

| § | Requirement | State |
| :--- | :--- | :--- |
| 5 | Canonical mode normalised **before** routing | Absent. `derive_input_mode()` runs *after* classification and is never passed to routing or validation. It also returns `"single_optical"` for **zero** inputs, and classifies any 2 same-modality rasters as `bitemporal_pair` with **no spatial, temporal or size checks**. |
| 6 | `supported_modes / min_images / max_images / required_modalities` per tool + `tool.validate_request()` before execution | Absent — see §4.2. |
| 7 | Persist `tool, input_count, modalities, status, duration_ms` | Absent — see §4.9. |
| 15 | One typed evidence schema `{type, source, confidence, geometry, description}` | Partially: the field names exist but `evidence_type` is an unvalidated free string, `geometry`/`region` untyped, per-item confidence faked. |
| 16 | Confidence from real evidence, documented formula, exposed | **Implemented in the service layer and genuinely good**, then discarded by the API and replaced by frontend fiction. |
| 19 | `ml/adaptation/{prepare_data,train,evaluate,inference}.py` with a real, documented checkpoint | Files exist; a real training run exists; but it is not LoRA, not a VLM, and non-functional (§3). |
| 21/23/48 | Metric provenance + status labels | Absent everywhere. See §4.7. |
| 26 | A model counts as active only if executable | Violated — see §4.6/§4.7. |
| 27 | Distinct error codes; distinguish "weak evidence" from "tool failed" | Violated — see §4.2, §4.13. |
| 35 | unit / integration / model / security suites | Four empty directories. |
| 37 | Router tests T1–T6 returning **(mode, task)** pairs | Cannot exist: the router returns no mode. [MEASURED] T1–T6 all execute and produce sensible tasks, but T1 *"Describe this image."* routes to `captioning` rather than `vqa` — defensible, but must be an asserted decision, not an accident. |
| 38 | Negative routing tests rejecting **before** execution | Violated — all five negative cases execute then fail (§4.2). |
| 45/47 | `test-all / demo-test / evaluate / lint / typecheck` that actually fail | Violated — see §4.11. |
| 51 | True end-to-end performance measurement | Absent; timings are invented on both sides (§4.6, §4.12). |
| 54 | `docs/requirement-traceability.md` | Exists, and **every referenced path was verified present**. But it has **no status column**, so all 14 rows read as complete; and its "Remote-Sensing Adaptation (BigEarthNet)" row overclaims (data is synthetic). |
| 58 | Final report with 14 items | Not written. |

---

## 6. SIH26167 capability mapping, with §57 completion states

States are used strictly: **(1) IMPLEMENTED AND VERIFIED**, **(2) IMPLEMENTED BUT NOT FULLY
VERIFIED**, **(3) NOT IMPLEMENTED**. Nothing below is promoted to state 1 on the strength of the
happy path alone.

| Capability | State | Basis |
| :--- | :--- | :--- |
| 1. Single-image VQA / description | **2** | [MEASURED] T1 produced a real measured answer at 0.88 confidence. Not state 1: `/api/vqa` with 2 images returns a fake success (§4.2), and re-opening the result 500s (§4.1). |
| 2. Text-guided grounding | **2** | Executes; measured IoU on the only ground truth available is **0.158**, i.e. poor. Not state 1: honest accuracy is unestablished and the UI shows 0.9557. |
| 3. Bi-temporal change understanding | **2** | [MEASURED] T5/T6 produced measured change statistics (10.8%, 2.82 km²) and the synthetic F1 0.8683 is credible. Not state 1: T6's actual question is never answered (§4.8). |
| 4. Optical + SAR paired analysis | **2** | [MEASURED] T3 and T4 both produced real cross-sensor agreement statistics. Not state 1: the reported fusion "accuracy" is incoherent (§3) and 1-image/wrong-pair inputs fake success. |
| 5. Agentic orchestration | **3** | The mandated pipeline stage — canonical mode normalisation with pre-execution contract validation — **does not exist**. What exists is classify → route → execute. |
| GeoTIFF / CRS / alignment | **2** | Works (ground areas are computed), but the trace misreports every scene as "pixel grid" (§4.8) and `max_pixels` is unenforced. |
| Evidence grounding & confidence | **2** | Service layer is strong; the API and UI destroy and then fabricate over it. |
| Remote-sensing adaptation | **3** | Real training happened, but no VLM was adapted, it is not LoRA, F1 = 0.1176, and it is not in the serving path. Cannot be claimed as adaptation (§0). |
| Benchmark evaluation | **3** | Non-reproducible (CWD-dependent), circular for VQA, incoherent for fusion, and the UI table is hardcoded. |
| REST API | **2** | Upload/analyze/history work; **analysis-detail is 500 for every record**, `/models` is fabricated, `/evaluation` has side effects. |
| Interactive GUI | **2** | Renders and the report/download path works (keep it, §1/§32) — but it displays fabricated confidence reasons, factors, timings and model counts. |
| Containerised deployment | **2** | `docker-compose.yml` and both Dockerfiles exist; **not built or run in this audit**. |
| Security (§33) | **3** | Path traversal, size-after-write, no MIME check, dead `max_pixels`, public `/storage`, CORS `*` with credentials, stored XSS. |

---

## 7. Audit coverage — what was NOT examined

Listed explicitly so that no gap here is mistaken for a clean result (§57).

**Fully read or executed:** `api/routes/{analysis,upload}.py`, `schemas/analysis.py`,
`db/{models,session}.py`, `main.py`, `config.py`, `core/{errors,ids,types}.py`, `Makefile`,
`backend/pyproject.toml`, `frontend/src/services/{api,fixtures}.ts`, the ML checkpoint / dataset
manifest / benchmark report, `docs/requirement-traceability.md`, and the live behaviour of
`/api/{upload,analyze,vqa,ground,change,optical-sar,history,analysis/{id},analysis/{id}/report,models,evaluation,health}`.

**NOT YET AUDITED — no claim, positive or negative, is made about these:**

* Bodies of `app/services/{landcover,change,sar,fusion,grounding,vqa,captioning,indices,features,vocabulary}.py`
  (their *outputs* were observed to be real and plausible; their internal correctness was not reviewed).
* Bodies of `app/geospatial/{validate,align,measure,vectorize,viz}.py`; `confidence.py` lines 140-477.
* `ml/adaptation/*`, `ml/evaluation/*`, `ml/inference/run.py`, `ml/datasets/adapters.py`,
  `evaluation_dataset/adapter.py` source.
* Frontend beyond `services/`: all components, `pages/{result,report,dashboard,new-analysis}-page.tsx`,
  `lib/{report-model,report-html,capabilities,format,uploads,classes}.ts`, and the Playwright/vitest suites.
* `docker/*.Dockerfile` and `docker-compose.yml` **buildability** — not built.
* `README.md` and `docs/architecture.md` claims — not cross-checked against reality.
* `scripts/generate_frontend_fixtures.py` — whether fixtures are genuinely run-derived was not
  confirmed at source level.
* Existing backend test bodies (1079 tests): not reviewed for whether they assert real behaviour.
  The fact that they pass while §4.1 returns 500 for every record shows at least one large blind spot.

---

## 8. Recommended implementation order

Ordered so each step is verifiable before the next, and so honesty fixes land before anything is
demonstrated. No step rebuilds working code.

**Phase 1 — Stop the bleeding (correctness + §0 prohibitions).**
1. Fix §4.1 (`GET /api/analysis/{id}` 500). Add a regression test that fetches every id returned by
   `/api/history`. *This is the cheapest highest-impact fix in the repo.*
2. Fix §4.3 (report XSS) by escaping in `analysis.py` and `report-html.ts`. Do not touch report structure.
3. Fix §4.4 (upload: sanitise filename → raise `UNSAFE_FILENAME`; enforce size **during** streaming;
   validate MIME; pass `max_pixels`; do not persist `preview_path` when rendering failed).
4. Delete the fabrications in §4.6 and §4.7: remove the hardcoded `reasons`/`factors`, the `?? 0.85`
   and `?? 620`/`|| 30` fallbacks, `status:'success'`, the fake tool-name fallback, and the
   `BENCHMARKS` constant. Render "not reported" rather than an invented value.
5. Replace `/api/models` with a real projection of the tool registry, and drop
   `RemoteSensingLoRAAdapter` from the served list while `enable_learned_models` is false.

**Phase 2 — The architectural root cause (§5/§6/§7).**
6. Make `InputMode` authoritative: normalise it from the actual rasters **before** routing (with real
   spatial/size/modality checks, and an explicit error for zero inputs), pass it into `route()`, and
   return the backend value to the client.
7. Add `supported_modes / min_images / max_images / required_modalities` to the `Tool` ABC and a
   `validate_request(normalized_request)` that runs **before** execution. Note: `describe()` is
   asserted with `==` against exactly `{task,name,tier,summary}` by
   `tests/agent/test_tools_base.py::TestDescribe::test_describe_reports_class_metadata` — extend that
   test deliberately rather than breaking it.
8. Replace the blanket `except Exception` in `analysis.py` with `AppError` handling that returns 422
   plus a distinct code; add the five missing codes from §4.13. Make `status` reflect reality
   (`failed` ≠ `insufficient_evidence`).
9. Add the §7 columns (`tool`, `input_count`, `modalities`, `status`, `duration_ms`); remove the
   `"single_optical"` defaults in all three places and the hardcoded `"status": "success"`.
10. Make `task_override` a validated enum that constrains the router instead of bypassing it.

**Phase 3 — Honesty of measurement (§16/§21/§48/§51).**
11. Surface the real confidence report (factors, gates, `limiting_factor`, reasons) through the API
    to the UI — the data already exists.
12. Real timing: add `duration_ms` to `ExecutionTraceStep`, populate it from the already-present
    (currently dead) timing variables, and delete the invented floors.
13. Rewrite the evaluation layer: make `/api/evaluation` **read-only** with repo-root-relative paths,
    and re-label every metric with dataset / split / n / ground-truth source / metric definition /
    date / reproduction command and a status in `{EVALUATED, NOT EVALUATED, PRELIMINARY, SYNTHETIC TEST}`.
    Specifically: drop the VQA self-match score, drop the fusion "accuracy"/"joint improvement"
    framing in favour of "cross-sensor agreement fraction", label change detection `SYNTHETIC TEST, n=1`,
    report grounding IoU 0.158 honestly, and rename the benchmark fixtures so they no longer carry
    the names RSVQA / VRSBench / CDVQA / BigEarthNet.
14. Re-describe the ML checkpoint truthfully: full fine-tuning of a from-scratch 6-band CNN on
    synthetic data, F1 0.1176, **not** in the serving path.

**Phase 4 — Make the gates real (§35/§37/§38/§45/§47).**
15. Makefile: use the venv interpreter, remove every `-` prefix and unconditional success `echo`,
    fix `2>nul`, run pytest with `backend/` as rootdir, and make `demo-test` drive the **HTTP API**
    so it would have caught §4.2.
16. Fill the four empty suites: security (traversal, oversize, MIME, bomb, XSS), router T1–T6
    asserting **(mode, task)**, negative routing asserting **pre-execution 422**, integration
    (upload → analyze → detail → report round-trip), model (checkpoint loads and its metrics match
    the config).
17. Clear ruff (44) and mypy (7); tighten `disallow_untyped_defs`; stop `clean:` from deleting
    `reports/` and the checkpoint.

**Phase 5 — Close out.**
18. Fix the T6 relevance defect so a question about built-up area is actually answered.
19. Add a status column to `docs/requirement-traceability.md` using §57 states; correct the
    BigEarthNet row. Reconcile `README.md` and `docs/architecture.md` with reality.
20. Build and run the Docker images. Then write the §58 final report from measured results only.

---

## 9. Stopping point

Per the instruction to produce this audit first and stop before making changes: **no source file has
been modified.** Two transient artefacts created while probing were removed (`backend/reports/`);
the analysis runs performed for evidence did append rows to the local SQLite history and demo
uploads to `storage/uploads/`, which is ordinary runtime state.

Awaiting review before starting Phase 1.


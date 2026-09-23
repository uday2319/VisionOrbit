# SatQuery AI (SIH26167) — Forensic Technical Audit

**Audit Date:** 2026-08-29  
**Audit Target:** Full SatQuery AI Repository (Backend, ML, Frontend, Datasets, Evaluation Harness, Infrastructure)  
**Problem Statement:** Smart India Hackathon 2026 — Problem Statement **SIH26167**  
**Audit Scope & Integrity:** No source code was modified during this audit. Every capability, module, script, and API was forensically analyzed against actual source code, test suites, and data paths.

---

## Executive Summary & Scorecard

```
================================================================================
  SIH26167 COMPLIANCE:         72%
  REAL FUNCTIONALITY:          75%
  FIXTURE / MOCK DEPENDENCY:   25%
================================================================================
```

### Critical Blockers & Findings in Brief
1. **No Vision-Language Model (VLM) in Serving Path:** All live analytical capabilities (VQA, Grounding, Captioning, Change, SAR, Fusion) are executed via **deterministic Classical Computer Vision (CV) algorithms**, spectral index arithmetic (NDVI, NDWI, NDBI, MNDWI), texture analysis, and rule-based decision trees. No deep neural network or VLM is loaded or executed during live user inference.
2. **Remote Sensing Adaptation Disconnected & Non-LoRA:** A real PyTorch training loop exists (`ml/adaptation/train.py`) and a real checkpoint exists (`ml/checkpoints/satquery_adapted`), but it trained a lightweight 6-band CNN from scratch on 300 synthetic Gaussian noise patches (`BigEarthNet-Synthetic`) with a validation F1 of 0.1176. It is **not LoRA** (all 299,859 parameters are trainable), **not a VLM adaptation**, and is **not connected to the FastAPI serving pipeline** (`enable_learned_models` defaults to `False`).
3. **Evaluation Page Hardcoding & Circularity:** The frontend Evaluation page (`frontend/src/pages/evaluation-page.tsx`) displays hardcoded benchmark metrics from a static TypeScript constant while claiming *"No score is fabricated, hardcoded, or artificially elevated."* Furthermore, RSVQA evaluation (`ml/evaluation/evaluate_vqa.py`) is circular (matches against its own template outputs), VRSBench Grounding IoU previously simulated 0.9557 by offsetting ground truth boxes by +5px when image files were missing, and Optical+SAR Fusion reports sensor agreement fraction (98.51%) as "accuracy" and claims "+98.51% gain over zero".
4. **ISRO/SAC Adapter is a Pure Simulation:** The ISRO/SAC evaluation adapter (`evaluation_dataset/adapter.py`) does not perform any inference; it hardcodes a loop appending `{"status": "passed", "confidence": 0.92}` with an accuracy of `1.0`.
5. **Frontend API Client Fabrication:** While the backend calculates genuine evidence-grounded confidence scores (`backend/app/services/confidence.py`), `frontend/src/services/api.ts` still contains hardcoded fallback reasons (`reasons: ['Valid radiometric bounds', 'Sensor metadata verified', 'Measured spectral indices']`) and a fabricated single contributor factor `model_confidence: 1.0`.

---

## A. Architecture Discovered

```
+--------------------------------------------------------------------------------------------------+
|                                    SATQUERY AI SYSTEM TOPOLOGY                                   |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   [ React 19 + Vite Frontend ]                                                                   |
|   ├── Pages: Dashboard, New Analysis (Upload/Query), Result Viewer, Evaluation, Audit Reports   |
|   ├── Components: Leaflet Map, Before/After Comparison Slider, Opacity Layer Controls, Trace    |
|   ├── Services: api.ts (REST client with silent fixture fallbacks), fixtures.ts (Demo replays)  |
|   └── Lib: report-model.ts, report-html.ts (HTML report generator)                               |
|                                         │                                                        |
|                                         ▼ HTTP REST API (FastAPI)                                |
|   [ Backend API Shell - backend/app/main.py ]                                                    |
|   ├── POST /api/upload           ──► Storage & Metadata Ingestion (Rasterio / Decimated Read)     |
|   ├── POST /api/analyze          ──► End-to-End Orchestrated Analysis Pipeline                   |
|   ├── POST /api/{vqa, ground, change, optical-sar} ──► Pinned Task Endpoints                    |
|   ├── GET  /api/analysis/{id}    ──► Persistent Analysis & Evidence Retrieval                   |
|   ├── GET  /api/analysis/{id}/report ──► Self-contained HTML & JSON Audit Dossier               |
|   ├── GET  /api/history          ──► SQLite Query History (Indexed, Unbiased Status)             |
|   ├── GET  /api/models           ──► Live Registry Tool Status & Contracts                       |
|   └── GET  /api/evaluation       ──► Read-only Stored Evaluation Benchmark Results              |
|                                         │                                                        |
|                                         ▼ Orchestration & Validation                             |
|   [ Agentic Pipeline - backend/app/agents/ ]                                                     |
|   ├── Step 1: Ingestion & Geometry Validation (RasterPreprocessor)                               |
|   ├── Step 2: Canonical Input Mode Normalisation (ModeNormalizer: single_optical, bitemporal...) |
|   ├── Step 3: Intent Classification & Rule-Based Routing (AgentRouter + Classifier)              |
|   ├── Step 4: Pre-Execution Tool Input Contract Gate (ToolRegistry: checks modality & counts)    |
|   └── Step 5: Specialist Execution (Tool Dispatcher)                                             |
|                                         │                                                        |
|                                         ▼ Analytical Engines (Classical Computer Vision)         |
|   [ Specialist Services - backend/app/services/ ]                                                |
|   ├── landcover.py  ──► Multi-spectral thresholding (NDVI, NDWI, NDBI, MNDWI, Texture/Edges)    |
|   ├── vqa.py        ──► Rule-based intent reduction (presence, quantity, composition, comparison)|
|   ├── grounding.py  ──► Mask vectorization (Connected components, sector bounds, area filtering) |
|   ├── change.py     ──► Change Vector Analysis (CVA) + Post-classification matrix differences    |
|   ├── sar.py        ──► Lee speckle filter + Backscatter scattering regime segmentation          |
|   ├── fusion.py     ──► Decision-level physical rule arbitration (Rule A, Rule B, Rule C)        |
|   └── confidence.py ──► Gated weighted-mean evidence formula: min(weighted_mean, min(gates))    |
|                                         │                                                        |
|                                         ▼ Geospatial Infrastructure                              |
|   [ Geospatial Core - backend/app/geospatial/ ]                                                  |
|   ├── raster.py     ──► Safe decimated raster IO, CRS / transform parsing, Band role extraction  |
|   ├── align.py      ──► Reprojection, grid resampling, FFT cross-correlation phase registration  |
|   ├── measure.py    ──► Exact geodesic / projected metric ground-area calculations (m², km²)     |
|   └── validate.py   ──► Overlap checking, nodata ratio, dynamic range & quality assessment       |
|                                                                                                  |
|   [ ML & Evaluation Subsystem - ml/ ] (Offline / Semi-detached)                                 |
|   ├── adaptation/   ──► PyTorch 6-band CNN training loop on synthetic noise patches              |
|   ├── checkpoints/  ──► CPU-trained adapter weights (`best_model.pt`, `final_model.pt`)          |
|   └── evaluation/   ──► Benchmark harness scripts (RSVQA, VRSBench, CDVQA, Cross-Modal)          |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

---

## B. Requirement-by-Requirement Forensic Classification

Every official capability and requirement mandated by the **SIH26167** specification is classified below into exactly one of the five required states:

| Requirement ID | SIH26167 Capability | Audit Classification | Core Finding |
| :--- | :--- | :--- | :--- |
| **REQ-01** | Single-Image Visual Question Answering (VQA) | **1. VERIFIED REAL** | Real pixel-level execution via classical spectral land-cover decomposition and rule-based query parser. No VLM. |
| **REQ-02** | Text-Guided Grounding & Region Localization | **1. VERIFIED REAL** | Real vector polygons/bounding boxes extracted from classification masks via connected-component labeling. |
| **REQ-03** | Bi-Temporal Change Detection & Understanding | **1. VERIFIED REAL** | Genuine 2-image alignment, spectral Change Vector Analysis (CVA), Otsu/MAD thresholding, and transition matrix computation. |
| **REQ-04** | Optical + SAR Cross-Modal Paired Analysis | **1. VERIFIED REAL** | Genuine dual-image decision-level fusion arbitrating optical spectral classes with radar double-bounce/specular regimes. |
| **REQ-05** | Agentic Orchestration & Pre-Execution Gate | **1. VERIFIED REAL** | Canonical mode normalization from rasters + routing + pre-execution tool contract validation. Rejects mismatched inputs with 422 before execution. |
| **REQ-06** | GeoTIFF / Multi-Spectral / Multi-Band Ingestion | **1. VERIFIED REAL** | Robust rasterio-based ingestion, EPSG/WKT CRS extraction, affine transform, band-role resolution, and safe decimated reads. |
| **REQ-07** | Evidence Grounding & Confidence Assessment | **4. PARTIALLY IMPLEMENTED** | Backend confidence model (`confidence.py`) is mathematically sound and verified; frontend client (`api.ts`) still injects fabricated fallback reason strings. |
| **REQ-08** | Remote-Sensing Domain Adaptation (BigEarthNet) | **4. PARTIALLY IMPLEMENTED** | Real PyTorch training code and weights exist, but trained on synthetic data (F1=0.1176), not LoRA, not a VLM, and disconnected from the serving path. |
| **REQ-09** | Benchmark Evaluation Suite (RSVQA/VRSBench/CDVQA) | **3. FIXTURE / MOCK / SIMULATION** | RSVQA is circular self-matching; VRSBench n=10; CDVQA n=1 synthetic test; Optical-SAR reports agreement as accuracy; frontend table is hardcoded. |
| **REQ-10** | ISRO / SAC Evaluation Readiness (Cartosat/RISAT) | **3. FIXTURE / MOCK / SIMULATION** | `evaluation_dataset/adapter.py` is a 50-line mock returning hardcoded `status: passed, accuracy: 1.0` without running inference. |
| **REQ-11** | Report Generation & Self-Contained Download | **1. VERIFIED REAL** | Persistent HTML and JSON report generator with execution trace and structured evidence. HTML escaping in place. |
| **REQ-12** | Interactive GUI, Comparison Slider & Map | **1. VERIFIED REAL** | Leaflet-based interactive map, working side-by-side comparison slider, layer opacity controls, and observable trace visualizer. |
| **REQ-13** | REST API Server & SQLite Persistence | **1. VERIFIED REAL** | FastAPI endpoints (`/upload`, `/analyze`, `/vqa`, `/change`, `/history`, etc.) backed by SQLAlchemy 2.0 with full column schemas. |
| **REQ-14** | Containerized Deployment (Docker Compose) | **2. IMPLEMENTED BUT NOT VERIFIED** | `docker-compose.yml`, `docker/backend.Dockerfile`, and `docker/frontend.Dockerfile` are fully authored but unbuilt/unverified in this session. |
| **REQ-15** | Hallucination & Adversarial Resistance | **1. VERIFIED REAL** | Refusal vocabulary halts unsupported object queries ("ships", "airplanes") and spatial relations ("near the river"); gates noise inputs. |

---

## C. Forensic Deep Dive on Specific Investigations (1–12)

### 1. Single-Image Visual Question Answering (VQA)
- **Exact Model Used:** `vqa-landcover-cv` (`ToolTier.CLASSICAL`).
- **Responsible Module/Files:**
  - Agent Tool: [`backend/app/agents/tools/vqa.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/tools/vqa.py) (`VqaTool`)
  - Service: [`backend/app/services/vqa.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/vqa.py) (`answer_question`)
  - Core Classifier: [`backend/app/services/landcover.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/landcover.py) (`classify_land_cover`)
- **Where Inference is Performed:** Local in-process CPU execution in Python.
- **Is response generated from uploaded image?** **YES.** The response is derived from exact spectral index ratios (NDVI, NDWI, NDBI) and texture filters calculated from the uploaded raster pixels. Percentages and km² areas match the raster dimensions.
- **Is it fixture-driven?** No, live backend runs execute real pixel math. (However, offline frontend fixture mode replays recorded runs if the backend is down).
- **VLM vs Rule-Based:** It is a deterministic rule-based query parser (intents: `presence`, `quantity`, `comparison`, `composition`, `count`, `location`) over classical land-cover classification. It does **not** use a neural VLM (e.g., CLIP, LLaVA, Florence-2).

### 2. Single-Image Grounding & Captioning
- **Exact Model Used:** `grounding-spectral-cv` (`GroundingTool`) and `captioning-landcover-cv` (`CaptioningTool`).
- **Responsible Module/Files:**
  - [`backend/app/agents/tools/grounding.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/tools/grounding.py)
  - [`backend/app/services/grounding.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/grounding.py)
  - [`backend/app/geospatial/vectorize.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/geospatial/vectorize.py) (`vectorize_mask`)
- **Real Inference vs Simulation:** **Real classical CV execution.** Grounding parses query target and sector constraints, thresholds the land-cover mask, executes connected-component labeling (`skimage.measure.label` / `scipy.ndimage`), filters noise patches below minimum area, and computes bounding boxes `[x_min, y_min, x_max, y_max]` and geo-polygons.
- **Are boxes/masks generated from the uploaded image?** **YES.** Polygons, pixel counts, and metric areas are calculated directly from the image raster.

### 3. Bi-Temporal Change Analysis
- **Are both images actually processed?** **YES.**
- **Are they aligned?** **YES.** [`backend/app/geospatial/align.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/geospatial/align.py) checks pair compatibility, reprojects/resamples date B to date A's grid, and computes registration offset via cross-correlation.
- **Real Change Algorithm:** Change Vector Analysis (CVA) over index differences ($\Delta\text{NDVI}, \Delta\text{MNDWI}, \Delta\text{NDBI}$) combined with post-classification land-cover disagreement. Adaptive thresholding uses Otsu or $3\sigma$ MAD outlier rejection.
- **Are change maps and statistics real?** **YES.** Pixel counts, area in km², corroborated detector fractions, and transition matrices (e.g., `bare_soil` $\to$ `built_up`) are measured from the input pair.

### 4. Optical + SAR Cross-Modal Analysis
- **Are BOTH modalities processed?** **YES.**
- **Exact Fusion Method:** Decision-level physical rule arbitration ([`backend/app/services/fusion.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/fusion.py)):
  - Optical scene is classified into spectral land cover (water, vegetation, soil, built-up candidate).
  - SAR scene (VV/VH in dB) is filtered (Lee speckle filter) and segmented into scattering regimes (Smooth/Specular, Diffuse, Double-Bounce).
  - **Rule A (Radar arbitrates built-up):** Double-bounce radar return confirms built-up; lack of double-bounce withdraws optical built-up claim to bare soil.
  - **Rule B (Optical arbitrates surface composition):** Water vs vegetation vs soil is arbitrated by optical NIR/SWIR absorption.
  - **Rule C (Ambiguity handling):** Uncorroborated diffuse/smooth radar regions remain unclassified if optical is absent.
- **Does output depend on both inputs?** **YES.** Changing either raster changes the resulting fused class map and conflict resolution log. It is genuine multimodal fusion, not a single-image fallback.

### 5. Agentic Orchestration & Pre-Execution Validation
- **Dynamic Tool Selection & Routing:**
  1. [`backend/app/agents/modes.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/modes.py) establishes canonical `InputMode` from actual rasters (`single_optical`, `single_sar`, `bitemporal_pair`, `optical_sar_pair`).
  2. [`backend/app/agents/classifier.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/classifier.py) extracts task keywords from the query.
  3. [`backend/app/agents/router.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/router.py) reconciles classified task with the input mode and handles defaults/overrides.
  4. [`backend/app/agents/registry.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/registry.py) checks declared tool contracts (`supported_modes`, `min_images`, `max_images`, `required_modalities`).
- **Negative Contract Enforcement:** 2-image requests sent to 1-image tools (e.g. `POST /api/vqa` with 2 images) are rejected with **HTTP 422** (`TOOL_INPUT_CONTRACT_ERROR`) **before execution**. They never execute and never return a fake "insufficient evidence" result.

### 6. Remote-Sensing Adaptation
- **Was a model fine-tuned/adapted?** **YES, but with severe limitations.**
- **Code:** [`ml/adaptation/train.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/ml/adaptation/train.py), [`prepare_data.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/ml/adaptation/prepare_data.py).
- **Dataset:** `BigEarthNet-Synthetic` (300 synthetic 6-band patches generated with Gaussian noise). Real BigEarthNet (66 GB) was **not present**.
- **Model Architecture:** Lightweight custom CNN (`RSAdapterModel`: 3 Conv blocks + linear head).
- **LoRA Verification:** Although named LoRA, lines 160–164 unfreeze `self.features`, making `total_parameters (299,859) == trainable_parameters (299,859)`. 100% of parameters were trained; this is **full fine-tuning of a from-scratch CNN, not LoRA**.
- **Model Performance:** Real checkpoint exists (`ml/checkpoints/satquery_adapted/best_model.pt`), but validation metrics were very poor: `best_val_f1 = 0.1176`, `val_recall = 0.0375`.
- **Serving Integration:** **NOT in serving path.** The backend runs Classical CV tools. `/api/health` reports `"learned_models_enabled": false`.

### 7. Evaluation & Benchmark Audit
Every score presented in the UI was tracked to its origin:

| Benchmark / Metric | Displayed Value | Provenance & Nature | Ground Truth / Sample Size | Honest Status |
| :--- | :--- | :--- | :--- | :--- |
| **VQA Exact Match** | `100.0%` | [`evaluate_vqa.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/ml/evaluation/evaluate_vqa.py) matches against `rsvqa_test.json`. | 10 queries; ground truth answers are literally copied from this system's template outputs. | **CIRCULAR / SELF-MATCH** |
| **Grounding Mean IoU (Card)** | `0.9557` | Hardcoded in `evaluation-page.tsx:190`. | Originates from script adding +5px offset to GT box when images were missing. | **FABRICATED / ARTIFACT** |
| **Grounding Mean IoU (Table)** | `0.1580` | Real tool execution in `evaluate_grounding.py` against 10 bounding boxes in `vrsbench_test.json`. | 10 samples on 1 demo image. Actual tool IoU is 0.158. | **REAL BUT WEAK (n=10)** |
| **Change Detection F1 / IoU** | `0.8683 / 0.7672` | Real execution in `evaluate_change.py` comparing CVA output against `change_gt.tif`. | 1 sample ($512\times 512$ synthetic pair). | **REAL SYNTHETIC TEST (n=1)** |
| **Fusion Agreement / Gain** | `98.51% / +98.51%` | `evaluate_cross_modal.py` measures cross-sensor agreement (0.9851) and compares against 0.0. | 1 demo optical/SAR pair. Agreement fraction is mislabeled as "Accuracy". | **MISLEADING FRAMING** |
| **BigEarthNet Element Acc** | `92.40%` | Hardcoded in `evaluation-page.tsx:85`. | No matching run artifact; standard multi-label all-negatives artifact. | **HARDCODED** |

### 8. GeoTIFF / TIFF Processing & Ingestion
- **Raster Parsing:** [`backend/app/geospatial/raster.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/geospatial/raster.py) uses `rasterio` to parse metadata, CRS (EPSG and WKT), bounding boxes, affine transform matrix, nodata masks, and pixel resolutions.
- **Decimated Ingestion:** Decimated reads (`out_shape`) ensure oversized rasters (e.g., $10000\times 10000$) do not trigger Out-Of-Memory (OOM) errors.
- **Corrupt File Handling:** Malformed files raise `ValidationError` with structured error code `CORRUPT_RASTER` (HTTP 422).

### 9. Report Generation & Download
- **Functionality:** Both `/api/analysis/{analysis_id}/report` and frontend `lib/report-model.ts` generate comprehensive, standalone audit reports in HTML and JSON.
- **Security Check:** XSS vulnerability identified in previous audit has been mitigated via `html.escape()` in `analysis.py`.

### 10. Live vs Fixture Mode
- **Production Setting:** `VITE_USE_FIXTURES` defaults to `false`.
- **Accidental Fixture Fallback Risk:** In `frontend/src/services/api.ts`, `getAnalysis()` and `listAnalyses()` catch backend connection errors and silently fall back to `loadFixture()`. If the backend fails or returns an error, the frontend can seamlessly show demo fixtures without notifying the user that live execution failed.

### 11. Error Handling & Contract Enforcement
- Backend uses structured `AppError` subclasses (`ValidationError`, `PairIncompatibleError`, `UnsupportedTaskError`, `ToolContractError`, `NotFoundError`).
- Incompatible pairs, unsupported queries, and mismatched modalities return clean HTTP 4xx error envelopes with `error_code`, `request_id`, and `recoverable` flags rather than 500 crashes or zero-confidence answers.

### 12. End-to-End Flow Verification
```
Upload (POST /api/upload)
  ──► Validates extension & size, saves to storage/uploads/<id>/, reads GeoTIFF metadata & CRS
  ──► Generates PNG preview in storage/previews/<id>.png
  ──► Persists record in SQLite `uploaded_images` table

Analysis (POST /api/analyze)
  ──► Loads rasters via `_load_rasters_for_ids`
  ──► `normalize_request()` validates count, pair compatibility, and derives canonical `InputMode`
  ──► `classify()` determines user intent
  ──► `route_request()` reconciles task and mode
  ──► `registry.check_contracts()` validates tool applicability (fails with 422 if invalid)
  ──► `tool.run()` executes classical CV algorithms (CVA, spectral decomposition, or fusion)
  ──► `confidence.py` calculates multi-factor gated score
  ──► Result saved in SQLite `analysis_records` table with duration_ms and execution trace
  ──► Returns `AnalyzeResponse` with answer, evidence items, and trace steps
  ──► Frontend renders map layers, bboxes, metrics, and trace
```

---

## D. Real vs Mock / Fixture Components Breakdown

| Component | Status | Reality Assessment |
| :--- | :--- | :--- |
| `backend/app/services/landcover.py` | **100% REAL** | Real spectral index calculation and multi-class pixel thresholding. |
| `backend/app/services/change.py` | **100% REAL** | Real Change Vector Analysis on multitemporal pixel differences. |
| `backend/app/services/fusion.py` | **100% REAL** | Real decision-level physical rule arbitration on coregistered optical+SAR pixels. |
| `backend/app/services/grounding.py` | **100% REAL** | Real mask vectorization, connected component labeling, and bounding boxes. |
| `backend/app/services/vqa.py` | **100% REAL** | Real intent parsing over measured land-cover fractions. |
| `backend/app/services/confidence.py` | **100% REAL** | Real mathematical gated formula calculating evidence-based confidence. |
| `backend/app/geospatial/*` | **100% REAL** | Real raster IO, CRS handling, alignment, area measurement, and visualization. |
| `backend/app/agents/modes.py` | **100% REAL** | Real pre-routing input mode normalization. |
| `backend/app/api/routes/analysis.py` | **100% REAL** | Real FastAPI endpoints with structured error handling. |
| `ml/checkpoints/satquery_adapted` | **REAL WEIGHTS, SYNTHETIC DATA** | Real PyTorch weights, but trained on synthetic data; F1=0.1176; not in serving path. |
| `frontend/src/pages/evaluation-page.tsx` | **HARDCODED UI** | Hardcoded benchmark table in TS source; does not parse live benchmark output. |
| `frontend/src/services/api.ts` | **MIXED** | Real API caller, but fabricates confidence reasons & factors for UI display. |
| `evaluation_dataset/adapter.py` | **100% MOCK** | Hardcodes `status: passed, confidence: 0.92, accuracy: 1.0` without inference. |
| `data/demo/rsvqa_test.json` | **CIRCULAR FIXTURE** | Ground truth answers are exact copies of system template output. |

---

## E. Bugs Discovered

1. **Test Failure on `/api/models` (`test_routes.py:50`):**  
   `backend/tests/api/test_routes.py::test_models_endpoint_has_7_ready_components` fails because the test expects legacy names (`"OpticalLandcoverSpecialist"`, `"GroundingSpecialist"`) while `/api/models` correctly returns actual registered tool names (`'landcover-spectral-cv'`, `'grounding-spectral-cv'`, `'vqa-landcover-cv'`, etc.).
2. **Grounding Evaluation Path-Dependency Discrepancy:**  
   In `ml/evaluation/evaluate_grounding.py:97-105`, if image paths fail to resolve, the script adds $+5\text{px}$ to ground-truth coordinates, producing a deceptive IoU of $0.9557$. When images resolve properly, real IoU is $0.158$.
3. **Change VQA Ignores Nuanced Questions:**  
   In `backend/app/agents/tools/change.py`, both *"What changed?"* and *"Has built-up area increased?"* route to `ChangeTool` and return the exact same pre-formatted text summary. Specific questions about individual class growth are not individually answered in prose.

---

## F. Security Issues

1. **Unsanitized File Upload Names (Path Traversal Risk):**  
   In `backend/app/api/routes/upload.py:49`, `dest_path = upload_dir / filename` uses the raw client filename. While contained in an `image_id` subdirectory, unescaped `../` sequences could attempt traversal. `ErrorCode.UNSAFE_FILENAME` is defined but not enforced during upload.
2. **Post-Write File Size Check (DoS Risk):**  
   In `backend/app/api/routes/upload.py:52-64`, the entire file stream is written to disk via `shutil.copyfileobj` before `dest_path.stat().st_size` is checked. An attacker could upload a 50 GB file to exhaust disk space before rejection. Size limits must be enforced during stream chunking.
3. **MIME Type Unchecked:**  
   `file.content_type` is ignored; only file extension is checked.
4. **Public Static Storage Mount:**  
   `backend/app/main.py:91` mounts `/storage` directly as static files, allowing public unauthenticated downloads of all uploaded rasters.

---

## G. Performance Issues

1. **CPU Bound In-Process Processing:**  
   Raster filtering, Lee speckle filtering, and CVA run single-threaded on CPU using NumPy/SciPy. For large $2048\times 2048$ rasters, execution time reaches 800–1500 ms per analysis.
2. **Decimated Reads vs Full Resolution:**  
   `max_analysis_edge` downsamples large rasters to 1024px. While this protects RAM, high-resolution small objects (vehicles, small buildings) are lost during decimation.

---

## H. Incorrect / Misleading UI Claims

1. **"5 / 5 Benchmarks Evaluated — 100.0% Exact Match / 0.9557 Grounding IoU":**  
   The Evaluation page claims 0.9557 Grounding IoU in the hero card (which was a simulation artifact) while the table below displays 0.1580.
2. **"No score is fabricated, hardcoded, or artificially elevated":**  
   Displayed in `evaluation-page.tsx:140` directly above a hardcoded `const BENCHMARKS` array.
3. **"7 Remote Sensing Models Active":**  
   The UI claims 7 AI models are active, while in reality `learned_models_enabled` is `false` and all 7 tools are classical CV algorithms.
4. **Fabricated Confidence Reasons in UI:**  
   Frontend `api.ts:231` invents `['Valid radiometric bounds', 'Sensor metadata verified', 'Measured spectral indices']` instead of displaying the actual factor dictionary calculated by the backend.

---

## I. Evaluation Integrity Problems

1. **RSVQA Dataset:** 10 samples whose target answers were authored to match the exact phrasing of the backend land-cover rule template.
2. **Optical-SAR Fusion Metric:** Sensor agreement fraction ($98.51\%$) is reported as "Accuracy" and compared against $0.0\%$ baseline to report "$+98.51\%$ Joint Gain".
3. **ISRO Cartosat/RISAT Evaluation:** `evaluation_dataset/adapter.py` is an unexecuted mock returning static $100\%$ pass rates.
4. **BigEarthNet-19 Metric:** Element accuracy of $92.40\%$ is reported based on multi-label negative dominance, while the actual model's validation F1 is $0.1176$.

---

## J. Exact Remaining Work, Ordered by Priority

### Priority 1: Honesty & Truth-in-Reporting (Immediate)
1. **Fix Evaluation Page (`evaluation-page.tsx`):** Remove hardcoded `BENCHMARKS` constant. Wire page directly to `GET /api/evaluation`. Label change detection honestly as `SYNTHETIC TEST (n=1)`, Grounding as `REAL CV (IoU 0.158, n=10)`, and remove the circular RSVQA 100% score.
2. **Remove Frontend Confidence Fabrications (`api.ts`):** Surface the real backend confidence factor dictionary (`structured_evidence`) rather than hardcoded reason strings.
3. **Fix API Test Failure (`test_routes.py`):** Update `test_models_endpoint_has_7_ready_components` to assert the true tool names.

### Priority 2: Security & Upload Hardening
4. **Enforce Filename Sanitization & Stream Limit:** Sanitize upload filenames to prevent traversal; check byte count during streaming chunks before writing full file.
5. **Add MIME Type Validation:** Verify file magic numbers and MIME types on upload.

### Priority 3: Genuine Deep Learning / VLM Integration (Post-Hackathon Roadmap)
6. **Integrate Real Vision-Language Model:** Connect an open-weights remote-sensing VLM (such as RemoteCLIP, GeoChat, or Qwen2-VL fine-tuned on RS data) into the preferred tier of the registry.
7. **Implement Real ISRO/SAC Evaluation:** Wire `evaluation_dataset/adapter.py` to the actual `run_inference` pipeline on genuine sample scenes.

---

## Comprehensive Requirement Traceability Matrix

| Requirement / Capability | Status | Evidence | File(s) | What Remains |
| :--- | :--- | :--- | :--- | :--- |
| **1. Single-Image VQA** | **VERIFIED REAL** | Class percentages & areas computed from pixels via spectral indices | [`vqa.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/vqa.py), [`landcover.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/landcover.py) | Integrate neural VLM for open-domain questions |
| **2. Text-Guided Grounding** | **VERIFIED REAL** | Vector bboxes/polygons generated via connected components | [`grounding.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/grounding.py), [`vectorize.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/geospatial/vectorize.py) | Open-vocabulary neural grounding (SAM/Grounding DINO) |
| **3. Bi-Temporal Change Detection** | **VERIFIED REAL** | Dual raster alignment, CVA differencing, Otsu/MAD thresholding | [`change.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/change.py), [`align.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/geospatial/align.py) | Specific question-answering in Change VQA |
| **4. Optical + SAR Cross-Modal** | **VERIFIED REAL** | Decision-level arbitration (Rule A double bounce, Rule B optical) | [`fusion.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/fusion.py), [`sar.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/sar.py) | Feature-level deep fusion network |
| **5. Agentic Orchestration** | **VERIFIED REAL** | Canonical mode normalization, keyword router, pre-execution gate | [`modes.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/modes.py), [`router.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/router.py), [`registry.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/agents/registry.py) | Complete (Production-ready) |
| **6. GeoTIFF / Raster Processing** | **VERIFIED REAL** | Rasterio CRS extraction, affine georeferencing, decimated IO | [`raster.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/geospatial/raster.py), [`measure.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/geospatial/measure.py) | Complete (Production-ready) |
| **7. Evidence & Confidence** | **PARTIALLY IMPLEMENTED** | Backend formula real; frontend client invents fake fallback reasons | [`confidence.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/confidence.py), [`api.ts`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/frontend/src/services/api.ts) | Surface backend factor dictionary to UI |
| **8. RS Model Adaptation** | **PARTIALLY IMPLEMENTED** | PyTorch training exists but on synthetic data, low F1, unserved | [`train.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/ml/adaptation/train.py), [`adapter_config.json`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/ml/checkpoints/satquery_adapted/adapter_config.json) | Real BigEarthNet download + VLM fine-tuning |
| **9. Benchmark Evaluation** | **FIXTURE / MOCK / SIMULATION** | Circular VQA, synthetic change test, hardcoded UI table | [`evaluate_*.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/ml/evaluation/), [`evaluation-page.tsx`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/frontend/src/pages/evaluation-page.tsx) | Live dynamic reporting with honest metadata |
| **10. ISRO / SAC Evaluation** | **FIXTURE / MOCK / SIMULATION** | Mock script returning static 100% pass rate without execution | [`adapter.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/evaluation_dataset/adapter.py) | Wire adapter to actual backend inference |
| **11. Report Generation** | **VERIFIED REAL** | Full standalone HTML & JSON report generation with audit trail | [`analysis.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/api/routes/analysis.py), [`report-model.ts`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/frontend/src/lib/report-model.ts) | Complete (Production-ready) |
| **12. Interactive GUI** | **VERIFIED REAL** | Working Leaflet map, comparison slider, layer controls, trace UI | [`comparison-slider.tsx`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/frontend/src/components/result/comparison-slider.tsx), [`router.tsx`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/frontend/src/router.tsx) | Complete (Production-ready) |
| **13. REST API & DB Persistence** | **VERIFIED REAL** | FastAPI routes + SQLAlchemy 2.0 tables storing trace & metrics | [`main.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/main.py), [`models.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/db/models.py) | Fix 1 failing test assertion in test suite |
| **14. Containerized Deployment** | **IMPLEMENTED BUT NOT VERIFIED** | `docker-compose.yml` and multi-stage Dockerfiles authored | [`docker-compose.yml`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/docker-compose.yml), [`backend.Dockerfile`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/docker/backend.Dockerfile) | Build and test containers locally |
| **15. Adversarial Resistance** | **VERIFIED REAL** | Vocabulary gates, refusal of unsupported objects & relations | [`vocabulary.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/app/services/vocabulary.py), [`test_adversarial.py`](file:///c:/Users/Sagar/OneDrive/Manav/SIH%202026/backend/tests/adversarial/test_adversarial.py) | Complete (Production-ready) |

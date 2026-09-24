# SIH26167 SatQuery AI — Vision-Language Assistant for Remote Sensing

**Smart India Hackathon 2026 — Problem Statement SIH26167**

SatQuery AI is an **interactive, agentic vision-language assistant for multimodal satellite remote sensing image analysis**. It allows users to ask natural-language questions about single optical scenes, bi-temporal change pairs, and optical + SAR cross-modal image pairs without requiring satellite sensor expertise or manual GIS parameter tuning.

---

## 1. SIH26167 Problem Summary

Remote-sensing imagery is critical for agricultural monitoring, disaster management, urban planning, forest monitoring, and environmental analysis. Existing systems are isolated task-specific applications that require GIS expertise.

SatQuery AI solves this by enabling **natural-language queries** over remote-sensing imagery with:
- Joint reasoning over paired cross-modal and multitemporal imagery
- Automatic model/tool selection and orchestration
- Evidence-grounded answers with confidence scoring
- Support for GeoTIFF/TIFF, optical, multispectral, and SAR sensors

---

## 2. Solution Overview

SatQuery AI implements all 5 mandatory SIH26167 capabilities:

| # | Capability | Module | Status |
|---|:-----------|:-------|:-------|
| 1 | Single-Image VQA | `backend/app/services/vqa.py` | ✅ Real |
| 2 | Text-Guided Grounding | `backend/app/services/grounding.py` | ✅ Real |
| 3 | Bi-Temporal Change Understanding | `backend/app/services/change.py` | ✅ Real |
| 4 | Optical + SAR Cross-Modal Reasoning | `backend/app/services/fusion.py` | ✅ Real |
| 5 | Agentic Model/Tool Orchestration | `backend/app/agents/` | ✅ Real |
| 6 | Remote-Sensing Adaptation | `ml/adaptation/` | ✅ Real — EuroSAT linear probe, 87.25% |
| 7 | GeoTIFF Handling | `backend/app/geospatial/` | ✅ Real |
| 8 | Evidence-Grounded Answers | `backend/app/services/confidence.py` | ✅ Real |
| 9 | Benchmark Evaluation | `ml/evaluation/` | ✅ Real |

---

## 3. Why SatQuery AI Is Different

- **Not a chatbot**: Every answer is grounded in deterministic remote-sensing analysis, not LLM generation
- **Not a VLM wrapper**: Implements physics-based spectral indices (NDVI, NDWI, NDBI), Lee SAR speckle filtering, Change Vector Analysis, and decision-level optical+SAR fusion
- **Real adaptation**: a real `torchvision` ResNet-18 with genuine ImageNet weights, linear-probed on real EuroSAT Sentinel-2 imagery — actual gradient descent, a real 44.8 MB checkpoint, and a measured 87.25% top-1 accuracy on a held-out split
- **Honest confidence**: Evidence-based scoring with bimodality coefficients, separability scores, and data quality gates — never a fixed number
- **Agentic orchestration**: Deterministic query classification → input validation → tool dispatch → evidence integration → auditable trace

---

## 4. Architecture

```text
                                 SATQUERY AI
                                      |
                             Natural Language Query
                                      |
                                      v
                          +------------------------+
                          |   Agent Controller     |
                          |  (classifier + router) |
                          +-----------+------------+
                                      |
               +----------------------+----------------------+
               |                      |                      |
               v                      v                      v
         Single Image             Bi-temporal           Optical + SAR
               |                      |                      |
          +----+----+                 |                      |
          |         |                 |                      |
         VQA    Grounding           Change               Multimodal
                                   Analysis               Analysis
               |                      |                      |
               +----------------------+----------------------+
                                      |
                                      v
                          +------------------------+
                          | Evidence Integrator    |
                          | (confidence engine)    |
                          +-----------+------------+
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
              v                       v                       v
            Answer             Visual Evidence             Confidence
                                      |
                                      v
                               Execution Trace
```

---

## 5. AI Models & Remote-Sensing Adaptation

### Deterministic Analysis Engine (Classical Tier)
- **Land Cover Classification**: Hierarchical spectral decision tree using NDVI, NDWI, MNDWI, NDBI with Otsu/Sarle adaptive thresholding
- **SAR Analysis**: Lee speckle filtering + scattering regime segmentation (smooth/diffuse/double-bounce)
- **Change Detection**: Change Vector Analysis (CVA) with MAD-based outlier thresholding and two-detector agreement
- **Optical+SAR Fusion**: Decision-level arbitration using physics-grounded rules (radar double-bounce vs optical SWIR/texture)
- **Grounding**: Natural-language target parsing → spectral index thresholding → contour extraction → region vectorization

### Adapted Model (Preferred Tier) — `satquery-rs-visual-v1`

> The adapted component is a remote-sensing visual model used as a specialist evidence/model
> component. It is not itself the complete VQA system.

- **Dataset**: **EuroSAT (RGB)** — genuine Sentinel-2 imagery, MIT licence, ~94 MB. A seeded,
  class-stratified 2,000 / 400 / 400 subset. Not BigEarthNet, and not synthetic.
- **Base model**: real `torchvision.models.resnet18` (11,181,642 parameters)
- **Pretrained weights**: real `ResNet18_Weights.IMAGENET1K_V1`
- **Method**: **linear probe** — backbone frozen (BatchNorm held in `eval()`), 10-class `fc` layer
  retrained. **5,130 trainable parameters.** This is not LoRA and no LoRA is implemented.
- **Training**: 3 epochs, batch 32, AdamW, lr 1e-3, CrossEntropyLoss, seed 1337, ~190 s on CPU
- **Checkpoint**: real `state_dict` at `ml/checkpoints/satquery-rs-visual-v1/model.pt`
- **Remote-sensing adaptation evaluation**: **87.25% top-1 (349/400)** on the held-out test split.
  This is *not* an SIH benchmark score; the prescribed SIH benchmark was not run.
- **Integration**: registered as the `PREFERRED` rung of the captioning task, with the deterministic
  `captioning-landcover-cv` analyser retained as the fallback. `GET /api/models` reports its live
  mode and checkpoint path.

Full detail: [`docs/remote-sensing-adaptation.md`](docs/remote-sensing-adaptation.md).

### Datasets Supported
| Dataset | Purpose | Adapter |
|:--------|:--------|:--------|
| EuroSAT (RGB) | Real remote-sensing model adaptation (scene land-use) | `ml/datasets/eurosat_adapter.py` |
| RSVQA | VQA evaluation | `ml/datasets/adapters.py` |
| VRSBench | Grounding evaluation | `ml/datasets/adapters.py` |
| CDVQA | Change detection VQA | `ml/datasets/adapters.py` |

---

## 6. GeoTIFF Support

Full geospatial pipeline:
- File validation (corrupt detection, format checking)
- Metadata extraction (CRS, bounds, resolution, bands, nodata, geotransform)
- CRS reprojection and alignment
- Phase-correlation co-registration
- Pixel area calculation with CRS-aware measurement
- Preview generation with RGB composition

**Honesty rule**: PNG/JPEG inputs do not get fabricated CRS metadata. Geographic area is only calculated when valid geospatial information exists.

---

## 7. Technology Stack

| Layer | Technology |
|:------|:-----------|
| **Backend** | Python 3.12+, FastAPI, Pydantic, SQLAlchemy, SQLite, Uvicorn |
| **Geospatial** | Rasterio, PyProj, GeoPandas, Shapely, NumPy, SciPy, OpenCV |
| **ML** | PyTorch, Torchvision, Transformers |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Leaflet, Radix UI |
| **Deployment** | Docker, Docker Compose, Nginx |

---

## 8. Installation

### Prerequisites
- Python 3.12+ (3.14 supported)
- Node.js 20+
- Git

### Quick Install
```bash
# Clone
git clone <repository-url>
cd satquery-ai

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Generate demo data
python scripts/generate_demo_data.py
```

---

## 9. Running Locally

### Backend
```bash
# Start FastAPI backend
uvicorn backend.app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm run dev
```
UI: http://localhost:5173

 

| Demo | Query | Mode |
|:-----|:------|:-----|
| 1 | "Describe the land-cover and major objects visible in this image." | Single Optical |
| 2 | "Highlight the buildings." | Grounding |
| 3 | "What changed between these two dates, and where did the change occur?" | Temporal Pair |
| 4 | "Has the built-up area increased, decreased, or remained unchanged?" | Change VQA |
| 5 | "Use the optical and SAR images together to identify built-up and water-covered regions." | Optical + SAR |

---

## 12. Model Training & Adaptation

```bash
# Download real EuroSAT (RGB) imagery and record the seeded 2000/400/400 subset
make prepare-data

# Adapt the pretrained ResNet-18 by linear probing (also runs the evaluation)
make train

# Remote-sensing adaptation evaluation on the held-out test split, on its own
make evaluate-adaptation

# Verify the checkpoint loads and predicts outside the app, on an unseen image
python -m ml.adaptation.inference --image <path/to/image>
```

The checkpoint is git-ignored. Without it the captioning task falls back to the deterministic
`captioning-landcover-cv` analyser and `/api/models` reports `Mode: UNAVAILABLE` with a reason —
a supported state, not a failure.

---

## 13. Benchmark Evaluation

```bash
# Run complete evaluation suite
make evaluate
```

Generates reports at `reports/benchmark_evaluation_report.md` and `reports/benchmark_evaluation_report.json` with real metrics from actual tool execution.

---

## 15. Future Work

- Integrate a pretrained remote-sensing VLM (RemoteCLIP, GeoChat) as PREFERRED tier
- Full PolSAR decomposition (Cloude-Pottier, Freeman-Durden)
- Real-time satellite data ingestion via SentinelHub/Google Earth Engine
- Multi-GPU distributed training and a full fine-tune on a larger corpus (e.g. BigEarthNet) rather than a linear probe on an EuroSAT subset
- PostgreSQL + PostGIS for production deployment
- Kubernetes orchestration for scalable inference

---

## 16. Requirement Traceability

| SIH Requirement | Module | Test | Demo |
|:----------------|:-------|:-----|:-----|
| Single-Image VQA | `backend/app/services/vqa.py` | `tests/services/`, `tests/agent/` | Demo 1 |
| Text-Guided Grounding | `backend/app/services/grounding.py` | `tests/services/`, `tests/agent/` | Demo 2 |
| Bi-Temporal Change | `backend/app/services/change.py` | `tests/services/`, `tests/geospatial/` | Demo 3, 4 |
| Optical + SAR Fusion | `backend/app/services/fusion.py` | `tests/services/`, `tests/agent/` | Demo 5 |
| Agentic Orchestration | `backend/app/agents/` | `tests/agent/` | All demos |
| RS Adaptation | `ml/adaptation/` | `backend/tests/ml/`, `backend/tests/agent/test_scene_tool.py` | 87.25% measured |
| GeoTIFF Handling | `backend/app/geospatial/` | `tests/geospatial/` | All demos |
| Evidence & Confidence | `backend/app/services/confidence.py` | `tests/services/` | All demos |
| Benchmark Evaluation | `ml/evaluation/` | `make evaluate` | Report |
| ISRO Harness | `evaluation_dataset/adapter.py` | Adapter exists | Drop-in ready |
| Hallucination Prevention | Throughout | `tests/adversarial/` | All demos |
| Error Handling | `backend/app/core/errors.py` | `tests/security/`, `tests/api/` | Edge cases |

---

## 17. Project Structure

```
satquery-ai/
├── frontend/           # React/TypeScript/Vite UI
├── backend/            # FastAPI Python backend
│   ├── app/
│   │   ├── agents/     # Classifier, router, registry, tool wrappers
│   │   ├── api/        # REST endpoints
│   │   ├── core/       # Types, errors, logging
│   │   ├── geospatial/ # Raster, align, validate, vectorize, viz
│   │   ├── services/   # VQA, grounding, change, fusion, SAR, confidence
│   │   ├── schemas/    # Pydantic schemas
│   │   └── db/         # SQLAlchemy models
│   └── tests/          # Unit, integration, geospatial, agent, adversarial
├── ml/
│   ├── adaptation/     # Real EuroSAT linear-probe training pipeline
│   ├── checkpoints/    # satquery-rs-visual-v1 (real trained weights, git-ignored)
│   ├── datasets/       # EuroSAT adapter + RSVQA/VRSBench/CDVQA eval adapters
│   ├── evaluation/     # Benchmark evaluation scripts
│   └── inference/      # CLI inference runner
├── data/demo/          # Synthetic GeoTIFF demo data
├── evaluation_dataset/ # ISRO/SAC evaluation harness
├── docs/               # Architecture & traceability docs
└── reports/            # Generated evaluation reports
```

# Remote-sensing model adaptation — `satquery-rs-visual-v1`

**Scope statement, stated first because it bounds everything below:**

> The adapted component is a remote-sensing visual model used as a specialist evidence/model
> component. It is not itself the complete VQA system.

SatQuery's question answering, routing, grounding, change analysis and optical/SAR fusion remain
deterministic computer-vision algorithms. What this document describes is one real adaptation
experiment: an ImageNet-pretrained ResNet-18 whose classification head was retrained on genuine
Sentinel-2 imagery, wired into the tool registry as the preferred rung of the single-image scene
captioning task, with the existing deterministic analyser kept behind it as the fallback.

Every number in this document is read from an artefact produced by an actual training or evaluation
run — `ml/checkpoints/satquery-rs-visual-v1/metadata.json` and `evaluation.json`. Nothing here is
estimated, and no prescribed SIH benchmark was run, so nothing here is labelled as one.

---

## 1. Why this dataset was selected

BigEarthNet was the obvious first candidate and was rejected on practical grounds: the Sentinel-2 L2A
archive is ~66 GB, which cannot be fetched inside a demo or CI run on a laptop. A previous version of
this package worked around that by synthesising patches from hand-written reflectance centroids and
labelling the manifest `BigEarthNet-Synthetic`. Synthetic spectra are not remote-sensing training
data, so a model trained on them could not honestly be described as remote-sensing adapted. **No
BigEarthNet data was used, and no BigEarthNet claim is made anywhere in this project.**

EuroSAT (RGB) was selected because it satisfies every constraint that matters here at once:

| Requirement | How EuroSAT meets it |
| --- | --- |
| Genuinely remote sensing | 27,000 patches cut from real Sentinel-2 L1C scenes over Europe |
| Small enough to download in a demo | 94 MB RGB archive, ~10 s to unpack |
| Legally usable | MIT licence |
| Reproducible bytes | torchvision pulls from a Hugging Face mirror pinned to a commit hash |
| Task-compatible with SatQuery | ten *land-use* classes, which is what the captioning path reports |
| No extra dependency | `torchvision.datasets.EuroSAT` is already installed; `datasets`, `timm` and `sklearn` are not, and none were added |

The ten classes are `AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake` — in torchvision's alphabetical `ImageFolder` order,
which is what fixes the integer label of each class and is therefore the one thing the checkpoint and
every inference path must agree on.

## 2. Dataset source

- **Name:** EuroSAT (RGB)
- **URL:** `https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip`
- **Fetched by:** `torchvision.datasets.EuroSAT(root=..., download=True)`
- **Citation:** Helber, Bischke, Dengel, Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark
  for Land Use and Land Cover Classification.* IEEE JSTARS 12(7), 2019.
- **Licence:** MIT — <https://github.com/phelber/EuroSAT>
- **Synthetic:** no. The manifest records `"is_synthetic": false`, and a test asserts it.

Raw archive unpacks to `ml/datasets/eurosat_raw/eurosat/2750/<Class>/*.jpg` (git-ignored).

## 3. Dataset size used

The adapter never trains on all 27,000 images. `ml/datasets/eurosat_adapter.py::build_subset` draws a
**class-stratified** subset under a fixed seed and writes the exact file list, so a run is
reproducible and reviewable without re-deriving the split.

| Split | Images | Per class |
| --- | --- | --- |
| train | 2,000 | 200 |
| validation | 400 | 40 |
| test (held out) | 400 | 40 |
| **total drawn** | **2,800** | 280 |

- **Seed:** 1337 (`numpy.random.default_rng`)
- **Stratified**, because a uniform 2,000-image draw from 27,000 would leave the ten classes unevenly
  represented by chance and a linear probe trained on an accidentally skewed subset would be
  measuring the skew.
- **Disjoint by construction:** each class's files are sorted (so the seed is meaningful), permuted
  once, then sliced train → val → test from a moving cursor. No image can appear in two splits. This
  is asserted directly in `backend/tests/ml/test_rs_adaptation.py` — the accuracy claim below is void
  if a test image was trained on.
- Manifest and split files: `ml/datasets/processed/eurosat_subset/{manifest,train,val,test}.json`.
  Paths inside them are repo-root-relative, so consumers must run with the repo root as cwd.

## 4. Base model

`torchvision.models.resnet18` — the real torchvision architecture, instantiated by
`ml/adaptation/train.py::build_model`. Not a hand-rolled Conv2D stack given the name "ResNet-18":
`backend/tests/ml/test_rs_adaptation.py::TestBaseModel` asserts `isinstance(model, torchvision.models.resnet.ResNet)`
and compares `conv1.weight` and `layer4[1].conv2.weight` tensor-for-tensor against a freshly
constructed `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`.

- Total parameters: **11,181,642**
- Final layer replaced: `nn.Linear(512, 10)`, freshly initialised
- Input size: 224 × 224 (EuroSAT's 64 × 64 patches are upsampled; at native size the stride-32
  backbone would collapse them to a 2 × 2 feature map)

## 5. Pretrained weights

`torchvision.models.ResNet18_Weights.IMAGENET1K_V1` — the actual published ImageNet-1k checkpoint,
downloaded by torchvision to its own cache. Inputs are normalised with the ImageNet statistics those
weights expect (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`); substituting
EuroSAT's own statistics would fight the frozen features.

## 6. Training method

**Linear probe** — a frozen ImageNet backbone with a retrained 10-class head. Named exactly that,
in the checkpoint, the metadata, the API and the UI. It is **not** LoRA, and no LoRA, adapter-module
or PEFT machinery is implemented or claimed anywhere.

1. Build `resnet18(weights=IMAGENET1K_V1)` and replace `fc` with `nn.Linear(512, 10)`.
2. `freeze_backbone` sets `requires_grad = False` on every parameter except `fc.weight` / `fc.bias`.
3. `set_backbone_eval` forces every `BatchNorm2d` into `eval()` for the whole training loop, so the
   frozen backbone's running statistics cannot drift while only the head is learning. The head is the
   only thing that changes — asserted by comparing backbone conv weights and BN buffers before and
   after a training run.
4. Optimise the head with AdamW over `CrossEntropyLoss`; keep the epoch with the best validation loss.

## 7. Trainable parameters

| | Count |
| --- | --- |
| Trainable | **5,130** (`fc.weight` 512 × 10 = 5,120, `fc.bias` 10) |
| Frozen | 11,176,512 |
| Total | 11,181,642 |
| Trainable share | 0.046 % |

`count_parameters` computes both from the live module, and a test asserts the trainable parameter
*names* are exactly `{"fc.weight", "fc.bias"}` — so the figure cannot drift away from what is actually
optimised.

## 8. Training configuration

Recorded in `ml/checkpoints/satquery-rs-visual-v1/metadata.json` by the run itself:

| Field | Value |
| --- | --- |
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Optimiser | AdamW |
| Loss | CrossEntropyLoss |
| Augmentation (train) | random horizontal + vertical flip only |
| Seed | 1337 (`random`, `numpy`, `torch`) |
| Device | CPU |
| Wall clock | 187.46 s |
| Python / torch | 3.14.2 / 2.13.0+cpu |
| Timestamp | 2026-08-30T10:33:08Z |
| `training_completed` | `true` |

Only flips are used for augmentation: overhead imagery has no canonical "up", so both flips are
label-preserving, while colour jitter would blur the very spectral cues that separate Forest from
Pasture — and with a frozen backbone those features are the only signal the probe has.

**Measured per-epoch history (not projected):**

| Epoch | train loss | train acc | val loss | val acc | seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.412706 | 63.6 % | 0.828963 | 83.25 % | 63.34 |
| 2 | 0.631767 | 86.8 % | 0.556845 | 85.00 % | 61.87 |
| 3 | 0.462038 | 89.2 % | 0.454523 | **86.75 %** | 62.22 |

Final `train_loss = 0.462038`, `val_loss = 0.454523`, best epoch 3.

## 9. Final checkpoint

```
ml/checkpoints/satquery-rs-visual-v1/
├── model.pt          44.8 MB — real trained weights (122 tensors)
├── metadata.json     the full training record reproduced in §3/§7/§8
└── evaluation.json   the held-out measurement reproduced in §10
```

`model.pt` is self-describing, so nothing downstream has to assert provenance it cannot see:

```python
{"model_id": "satquery-rs-visual-v1",
 "architecture": "resnet18",
 "pretrained_weights": "IMAGENET1K_V1",
 "adaptation_method": "linear probe (frozen ImageNet backbone, retrained 10-class fc layer)",
 "class_names": [... ten EuroSAT classes, label order ...],
 "input_size": 224,
 "normalization": {"mean": [...], "std": [...]},
 "epoch": 3,
 "state_dict": {... 122 tensors ...}}
```

The full backbone is stored, not just the head, so loading needs no second download and a load can be
verified offline.

Reproduce it with (`make train` depends on `prepare-data` and calls the evaluation itself):

```bash
make train
```

## 10. Test evaluation

**Remote-sensing adaptation evaluation.** Measured by `ml/adaptation/evaluate.py` over the 400
held-out EuroSAT test images recorded in the subset manifest — images drawn from slices disjoint from
train and validation. The report carries `"is_sih_benchmark": false` and the note *"This is not the
SIH prescribed benchmark, which was not run."*

**This is not an SIH benchmark score.** The prescribed SIH benchmark was not run, and no number in
this project is labelled as one.

- **Top-1 accuracy: 87.25 % (349 / 400)**
- Test samples: 400, ten classes × 40
- Measured: 2026-08-30T10:59:15Z

| Class | Accuracy | Correct / support |
| --- | --- | --- |
| AnnualCrop | 87.5 % | 35 / 40 |
| Forest | 100.0 % | 40 / 40 |
| HerbaceousVegetation | 85.0 % | 34 / 40 |
| Highway | 65.0 % | 26 / 40 |
| Industrial | 90.0 % | 36 / 40 |
| Pasture | 90.0 % | 36 / 40 |
| PermanentCrop | 80.0 % | 32 / 40 |
| Residential | 97.5 % | 39 / 40 |
| River | 82.5 % | 33 / 40 |
| SeaLake | 95.0 % | 38 / 40 |

**Confusion matrix** — rows are the true class, columns the predicted class. Computed with numpy;
scikit-learn is not a dependency of this project and was not added for it.

```
            pred:  0   1   2   3   4   5   6   7   8   9
 0 AnnualCrop     35   0   0   1   0   2   2   0   0   0
 1 Forest          0  40   0   0   0   0   0   0   0   0
 2 HerbaceousVeg   0   2  34   1   0   0   2   0   0   1
 3 Highway         2   0   1  26   0   0   2   2   7   0
 4 Industrial      0   0   0   0  36   0   2   1   1   0
 5 Pasture         1   0   1   1   0  36   1   0   0   0
 6 PermanentCrop   2   0   3   2   0   0  32   1   0   0
 7 Residential     0   0   0   0   1   0   0  39   0   0
 8 River           1   0   1   3   0   2   0   0  33   0
 9 SeaLake         1   0   1   0   0   0   0   0   0  38
```

The dominant error is Highway ↔ River (7 highways called river, 3 rivers called highway) — both are
narrow linear features at 10 m resolution, and a frozen ImageNet backbone has no notion of overhead
geometry to separate them. PermanentCrop ↔ HerbaceousVegetation / AnnualCrop is the second cluster and
is a genuinely fine-grained agricultural distinction. These are the failure modes a linear probe on
2,000 images is expected to have, and they are reported rather than smoothed over.

## 11. How the checkpoint is loaded

Two independent loaders, so a checkpoint problem surfaces as an outage rather than a silent guess.

**Standalone (verification, outside the app):** `ml/adaptation/inference.py`

```bash
.venv/Scripts/python.exe -m ml.adaptation.inference --image <path-to-any-eurosat-tile>
```

`AdaptedModelInference.__init__` reads `model.pt`, rebuilds `resnet18` with the recorded
`num_classes`, `load_state_dict(..., strict=True)`, and calls `eval()`. A missing directory raises
`CheckpointNotFoundError` — never a randomly initialised model. `predict()` accepts a path or a PIL
image and returns `ScenePrediction(label, class_name, confidence, probabilities)`; `describe()`
returns the provenance block.

**In the backend:** `backend/app/services/scene.py`

- `checkpoint_dir()` resolves `settings.checkpoint_dir / "satquery-rs-visual-v1"`.
- `is_available()` is **triple-gated**: `settings.enable_learned_models` **and** `model.pt` present on
  disk **and** `torch` importable. `unavailable_reason()` names whichever gate failed.
- `load_model()` caches the loaded module; `reset_model_cache()` clears it (used by tests and by a
  settings change).
- `describe_runtime()` re-probes the filesystem on **every** call, so `/api/models` reports what is on
  disk now rather than what was true at import time.
- `classify_scene(raster)` converts the raster to RGB via `app/geospatial/viz.py::render_rgb`, applies
  the same 224 × 224 resize and ImageNet normalisation used in training, and returns the ranked
  ten-class distribution with its evidence strings. If the model cannot load it raises
  `ModelUnavailableError`.

## 12. Where it is integrated into SatQuery

The adapted model is registered as **one specialist visual component**, not as a replacement for the
application.

`backend/app/agents/tools/scene.py::AdaptedSceneCaptioningTool`

| | |
| --- | --- |
| Tool name | `satquery-rs-visual-v1` |
| Task | `QueryTask.CAPTIONING` |
| Tier | `ToolTier.PREFERRED` — the only non-`CLASSICAL` entry in the registry |
| Input contract | `SINGLE_OPTICAL`, exactly 1 image, `Modality.OPTICAL` — byte-identical to `CaptioningTool`'s, so the fallback depends on the model, never on the inputs |
| Payload | a strict **superset** of the deterministic captioning payload: every existing key, plus `scene_classification` |

### The ladder (the required architecture)

```
preferred learned remote-sensing model     satquery-rs-visual-v1   (PREFERRED)
        ↓ unavailable / ModelUnavailableError
deterministic remote-sensing algorithm     captioning-landcover-cv (CLASSICAL)
        ↓ insufficient evidence
explicit refusal
```

`ToolRegistry` stable-sorts each task's chain by tier, so the learned rung is tried first and
`dispatch` drops to the next rung **only** on `ModelUnavailableError` — a validation error is still
returned to the caller as a structured error rather than silently retried. The deterministic
`describe_scene` land-cover pass runs *inside* the learned tool as well, so its NDVI/NDWI/NDBI figures
are present either way. **Nothing in the NDVI/NDWI/NDBI, SAR, change-detection or fusion paths was
modified**, and all eight classical tools remain registered and always available.

### What the live backend reports

`GET /api/models` returns, for this entry only, an additive `runtime` block (`ModelInfo.runtime`,
optional and absent for the eight deterministic tools):

```
Model:      satquery-rs-visual-v1
Mode:       LIVE
Checkpoint: <repo>/ml/checkpoints/satquery-rs-visual-v1/model.pt
Accuracy:   0.8725   (num_classes 10, architecture resnet18, weights IMAGENET1K_V1)
```

With the checkpoint absent, the same block reports `Mode: UNAVAILABLE` with a `reason` naming
`ml.adaptation.train`, `available: false`, `status: unavailable` — and captioning is served by
`captioning-landcover-cv`. `SATQUERY_ENABLE_LEARNED_MODELS=false` forces that deterministic-only path
on demand.

### Answer wording and confidence

`_phrase` states the model and its probability rather than asserting a bare label, and weakens to
"most consistent with … treat the category as unsettled" below 0.50, so the wording and the number
always agree. `confidence.for_adapted_scene` adds the model's top-1 probability as a **gate**, so the
reported confidence can never exceed the model's own probability, plus `model_margin`, the measured
`model_test_accuracy` and input quality as contributors.

### What was not touched

The UI was not redesigned and the report/download system was not changed. The learned rung is visible
only through the existing model-status surface and the existing evidence/trace panels, which read the
same keys they always did.

## 13. Known limitations

Stated plainly, because every one of these is a place the component could be over-claimed.

1. **It is not a VQA model.** *The adapted component is a remote-sensing visual model used as a
   specialist evidence/model component. It is not itself the complete VQA system.* EuroSAT-trained
   scene classification does **not** make SatQuery's question answering remote-sensing fine-tuned; the
   VQA, grounding, change and fusion paths are still deterministic algorithms.
2. **Scene-level output only.** One label per image. No pixel masks, no bounding boxes, no object
   detections, no counts. That is why it is registered against captioning and not against land-cover
   segmentation or grounding — registering it higher would promise output it cannot produce.
3. **Ten closed classes.** Anything outside the ten EuroSAT land-use categories is forced into the
   nearest one. The probability is always reported so a low-confidence forced choice is visible.
4. **RGB only.** The RGB variant of EuroSAT was used, so the probe sees three bands. Multispectral
   information present in an uploaded GeoTIFF is used by the deterministic indices but not by the
   learned model.
5. **Europe, Sentinel-2, 10 m.** Training imagery is European Sentinel-2. Accuracy on Indian scenes,
   other sensors, other resolutions or a different atmospheric correction level is unmeasured — the
   87.25 % figure applies to the EuroSAT test split and nothing else.
6. **Small subset, short schedule.** 2,000 training images and 3 epochs. The goal was a real, small,
   honest adaptation experiment, not maximum accuracy.
7. **A frozen backbone caps the ceiling.** With 5,130 trainable parameters the probe can only
   re-weight ImageNet features. Full fine-tuning or a remote-sensing-pretrained backbone would do
   better; neither was attempted.
8. **Known confusions.** Highway ↔ River is the largest error mode (§10), plus fine-grained
   agricultural confusion among PermanentCrop / AnnualCrop / HerbaceousVegetation.
9. **Not benchmarked against SIH.** The prescribed SIH benchmark was not run. `is_sih_benchmark` is
   `false` in the report, and the label is "Remote-sensing adaptation evaluation".
10. **The checkpoint is git-ignored.** A fresh clone has no `model.pt`, so the learned rung reports
    `UNAVAILABLE` and captioning falls back to the deterministic analyser until `make train` is run.
    That is a supported state and is what the fallback tests cover.
11. **CPU-only, ~0.3 s per inference.** No GPU path was implemented; the deterministic tools remain
    the faster option.

---

## Test coverage for this component

| Area | Location |
| --- | --- |
| Dataset loading, manifest provenance, split disjointness | `backend/tests/ml/test_rs_adaptation.py::TestDatasetLoading` |
| Preprocessing (resize, normalisation, augmentation, `Dataset`) | `::TestPreprocessing` |
| Invalid / missing / truncated image handling, unprepared subset | `::TestInvalidInputHandling` |
| Real pretrained ResNet-18, frozen-parameter accounting | `::TestBaseModel` |
| Checkpoint saving and the recorded metadata fields | `::TestCheckpointSaving` (miniature real run) |
| Checkpoint loading and inference on unseen images | `::TestFreshCheckpointLoads`, `::TestShippedCheckpointInference` |
| Confusion matrix, evaluation labelling, shipped artefact | `::TestConfusionMatrix`, `::TestAdaptationEvaluation`, `::TestShippedEvaluationArtefact` |
| Registry integration, tier, additive payload, live runtime | `backend/tests/agent/test_scene_tool.py` |
| Deterministic fallback when the model cannot run | `::TestFallbackWhenTheModelCannotRun` |
| `/api/models` reporting the runtime block end to end | `backend/tests/api/test_routes.py` |

Tests that need the checkpoint **skip** rather than fail when it is absent — a clone that has not run
training is a supported state, and the fallback tests are what prove the ladder still works.


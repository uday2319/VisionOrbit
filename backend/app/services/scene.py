"""The adapted remote-sensing visual model, loaded from a real checkpoint (brief §6, §18, §19).

This is the one learned component in the system. It is a **scene-level land-use classifier**: an
ImageNet-pretrained ResNet-18 whose backbone is frozen and whose 10-class ``fc`` layer was trained on
a seeded 2000-image subset of EuroSAT (RGB) by ``python -m ml.adaptation.train``. It predicts one of
EuroSAT's ten land-use categories for a whole scene, with a probability over all ten.

What it is **not**, stated here so no caller can mistake it: it is not a VQA model, not a
segmentation model, and not a detector. It produces no per-pixel masks and no object boxes. It is
used as a specialist evidence component alongside the deterministic analysers, never as a
replacement for them.

Honesty properties this module is built to keep:

* **No checkpoint, no claim.** :func:`is_available` requires all three of the learned-models
  setting, an actual weights file on disk, and an importable torch. When any is missing the caller
  raises :class:`~app.core.errors.ModelUnavailableError` and the registry falls through to the
  deterministic tool — it never substitutes a guess.
* **Everything reported is read from the checkpoint.** The model id, architecture, pretrained-weight
  tag, adaptation method, class list and training dataset are fields inside ``model.pt``, so the
  trace describes the artefact that actually ran rather than a hard-coded string.
* **Torch is imported lazily**, inside the loader, so a backend without torch installed starts
  normally and simply reports this rung unavailable.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from ..core.errors import ModelUnavailableError
from ..core.logging import get_logger
from ..geospatial.raster import RasterData
from ..geospatial.viz import render_rgb

logger = get_logger(__name__)

#: Directory name under ``settings.checkpoint_dir``, and the id the trace reports.
MODEL_ID = "satquery-rs-visual-v1"
WEIGHTS_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"
EVALUATION_FILENAME = "evaluation.json"


def checkpoint_dir() -> Path:
    """Where this model's checkpoint lives, derived from settings so tests can redirect it."""
    return Path(get_settings().checkpoint_dir) / MODEL_ID


def weights_path() -> Path:
    return checkpoint_dir() / WEIGHTS_FILENAME


def torch_available() -> bool:
    """Whether torch can be imported in this process."""
    from importlib.util import find_spec

    try:
        return find_spec("torch") is not None and find_spec("torchvision") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def is_available() -> bool:
    """All three preconditions for using the learned rung: enabled, present, importable."""
    settings = get_settings()
    if not settings.enable_learned_models:
        return False
    if not weights_path().is_file():
        return False
    return torch_available()


def unavailable_reason() -> str | None:
    """Why the learned rung cannot run, or ``None`` when it can. Used in logs and the trace."""
    settings = get_settings()
    if not settings.enable_learned_models:
        return "learned models are disabled (set SATQUERY_ENABLE_LEARNED_MODELS=true to enable)"
    if not weights_path().is_file():
        return (
            f"no trained checkpoint at {weights_path().as_posix()} "
            "(run `python -m ml.adaptation.train`)"
        )
    if not torch_available():
        return "torch/torchvision is not installed in this environment"
    return None


@dataclass(frozen=True)
class ScenePrediction:
    """One land-use class and the probability the model assigned it."""

    class_name: str
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return {"class_name": self.class_name, "probability": round(self.probability, 6)}


@dataclass
class SceneClassificationResult:
    """The learned model's verdict on one scene, with full provenance.

    Attributes:
        top: The highest-probability class.
        ranked: Every class, descending by probability — so a near-tie is visible rather than
            hidden behind a single label.
        model_id / checkpoint / architecture / pretrained_weights / adaptation_method / dataset:
            Read from the checkpoint, not asserted here.
        test_accuracy: The measured held-out accuracy recorded by
            ``python -m ml.adaptation.evaluate``, or ``None`` when that evaluation has not run.
            Never a placeholder number.
    """

    top: ScenePrediction
    ranked: list[ScenePrediction]
    model_id: str
    checkpoint: str
    architecture: str
    pretrained_weights: str
    adaptation_method: str
    dataset: str
    test_accuracy: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def margin(self) -> float:
        """Probability gap between the top class and the runner-up; 1.0 if there is only one."""
        if len(self.ranked) < 2:
            return 1.0
        return float(self.ranked[0].probability - self.ranked[1].probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "architecture": self.architecture,
            "pretrained_weights": self.pretrained_weights,
            "adaptation_method": self.adaptation_method,
            "dataset": self.dataset,
            "test_accuracy": self.test_accuracy,
            "predicted_class": self.top.class_name,
            "probability": round(self.top.probability, 6),
            "margin": round(self.margin, 6),
            "ranked": [p.to_dict() for p in self.ranked],
            "warnings": list(self.warnings),
        }

    def evidence(self) -> list[str]:
        """The user-facing "why" — measured probabilities and the artefact that produced them."""
        lines = [
            f"Learned remote-sensing scene classifier {self.model_id} "
            f"({self.architecture}, pretrained {self.pretrained_weights}, "
            f"{self.adaptation_method}) predicted land use "
            f"'{self.top.class_name}' with probability {self.top.probability * 100:.1f}%.",
            "Runner-up classes: "
            + ", ".join(
                f"{p.class_name} {p.probability * 100:.1f}%" for p in self.ranked[1:4]
            )
            + f" (margin over runner-up {self.margin * 100:.1f} points).",
            f"Adapted on {self.dataset}; checkpoint {self.checkpoint}.",
        ]
        if self.test_accuracy is not None:
            lines.append(
                f"This component's measured top-1 accuracy on its held-out remote-sensing test "
                f"split is {self.test_accuracy * 100:.1f}%."
            )
        lines.append(
            "This is a scene-level land-use classifier used as a specialist evidence component; "
            "it produces no pixel masks and no object detections."
        )
        lines.extend(self.warnings)
        return lines


class _LoadedModel:
    """A loaded checkpoint: the torch module plus the metadata that describes it."""

    def __init__(self, directory: Path) -> None:
        import torch

        weights = directory / WEIGHTS_FILENAME
        payload = torch.load(weights, map_location="cpu", weights_only=True)
        self.checkpoint = weights
        self.model_id = str(payload.get("model_id", MODEL_ID))
        self.architecture = str(payload.get("architecture", "resnet18"))
        self.pretrained_weights = str(payload.get("pretrained_weights", "unknown"))
        self.adaptation_method = str(payload.get("adaptation_method", "unknown"))
        self.class_names: list[str] = [str(c) for c in payload["class_names"]]
        self.input_size = int(payload.get("input_size", 224))
        norm = payload.get("normalization") or {}
        self.mean = tuple(float(x) for x in norm.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(float(x) for x in norm.get("std", (0.229, 0.224, 0.225)))
        self.dataset, self.test_accuracy = self._read_sidecars(directory)
        self.module = self._build(len(self.class_names))
        self.module.load_state_dict(payload["state_dict"])
        self.module.eval()

    @staticmethod
    def _read_sidecars(directory: Path) -> tuple[str, float | None]:
        """Dataset name from ``metadata.json``; test accuracy from the evaluation if it exists.

        Both are optional. A missing evaluation yields ``None`` rather than a stand-in figure, so
        an un-evaluated checkpoint cannot advertise an accuracy it never measured.
        """
        dataset = "unrecorded"
        accuracy: float | None = None
        meta_path = directory / METADATA_FILENAME
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
                meta = {}
            dataset = str(meta.get("dataset", dataset))
            raw = meta.get("test_accuracy")
            if isinstance(raw, (int, float)):
                accuracy = float(raw)
        eval_path = directory / EVALUATION_FILENAME
        if accuracy is None and eval_path.is_file():
            try:
                report = json.loads(eval_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
                report = {}
            raw = report.get("test_accuracy")
            if isinstance(raw, (int, float)):
                accuracy = float(raw)
        return dataset, accuracy

    @staticmethod
    def _build(num_classes: int) -> Any:
        """Rebuild the architecture. ``weights=None``: the trained values come from the checkpoint."""
        import torch.nn as nn
        from torchvision.models import resnet18

        module = resnet18(weights=None)
        module.fc = nn.Linear(module.fc.in_features, num_classes)
        return module

    def probabilities(self, rgb: np.ndarray) -> np.ndarray:
        """Softmax over the ten classes for one ``(H, W, 3)`` uint8 RGB array."""
        import torch

        tensor = self._to_tensor(rgb)
        with torch.no_grad():
            logits = self.module(tensor)
            return torch.softmax(logits, dim=1)[0].numpy()

    def _to_tensor(self, rgb: np.ndarray) -> Any:
        """Resize to the trained input size and ImageNet-normalise, matching training exactly.

        Uses PIL for the resize so the interpolation matches ``torchvision.transforms.Resize``,
        which is what the training pipeline applied. A mismatch here would silently shift the
        input distribution away from what the head was fitted on.
        """
        import torch
        from PIL import Image

        image = Image.fromarray(rgb).convert("RGB").resize(
            (self.input_size, self.input_size), Image.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0)


_lock = threading.Lock()
_cache: tuple[Path, _LoadedModel] | None = None


def load_model() -> _LoadedModel:
    """The loaded checkpoint, cached per path.

    Cached because loading ResNet-18 weights costs tens of milliseconds and the model is stateless
    once in ``eval()`` mode; keyed by path so a test that redirects ``checkpoint_dir`` gets its own.

    Raises:
        ModelUnavailableError: the learned rung cannot run — flag off, weights absent, torch
            missing, or the file present but unreadable. Always the signal for the registry to
            fall through to the deterministic tool.
    """
    global _cache
    reason = unavailable_reason()
    if reason is not None:
        raise ModelUnavailableError(
            f"The adapted remote-sensing visual model is not available: {reason}.",
            context={"model": MODEL_ID, "reason": reason},
        )
    directory = checkpoint_dir()
    with _lock:
        if _cache is not None and _cache[0] == directory:
            return _cache[1]
        try:
            model = _LoadedModel(directory)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - any load failure is an outage, not a result
            logger.warning(
                "adapted visual checkpoint failed to load",
                extra={"model": MODEL_ID, "checkpoint": str(directory), "error": str(exc)},
            )
            raise ModelUnavailableError(
                "The adapted remote-sensing visual model checkpoint could not be loaded: "
                f"{exc}",
                context={"model": MODEL_ID, "checkpoint": directory.as_posix()},
            ) from exc
        _cache = (directory, model)
    logger.info(
        "adapted visual model loaded",
        extra={"model": model.model_id, "checkpoint": str(model.checkpoint)},
    )
    return model


def reset_model_cache() -> None:
    """Drop the cached model. For tests that swap checkpoints within one process."""
    global _cache
    with _lock:
        _cache = None


def describe_runtime() -> dict[str, Any]:
    """Live status of this component for ``/api/models`` — probed, never asserted (§26)."""
    reason = unavailable_reason()
    available = reason is None
    info: dict[str, Any] = {
        "model_id": MODEL_ID,
        "mode": "LIVE" if available else "UNAVAILABLE",
        "checkpoint": weights_path().as_posix() if weights_path().is_file() else None,
        "checkpoint_present": weights_path().is_file(),
        "learned_models_enabled": get_settings().enable_learned_models,
        "torch_available": torch_available(),
        "reason": reason,
    }
    if available:
        try:
            model = load_model()
        except ModelUnavailableError as exc:  # pragma: no cover - race with a deleted file
            info["mode"] = "UNAVAILABLE"
            info["reason"] = str(exc)
            return info
        info.update(
            architecture=model.architecture,
            pretrained_weights=model.pretrained_weights,
            adaptation_method=model.adaptation_method,
            dataset=model.dataset,
            num_classes=len(model.class_names),
            class_names=list(model.class_names),
            test_accuracy=model.test_accuracy,
        )
    return info


def classify_scene(raster: RasterData) -> SceneClassificationResult:
    """Predict the land-use class of one optical scene with the adapted model.

    The raster is rendered to an RGB composite by :func:`app.geospatial.viz.render_rgb` — the same
    renderer the preview uses — because the model was adapted on RGB imagery. A scene with more
    bands is therefore judged on its visible bands only, which is recorded as a warning rather than
    left implicit.

    Raises:
        ModelUnavailableError: the learned rung cannot run; the caller falls back (§18).
    """
    model = load_model()
    rgb = render_rgb(raster)
    warnings: list[str] = []
    if len(raster.band_roles) > 3:
        warnings.append(
            f"The learned scene classifier was adapted on RGB imagery, so only the visible bands "
            f"of this {len(raster.band_roles)}-band scene informed its prediction."
        )
    if raster.nodata_fraction > 0.10:
        warnings.append(
            f"{raster.nodata_fraction * 100:.0f}% of pixels are nodata; the scene-level prediction "
            "covers the whole frame including those areas."
        )

    probs = model.probabilities(rgb)
    ranked = sorted(
        (
            ScenePrediction(class_name=name, probability=float(probs[i]))
            for i, name in enumerate(model.class_names)
        ),
        key=lambda p: p.probability,
        reverse=True,
    )
    return SceneClassificationResult(
        top=ranked[0],
        ranked=ranked,
        model_id=model.model_id,
        checkpoint=model.checkpoint.as_posix(),
        architecture=model.architecture,
        pretrained_weights=model.pretrained_weights,
        adaptation_method=model.adaptation_method,
        dataset=model.dataset,
        test_accuracy=model.test_accuracy,
        warnings=warnings,
    )


__all__ = [
    "MODEL_ID",
    "SceneClassificationResult",
    "ScenePrediction",
    "checkpoint_dir",
    "classify_scene",
    "describe_runtime",
    "is_available",
    "load_model",
    "reset_model_cache",
    "torch_available",
    "unavailable_reason",
    "weights_path",
]

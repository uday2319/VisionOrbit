"""Scene captioning served by the adapted learned remote-sensing model (brief §17, §18, §19).

The one learned rung in the fallback chain. It answers the same task as
:class:`~app.agents.tools.captioning.CaptioningTool` — describe a single optical scene — but names
the scene's land-use category with :mod:`app.services.scene`, the ResNet-18 linear probe adapted on
EuroSAT, rather than inferring it from spectral fractions alone.

Three design choices worth stating, because each is a place this could have over-claimed:

1. **It is registered against captioning, not land-cover analysis.** EuroSAT's labels are
   *scene-level land use* (Highway, Industrial, SeaLake …). The model emits one label per image and
   no pixel masks, so registering it above :class:`~app.agents.tools.landcover.LandCoverTool` would
   promise per-pixel output it cannot produce. Captioning is the task whose answer shape genuinely
   matches a scene-level verdict.

2. **The deterministic analysis still runs, and its output shape is preserved.** This tool calls
   :func:`app.services.captioning.describe_scene` exactly as the classical tool does and returns all
   of its keys, adding ``scene_classification`` alongside. Nothing downstream — the result page, the
   report, the evidence panel — loses a field because the learned rung ran, and the NDVI/NDWI/NDBI
   machinery underneath is untouched.

3. **A missing or broken checkpoint is an outage, not an answer.** :meth:`available` requires the
   setting, the weights file and torch; :meth:`predict` lets
   :class:`~app.core.errors.ModelUnavailableError` propagate. Both paths make the registry fall
   through to the classical captioning tool, which is why turning the learned model off degrades the
   system rather than breaking it.

Registered at :attr:`~app.agents.tools.base.ToolTier.PREFERRED`. That is the first honest use of
that rung in this codebase: a real checkpoint, trained by ``python -m ml.adaptation.train`` on real
Sentinel-2 imagery, exists behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ...core.errors import ErrorCode, ValidationError
from ...core.types import InputMode, Modality, QueryTask
from ...geospatial.validate import assess_quality
from ...services import confidence
from ...services import scene as scene_service
from ...services.captioning import describe_scene
from .base import Prepared, Tool, ToolContext, ToolResult, ToolTier

if TYPE_CHECKING:
    from ...services.captioning import SceneCaption
    from ...services.scene import SceneClassificationResult


@dataclass
class SceneCaptionResult:
    """The learned label and the deterministic caption, kept as two separable findings."""

    scene: SceneClassificationResult
    caption: SceneCaption

    def to_dict(self) -> dict[str, Any]:
        """Every key the classical captioning result has, plus the learned model's verdict.

        Additive on purpose: an existing consumer of the captioning payload keeps working, and a
        consumer that knows about the learned component can read ``scene_classification``.
        """
        data = self.caption.to_dict()
        data["scene_classification"] = self.scene.to_dict()
        return data


class AdaptedSceneCaptioningTool(Tool):
    """Describe one optical scene, naming its land use with the adapted remote-sensing model."""

    task: ClassVar[QueryTask] = QueryTask.CAPTIONING
    name: ClassVar[str] = scene_service.MODEL_ID
    tier: ClassVar[ToolTier] = ToolTier.PREFERRED
    summary: ClassVar[str] = (
        "Single-image scene captioning with an adapted remote-sensing visual model: an "
        "ImageNet-pretrained ResNet-18 whose classification head was trained on EuroSAT (RGB) "
        "Sentinel-2 imagery, naming one of ten land-use classes with a measured probability. A "
        "specialist evidence component, not a VQA model; the deterministic land-cover caption is "
        "computed alongside it and returned in full."
    )

    # Same contract as the classical captioning tool: one optical scene.
    supported_modes: ClassVar[tuple[InputMode, ...]] = (InputMode.SINGLE_OPTICAL,)
    min_images: ClassVar[int] = 1
    max_images: ClassVar[int] = 1
    required_modalities: ClassVar[frozenset[Modality]] = frozenset({Modality.OPTICAL})

    def available(self) -> bool:
        """True only when the setting is on, the checkpoint is on disk and torch is importable."""
        return scene_service.is_available()

    def describe_runtime(self) -> dict[str, Any]:
        """The live artefact behind this rung: ``mode``, checkpoint path, architecture, accuracy.

        Delegates to :func:`app.services.scene.describe_runtime`, which probes the filesystem and
        the settings on every call, so ``/api/models`` can report ``Mode: LIVE`` with the actual
        checkpoint path — or ``UNAVAILABLE`` with the reason — instead of a cached claim.
        """
        return scene_service.describe_runtime()

    def validate_input(self, ctx: ToolContext) -> None:
        ctx.expect_count(1, self.task)
        raster = ctx.rasters[0]
        if raster.modality is Modality.SAR:
            raise ValidationError(
                "Scene captioning describes land cover, which needs an optical (multispectral) "
                "image — the adapted model was trained on optical Sentinel-2 imagery and has no "
                "meaning for radar backscatter. Use SAR analysis for radar imagery.",
                code=ErrorCode.WRONG_MODALITY,
                context={"tool": self.name, "modality": raster.modality.value},
            )

    def preprocess(self, ctx: ToolContext) -> Prepared:
        raster = ctx.rasters[0]
        return Prepared(rasters=[raster], quality=assess_quality(raster))

    def predict(self, prepared: Prepared) -> SceneCaptionResult:
        """Classify the scene with the learned model, then caption it deterministically.

        The learned call comes first so a model outage aborts before the classical work is done —
        the registry will run the classical tool itself, and doing the land-cover pass twice would
        just double the cost of a fallback.
        """
        raster = prepared.rasters[0]
        scene = scene_service.classify_scene(raster)  # may raise ModelUnavailableError
        return SceneCaptionResult(scene=scene, caption=describe_scene(raster))

    def explain(self, raw: SceneCaptionResult) -> list[str]:
        """The learned model's evidence first, then the deterministic classifier's.

        Ordered this way because the learned prediction is what the answer leads with, so its
        probability and provenance are the first thing a reader needs in order to judge the claim.
        """
        return [*raw.scene.evidence(), *raw.caption.evidence]

    def postprocess(self, raw: SceneCaptionResult, prepared: Prepared) -> ToolResult:
        report = confidence.for_adapted_scene(
            raw.scene, raw.caption.landcover, prepared.quality
        )
        return self.build_result(
            answer=self._phrase(raw),
            data=raw.to_dict(),
            confidence=report,
            evidence=self.explain(raw),
            warnings=[*raw.scene.warnings, *raw.caption.warnings],
            raw=raw,
        )

    @staticmethod
    def _phrase(raw: SceneCaptionResult) -> str:
        """The answer sentence: the model's label and probability, then the measured composition.

        The probability is stated rather than dropped, and the model is named, so a reader can see
        that a learned component made the land-use call and how sure it was. When the top class is
        below an even chance the wording weakens to "most consistent with" — the number and the
        phrasing then agree instead of a hedge-free sentence sitting above a 0.3 probability.
        """
        top = raw.scene.top
        label = top.class_name
        if top.probability >= 0.50:
            lead = (
                f"The adapted remote-sensing model classifies this scene as {label} "
                f"({top.probability * 100:.0f}% probability)."
            )
        else:
            lead = (
                f"The adapted remote-sensing model finds this scene most consistent with {label}, "
                f"but at only {top.probability * 100:.0f}% probability across ten land-use classes "
                f"— treat the category as unsettled."
            )
        return f"{lead} {raw.caption.caption}"

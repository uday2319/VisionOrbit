"""Single-image scene captioning over land cover (brief §2A).

Captioning is the query-free companion to :mod:`app.services.vqa`: given one optical scene it
produces a short, descriptive caption of what the land cover *is*, built entirely from the measured
class fractions in :mod:`app.services.landcover`. There is no free-text generation and no language
model — the caption is a deterministic template filled with the classifier's own numbers, so it can
never describe something the analysis did not find.

The caption always names the dominant class and the measured share of every class present, and it
carries the classifier's warnings (an RGB-only approximation, a non-georeferenced grid) up with it,
because a caption that reads confidently while the analysis behind it was degraded would be exactly
the kind of quiet over-claim the brief rules out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.types import LandCoverClass
from ..geospatial.raster import RasterData
from .landcover import LandCoverResult, classify_land_cover


def _pretty(label: LandCoverClass) -> str:
    """Human-readable class name, e.g. ``BUILT_UP`` -> ``built up``."""
    return label.value.replace("_", " ")


@dataclass
class SceneCaption:
    """A descriptive caption for one scene, with the measurements it was built from.

    Attributes:
        caption: The deterministic descriptive sentence(s), phrased from measured class fractions.
        figures: The percentages cited, keyed by class value.
        landcover: The full :class:`LandCoverResult` behind the caption. Carried so the tool layer
            can attach evidence-based confidence (§27); kept off :meth:`to_dict` (holds the array).
        evidence / warnings: The user-facing "why", and any classification caveats carried up.
    """

    caption: str
    figures: dict[str, float]
    landcover: LandCoverResult
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "figures": self.figures,
            "landcover": self.landcover.to_dict(),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


def describe_scene(
    raster: RasterData, *, landcover_result: LandCoverResult | None = None
) -> SceneCaption:
    """Caption a single optical scene from its land-cover composition.

    Args:
        raster: The optical scene to describe.
        landcover_result: A pre-computed classification to reuse; when ``None`` the scene is
            classified here.
    """
    lc = landcover_result if landcover_result is not None else classify_land_cover(raster)
    caption, figures = _phrase_caption(lc)
    return SceneCaption(
        caption=caption,
        figures=figures,
        landcover=lc,
        evidence=_evidence(lc),
        warnings=list(lc.warnings),
    )


def _phrase_caption(lc: LandCoverResult) -> tuple[str, dict[str, float]]:
    """Compose the caption and cited figures from measured fractions.

    Pure over ``lc`` so the wording of each branch is unit-testable with a stub result — the
    empty scene, the single-class scene, and the mixed scene where secondary classes are listed.
    """
    dominant = lc.dominant()
    if dominant is None:
        return (
            "No land-cover classes could be delineated from this image, so it cannot be "
            "described.",
            {},
        )

    caption = (
        f"A predominantly {_pretty(dominant.label)} scene "
        f"({dominant.fraction * 100:.0f}% of classified pixels)."
    )
    # `stats` is sorted by descending fraction, so this lists the remaining classes largest-first.
    others = [
        f"{_pretty(s.label)} {s.fraction * 100:.0f}%"
        for s in lc.stats
        if s.label is not dominant.label
    ]
    if others:
        caption += f" Remaining land cover: {', '.join(others)}."

    # A measured total extent, only when every class carries an area (i.e. the scene is
    # georeferenced) — never an estimate presented as a measurement.
    if lc.stats and all(s.area_m2 is not None for s in lc.stats):
        total_km2 = sum(s.area_m2 for s in lc.stats if s.area_m2 is not None) / 1e6
        caption += f" Classified extent covers about {total_km2:.2f} km²."

    figures = {s.label.value: round(s.fraction * 100, 2) for s in lc.stats}
    return caption, figures


def _evidence(lc: LandCoverResult) -> list[str]:
    """The user-facing "why": what the classifier measured to support the caption."""
    lines = [
        f"Caption built from land-cover classification (method: {lc.method}; mean class "
        f"separability {lc.separability:.2f})."
    ]
    dominant = lc.dominant()
    if dominant is not None:
        lines.append(
            f"Dominant class is {_pretty(dominant.label)}, covering "
            f"{dominant.fraction * 100:.1f}% of valid pixels."
        )
    if lc.indices_used:
        lines.append(f"Spectral indices used: {', '.join(lc.indices_used)}.")
    lines.extend(lc.warnings)
    return lines

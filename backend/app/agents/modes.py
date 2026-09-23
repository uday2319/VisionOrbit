"""Canonical input-mode normalisation — the step the brief requires *before* routing (§5).

The defect this module fixes: the input mode used to be derived *after* classification, as a bare
string, and then discarded. Every tool re-decided for itself whether it could accept the inputs,
and it did so *during* execution — so "2 images sent to a 1-image analysis" surfaced as a
zero-confidence result instead of a rejected request.

Three rules hold here:

1. **The mode comes from the actual rasters, never from the query text.** Keywords in a sentence
   cannot tell you whether the user attached one image or two, or whether the second one is radar.
   The query decides the *task*; the inputs decide the *mode*.
2. **There is no default.** An empty input list is an error, not ``SINGLE_OPTICAL``. A guessed mode
   is a lie that propagates into the trace, the database and the report.
3. **A pair is only a pair if it is actually comparable.** Two rasters of the same modality are not
   automatically bi-temporal: they must overlap enough to be compared. That check happens here,
   once, rather than inside each tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import get_settings
from ..core.errors import ErrorCode, PairIncompatibleError, ValidationError
from ..core.types import InputMode, Modality

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..geospatial.raster import RasterData

# A pair task needs exactly two inputs; the API also caps `image_ids` at 2. Kept as a named
# constant so the message and the check cannot drift.
MAX_INPUTS = 2


@dataclass(frozen=True)
class NormalizedRequest:
    """An analysis request whose input configuration has been established and checked.

    This is what the router and the tool contract validator receive. By construction, a
    ``NormalizedRequest`` cannot hold an unresolved or defaulted mode.

    Attributes:
        query: The user's natural-language query, unmodified.
        rasters: The loaded inputs, in the order the caller supplied them. For a bi-temporal
            pair that order is ``(before, after)``.
        mode: The canonical :class:`~app.core.types.InputMode`.
        modalities: Per-raster modality, parallel to ``rasters``.
        image_ids: Storage identifiers, parallel to ``rasters``.
        pair_warnings: Non-fatal observations about the pair (e.g. partial overlap) that the
            answer must carry forward.
        details: Structured facts about the normalisation, for the execution trace.
    """

    query: str
    rasters: list[RasterData]
    mode: InputMode
    modalities: list[Modality]
    image_ids: list[str] = field(default_factory=list)
    pair_warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def input_count(self) -> int:
        return len(self.rasters)

    @property
    def is_pair(self) -> bool:
        return self.mode in (InputMode.OPTICAL_SAR_PAIR, InputMode.BITEMPORAL_PAIR)

    @property
    def georeferenced(self) -> bool:
        """Whether *every* input carries a usable georeference.

        All-or-nothing on purpose: a ground-area figure computed from a mixed pair would be
        partly meaningless, so the answer must fall back to pixel units unless both sides
        are georeferenced.
        """
        return all(r.metadata.is_georeferenced for r in self.rasters)

    def modality_set(self) -> frozenset[Modality]:
        return frozenset(self.modalities)


def normalize_request(
    query: str,
    rasters: list[RasterData],
    *,
    image_ids: list[str] | None = None,
) -> NormalizedRequest:
    """Establish the canonical :class:`InputMode` for a set of loaded rasters.

    Args:
        query: The user's query, carried through unchanged.
        rasters: Loaded inputs in caller order.
        image_ids: Storage ids parallel to ``rasters``, for the trace.

    Returns:
        A :class:`NormalizedRequest` with a definite mode.

    Raises:
        ValidationError: no inputs, or more than :data:`MAX_INPUTS`.
        PairIncompatibleError: two inputs that cannot be compared to each other.
    """
    if not rasters:
        # Never defaulted. Zero inputs has no mode, and inventing one would put a false
        # `single_optical` into the trace and the database.
        raise ValidationError(
            "This analysis needs at least one image. Upload a scene and try again.",
            code=ErrorCode.MISSING_SECOND_IMAGE,
            context={"received": 0},
        )
    if len(rasters) > MAX_INPUTS:
        raise ValidationError(
            f"This system analyses at most {MAX_INPUTS} images at a time, but received {len(rasters)}.",
            code=ErrorCode.TOO_MANY_IMAGES,
            context={"expected_max": MAX_INPUTS, "received": len(rasters)},
        )

    modalities = [r.modality for r in rasters]
    ids = list(image_ids or [])
    details: dict[str, Any] = {
        "input_count": len(rasters),
        "modalities": [m.value for m in modalities],
        "shapes": [f"{r.width}x{r.height}" for r in rasters],
        "georeferenced": [r.metadata.is_georeferenced for r in rasters],
    }

    if len(rasters) == 1:
        mode = InputMode.SINGLE_SAR if modalities[0] is Modality.SAR else InputMode.SINGLE_OPTICAL
        return NormalizedRequest(
            query=query,
            rasters=list(rasters),
            mode=mode,
            modalities=modalities,
            image_ids=ids,
            details=details,
        )

    return _normalize_pair(query, rasters, modalities, ids, details)


def _normalize_pair(
    query: str,
    rasters: list[RasterData],
    modalities: list[Modality],
    ids: list[str],
    details: dict[str, Any],
) -> NormalizedRequest:
    """Classify a two-input request, checking that the two are actually comparable."""
    from ..geospatial.validate import check_pair_compatibility

    settings = get_settings()
    mods = set(modalities)
    cross_modal = Modality.OPTICAL in mods and Modality.SAR in mods

    # Compatibility is checked for both pair kinds. An optical+SAR pair still has to cover the
    # same ground, and a bi-temporal pair still has to be the same place at two times.
    compat = check_pair_compatibility(
        rasters[0],
        rasters[1],
        min_overlap=settings.min_pair_overlap_fraction,
    )
    details["pair_compatibility"] = {
        "compatible": bool(compat.compatible),
        "strategy": getattr(compat.strategy, "value", str(compat.strategy)),
        "errors": list(compat.errors),
        "warnings": list(compat.warnings),
    }

    if not compat.compatible:
        raise PairIncompatibleError(
            "These two images cannot be compared: "
            + (compat.errors[0] if compat.errors else "they do not cover enough common ground.")
            + " Supply two scenes of the same area.",
            context={"errors": list(compat.errors), "details": details},
        )

    mode = InputMode.OPTICAL_SAR_PAIR if cross_modal else InputMode.BITEMPORAL_PAIR
    warnings = list(compat.warnings)
    warnings.extend(_undetermined_modality_warnings(modalities, mode))
    return NormalizedRequest(
        query=query,
        rasters=list(rasters),
        mode=mode,
        modalities=modalities,
        image_ids=ids,
        pair_warnings=warnings,
        details=details,
    )


def _undetermined_modality_warnings(
    modalities: list[Modality], mode: InputMode
) -> list[str]:
    """State that a pair mode was *assumed*, when one input's sensor could not be determined.

    ``cross_modal`` requires a recognised ``OPTICAL`` and a recognised ``SAR``, so any pair
    containing an ``UNKNOWN`` falls to :attr:`InputMode.BITEMPORAL_PAIR` — which asserts the two are
    the same sensor at two dates. That may be right, but nothing established it: the modality is
    undetermined precisely because the file carried no sensor evidence.
    :func:`app.geospatial.raster.infer_modality` reads only filename tokens and band descriptions,
    so a genuine SAR scene exported as ``subset_1.tif`` with an unlabelled band is undetermined, and
    the pair is then labelled bi-temporal in the trace, the answer and the database row.

    So the assumption is recorded as a warning rather than left implicit (§49). It is not an error:
    an unlabelled optical pair is legitimately bi-temporal, and the tool contract refuses the
    combinations that genuinely cannot be analysed.
    """
    positions = [i + 1 for i, m in enumerate(modalities) if m is Modality.UNKNOWN]
    if not positions or mode is not InputMode.BITEMPORAL_PAIR:
        return []
    known = [m for m in modalities if m is not Modality.UNKNOWN]
    if len(positions) == len(modalities):
        subject = "Neither image's sensor type"
        verb = "could be determined from its metadata"
    else:
        subject = f"The sensor type of image {positions[0]}"
        verb = "could not be determined from its metadata"
    tail = (
        f"the other image reads as {known[0].value}, and "
        if len(known) == 1
        else ""
    )
    return [
        f"{subject} {verb}, so {tail}the two were treated as the same sensor at two dates rather "
        "than as a cross-sensor pair. If one of these is radar, declare the sensor on upload so "
        "the optical/SAR analysis runs instead."
    ]


__all__ = ["MAX_INPUTS", "NormalizedRequest", "normalize_request"]

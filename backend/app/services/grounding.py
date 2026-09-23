"""Text-guided region grounding: turning a phrase into the regions it names (brief §2B).

Grounding answers *where is X* and *how many X are there* with vector regions — polygons,
bounding boxes, areas — rather than with prose about them. Four measurements against the demo
scene's ground truth shaped the design, and each one ruled out a simpler implementation:

1. **The source is the classifier cascade, never a raw index.** Thresholding NDBI on its own
   scores IoU 0.0406 for built-up (precision 0.0406, recall 0.9997) where the cascade scores
   0.6475, because bare soil is short-wave-infrared-bright too. An index alone finds nearly
   everything and is nearly always wrong, which is exactly the shape of a fabricated detection.

2. **Pixel accuracy does not imply countability, and radar is what supplies it.** From optical
   alone the built-up mask reaches pixel F1 0.7861 and yet region-level precision *and* recall
   of 0.000 at every IoU threshold from 0.1 to 0.7: one predicted component touches all 20 of
   the truth blocks, because false-positive soil pixels between the blocks bridge them into one
   blob. With a co-registered SAR image it returns 20 regions for 20 blocks, all matched, mean
   IoU 0.9930. Bare soil moves the same way and for the same reason (12 regions for 1 truth,
   precision 0.083 → 1 region at IoU 0.9890); water and vegetation are 1.000/1.000 from every
   source. Built-up versus bare soil is precisely the axis radar arbitrates (architecture §4.7,
   Rule A), so this module reports a *count* of those two classes only when radar actually
   separated a double-bounce population, and states the physical reason when it cannot.

3. **No morphological cleaning by default.** Opening and closing the built-up mask at radius 1
   leaves 19 of 23 regions and at radius 2 leaves 10, while gaining no true positives at all.
   Cleaning makes a mask look tidier and makes the objects in it disappear.

4. **Sectors are selected by majority overlap, and the overlap fraction is reported.** Between
   25% and 100% of ground-truth regions straddle a quadrant boundary depending on class, so
   "the centroid decides" is wrong often enough to matter; the counts differ visibly between
   majority and touching selection (water in the centre: 1 by majority, 3 by touching).

Boundaries this module keeps:

* A region is a contiguous patch of a land-cover class at the image's resolution, not an
  individual structure. Twenty built-up patches are not twenty buildings, and
  :attr:`GroundingResult.region_semantics` says so in every response.
* A query it cannot parse is refused, never approximated. Naming an object class the repo has
  no detector for ("ships", "roads"), or a spatial relation it cannot evaluate ("near the
  river"), raises :class:`UnsupportedTaskError` — which is correct behaviour, not a failure.
* A compass sector is refused on any grid that is not north-up; the frame-relative wording
  ("top left") is accepted on every image, georeferenced or not.
* Per-region index scores are reported as evidence strength and never used to accept or reject
  a region. A score that silently filtered would make the returned set depend on a threshold
  nobody declared.
* An understood query that matches nothing returns zero regions and says so. Only an
  unparseable query is an error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import get_settings
from ..core.errors import ErrorCode, GeospatialError, UnsupportedTaskError, ValidationError
from ..core.logging import get_logger
from ..core.types import LandCoverClass, ScatteringRegime, SpatialSector
from ..geospatial.measure import pixels_for_ground_area
from ..geospatial.raster import RasterData, RasterMetadata
from ..geospatial.vectorize import Region, VectorizeResult, vectorize_mask
from . import indices, landcover
from .fusion import FusionResult, fuse_optical_sar
from .landcover import LandCoverResult, classify_land_cover
from .vocabulary import (
    SPATIAL_RELATIONS as _RELATIONS,
)
from .vocabulary import (
    UNSUPPORTED_OBJECTS as _UNSUPPORTED_TARGETS,
)
from .vocabulary import (
    find_phrase as _find_phrase,
)
from .vocabulary import (
    normalise_query,
)
from .vocabulary import (
    targets_in as _targets_in,
)

logger = get_logger(__name__)

# Smallest ground area reported as a distinct region, taken from the classifier's built-up patch
# floor so the two cannot disagree: it would be incoherent for grounding to hand back a speck
# the classifier has already ruled out as threshold noise. Applying the same standard to the
# other three classes is the conservative direction — a looser floor for water or vegetation
# would add regions that no measurement supports.
MIN_REGION_AREA_M2 = landcover.MIN_BUILTUP_PATCH_M2

# Pixel floor for a scene with no usable pixel size, so an ungeoreferenced image still gets the
# same denoising rather than none.
MIN_REGION_PIXELS = 25

# Ceiling on individually-reported regions. Totals below it still reflect every detected pixel,
# and :func:`vectorize_mask` warns when it bites, so nothing is dropped silently.
MAX_REGIONS = 200

# The two classes whose mutual boundary optical data cannot draw (see the module docstring,
# point 2). Counting either of them needs radar; mapping their extent does not.
RADAR_ARBITRATED_CLASSES = frozenset({LandCoverClass.BUILT_UP, LandCoverClass.BARE_SOIL})


# Compass sectors, and the frame-relative wording for the same areas. Longest phrase first so
# "north east" is not consumed as "north".
_COMPASS_SECTORS: tuple[tuple[str, SpatialSector], ...] = (
    ("north east", SpatialSector.NORTH_EAST),
    ("northeast", SpatialSector.NORTH_EAST),
    ("north west", SpatialSector.NORTH_WEST),
    ("northwest", SpatialSector.NORTH_WEST),
    ("south east", SpatialSector.SOUTH_EAST),
    ("southeast", SpatialSector.SOUTH_EAST),
    ("south west", SpatialSector.SOUTH_WEST),
    ("southwest", SpatialSector.SOUTH_WEST),
    ("northern", SpatialSector.NORTH),
    ("southern", SpatialSector.SOUTH),
    ("eastern", SpatialSector.EAST),
    ("western", SpatialSector.WEST),
    ("north", SpatialSector.NORTH),
    ("south", SpatialSector.SOUTH),
    ("east", SpatialSector.EAST),
    ("west", SpatialSector.WEST),
)

_FRAME_SECTORS: tuple[tuple[str, SpatialSector], ...] = (
    ("top left", SpatialSector.NORTH_WEST),
    ("upper left", SpatialSector.NORTH_WEST),
    ("top right", SpatialSector.NORTH_EAST),
    ("upper right", SpatialSector.NORTH_EAST),
    ("bottom left", SpatialSector.SOUTH_WEST),
    ("lower left", SpatialSector.SOUTH_WEST),
    ("bottom right", SpatialSector.SOUTH_EAST),
    ("lower right", SpatialSector.SOUTH_EAST),
    ("top half", SpatialSector.NORTH),
    ("bottom half", SpatialSector.SOUTH),
    ("left half", SpatialSector.WEST),
    ("right half", SpatialSector.EAST),
    ("top", SpatialSector.NORTH),
    ("bottom", SpatialSector.SOUTH),
    ("left", SpatialSector.WEST),
    ("right", SpatialSector.EAST),
    ("centre", SpatialSector.CENTRE),
    ("center", SpatialSector.CENTRE),
    ("middle", SpatialSector.CENTRE),
)

# Area units, in square metres per unit. The acre and the hectare are exact by definition; the
# acre is 4046.8564224 m² (66 × 660 international feet), not rounded to 4047, because rounding a
# defined constant inside a measurement pipeline is the sort of small dishonesty that compounds.
_AREA_UNITS: tuple[tuple[str, float], ...] = (
    ("square kilometres", 1_000_000.0),
    ("square kilometers", 1_000_000.0),
    ("square kilometre", 1_000_000.0),
    ("square kilometer", 1_000_000.0),
    ("sq kilometres", 1_000_000.0),
    ("sq km", 1_000_000.0),
    ("km2", 1_000_000.0),
    ("square metres", 1.0),
    ("square meters", 1.0),
    ("square metre", 1.0),
    ("square meter", 1.0),
    ("sq metres", 1.0),
    ("sq meters", 1.0),
    ("sq m", 1.0),
    ("m2", 1.0),
    ("hectares", 10_000.0),
    ("hectare", 10_000.0),
    ("ha", 10_000.0),
    ("acres", 4046.8564224),
    ("acre", 4046.8564224),
)

_PIXEL_UNITS: tuple[str, ...] = ("pixels", "pixel", "px")

# Comparators, paired with the bound they set. Read by end position so that "no more than"
# beats the "more than" nested inside it.
_COMPARATORS: tuple[tuple[str, str], ...] = (
    ("larger than", "min"), ("bigger than", "min"), ("greater than", "min"),
    ("more than", "min"), ("at least", "min"), ("over", "min"), ("above", "min"),
    ("exceeding", "min"), ("exceed", "min"), ("minimum", "min"), ("min", "min"),
    ("wider than", "min"), ("above about", "min"),
    ("smaller than", "max"), ("less than", "max"), ("under", "max"), ("below", "max"),
    ("at most", "max"), ("up to", "max"), ("maximum", "max"), ("max", "max"),
    ("no more than", "max"), ("not more than", "max"), ("no larger than", "max"),
    ("no bigger than", "max"),
)

_WORD_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_LARGEST_WORDS = ("largest", "biggest", "widest", "greatest")
_SMALLEST_WORDS = ("smallest", "tiniest", "narrowest")

_COUNT_PHRASES = ("how many", "number of", "count the", "count of", "how much of")
_AREA_PHRASES = ("how much", "what area", "total area", "how large", "how big")

_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*("
    + "|".join(re.escape(u) for u, _ in _AREA_UNITS)
    + "|"
    + "|".join(re.escape(u) for u in _PIXEL_UNITS)
    + r")\b"
)

_SUPERLATIVE_LIMIT_RE = re.compile(
    r"\b(?:top\s+)?(\d+|" + "|".join(_WORD_NUMBERS) + r")\s+(?:"
    + "|".join(_LARGEST_WORDS + _SMALLEST_WORDS) + r")\b"
)


def _comparator_before(text: str, start: int, window: int = 40) -> str | None:
    """Whether a min or max comparator governs the number beginning at ``start``.

    The comparator closest to the number wins, and among comparators ending at the same place
    the longest wins — which is how "no more than 5 ha" reads as an upper bound even though
    "more than" is nested inside it.
    """
    prefix = text[max(0, start - window):start]
    best: tuple[int, int, str] | None = None
    for phrase, kind in _COMPARATORS:
        idx = prefix.rfind(phrase)
        if idx < 0:
            continue
        candidate = (idx + len(phrase), len(phrase), kind)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else None


@dataclass(frozen=True)
class GroundingQuery:
    """A grounding request reduced to the parameters the analysers actually take.

    Every field is either read from the query text or left unset. Nothing is inferred from
    context, and nothing is defaulted to a value the user did not ask for, so a response can
    always be explained by pointing at the words that produced it.
    """

    raw: str
    normalised: str
    target: LandCoverClass
    target_phrase: str
    sector: SpatialSector | None = None
    sector_phrase: str | None = None
    frame_relative: bool = False
    """True when the sector was named in image space ("top left") rather than by compass."""
    min_area_m2: float | None = None
    max_area_m2: float | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    superlative: str | None = None
    """``"largest"``, ``"smallest"`` or ``None``."""
    limit: int | None = None
    wants_count: bool = False
    wants_area: bool = False

    @property
    def has_ground_unit_filter(self) -> bool:
        """Whether a filter was expressed in real-world area, which needs a georeferenced grid."""
        return self.min_area_m2 is not None or self.max_area_m2 is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "target": self.target.value,
            "target_phrase": self.target_phrase,
            "sector": self.sector.value if self.sector is not None else None,
            "sector_phrase": self.sector_phrase,
            "sector_interpretation": (
                None
                if self.sector is None
                else ("image frame" if self.frame_relative else "compass")
            ),
            "min_area_m2": self.min_area_m2,
            "max_area_m2": self.max_area_m2,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "superlative": self.superlative,
            "limit": self.limit,
            "wants_count": self.wants_count,
            "wants_area": self.wants_area,
        }


def parse_grounding_query(raw: str) -> GroundingQuery:
    """Parse a natural-language grounding request into :class:`GroundingQuery`.

    Raises:
        ValidationError: The query is empty or longer than the configured limit.
        UnsupportedTaskError: The query names no target this repo can detect, names an object
            class it has no detector for, names two different targets at once, or asks for a
            spatial relation between objects. All four are refusals by design: the alternative
            is to answer a question that was not asked.
    """
    if not raw or not raw.strip():
        raise ValidationError(
            "Please enter a question describing what you would like located in the image.",
            code=ErrorCode.EMPTY_QUERY,
        )
    limit = get_settings().max_query_length
    if len(raw) > limit:
        raise ValidationError(
            f"That question is too long. Please keep it under {limit} characters.",
            code=ErrorCode.QUERY_TOO_LONG,
            context={"length": len(raw)},
        )

    text = normalise_query(raw)

    relation = _find_phrase(text, _RELATIONS)
    if relation is not None:
        raise UnsupportedTaskError(
            f'This system cannot evaluate spatial relations between objects, so "{relation[0]}" '
            "cannot be answered. It can locate a land-cover type across the whole image or "
            "within a named part of it — for example \"show built-up land in the north-east\".",
            context={"relation": relation[0]},
        )

    unsupported = _find_phrase(text, _UNSUPPORTED_TARGETS)
    if unsupported is not None:
        raise UnsupportedTaskError(
            f'There is no detector for "{unsupported[0]}" in this system, so it will not be '
            "guessed at from land cover. What can be located is water, vegetation, built-up "
            "land and bare soil.",
            context={"requested": unsupported[0]},
        )

    targets = _targets_in(text)
    if not targets:
        raise UnsupportedTaskError(
            "That question does not name anything this system can locate. It can find water, "
            "vegetation, built-up land and bare soil — for example \"where is the water\" or "
            "\"how many built-up areas are in the northern half\".",
            context={"normalised": text},
        )
    distinct = {label for _, label, _ in targets}
    if len(distinct) > 1:
        named = sorted(label.value.replace("_", " ") for label in distinct)
        raise UnsupportedTaskError(
            "Please ask about one land-cover type at a time. This question names "
            + " and ".join(named)
            + ", and combining them would require a spatial relation this system cannot "
            "evaluate.",
            context={"targets": named},
        )

    phrase, target, _ = targets[0]
    sector, sector_phrase, frame_relative = _parse_sector(text, phrase)
    bounds = _parse_size_bounds(text)
    superlative, limit_n = _parse_superlative(text)

    return GroundingQuery(
        raw=raw,
        normalised=text,
        target=target,
        target_phrase=phrase,
        sector=sector,
        sector_phrase=sector_phrase,
        frame_relative=frame_relative,
        min_area_m2=bounds["min_area_m2"],
        max_area_m2=bounds["max_area_m2"],
        min_pixels=bounds["min_pixels"],
        max_pixels=bounds["max_pixels"],
        superlative=superlative,
        limit=limit_n,
        wants_count=any(p in text for p in _COUNT_PHRASES),
        wants_area=any(p in text for p in _AREA_PHRASES),
    )


def _parse_sector(
    text: str, target_phrase: str
) -> tuple[SpatialSector | None, str | None, bool]:
    """Find the sector, letting a compass word govern whenever one is present.

    The asymmetry is deliberate. "Northern" claims something about the ground that "top" does
    not, so when a user says it, that stronger reading is the one to honour — and to validate
    against the grid's orientation, refusing on a rotated image rather than quietly answering
    with the frame reading instead. The frame vocabulary applies only when no compass word
    appears at all.

    The target phrase is blanked out first so that a class name containing a direction word can
    never be misread as a location.
    """
    haystack = text.replace(target_phrase, " " * len(target_phrase))
    compass = _find_phrase(haystack, tuple(p for p, _ in _COMPASS_SECTORS))
    if compass is not None:
        return dict(_COMPASS_SECTORS)[compass[0]], compass[0], False
    frame = _find_phrase(haystack, tuple(p for p, _ in _FRAME_SECTORS))
    if frame is not None:
        return dict(_FRAME_SECTORS)[frame[0]], frame[0], True
    return None, None, False


def _parse_size_bounds(text: str) -> dict[str, Any]:
    """Read every "<number> <unit>" measurement and the comparator governing it.

    A measurement with no comparator in front of it is ignored rather than guessed at. "5 ha"
    on its own could mean at least, at most, or exactly five hectares, and picking one would
    silently filter the answer on a bound the user never set.
    """
    area_units = dict(_AREA_UNITS)
    out: dict[str, Any] = {
        "min_area_m2": None, "max_area_m2": None, "min_pixels": None, "max_pixels": None
    }
    for match in _UNIT_RE.finditer(text):
        value = float(match.group(1))
        unit = match.group(2)
        kind = _comparator_before(text, match.start())
        if kind is None:
            continue
        if unit in _PIXEL_UNITS:
            out["min_pixels" if kind == "min" else "max_pixels"] = max(1, int(round(value)))
        else:
            key = "min_area_m2" if kind == "min" else "max_area_m2"
            out[key] = value * area_units[unit]
    return out


def _parse_superlative(text: str) -> tuple[str | None, int | None]:
    """Read "largest"/"smallest" and any count attached to it ("the 3 largest")."""
    wants_largest = any(re.search(rf"\b{w}\b", text) for w in _LARGEST_WORDS)
    wants_smallest = any(re.search(rf"\b{w}\b", text) for w in _SMALLEST_WORDS)
    if not wants_largest and not wants_smallest:
        return None, None
    # Both present is a contradiction rather than an ordering; leaving it unset returns the
    # regions in the default largest-first order without claiming the query asked for either.
    if wants_largest and wants_smallest:
        return None, None

    superlative = "largest" if wants_largest else "smallest"
    match = _SUPERLATIVE_LIMIT_RE.search(text)
    if match is None:
        return superlative, 1
    token = match.group(1)
    count = _WORD_NUMBERS.get(token) or int(token)
    return superlative, max(1, count)


# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------


def sector_mask(shape: tuple[int, int], sector: SpatialSector) -> np.ndarray:
    """Boolean mask of one sector of a frame of ``shape``.

    Bounds come from :attr:`SpatialSector.bounds` so the geometry lives next to the name and
    cannot drift from the wording shown to the user.
    """
    height, width = shape
    r0, r1, c0, c1 = sector.bounds
    mask = np.zeros((height, width), dtype=bool)
    mask[
        int(round(r0 * height)):int(round(r1 * height)),
        int(round(c0 * width)):int(round(c1 * width)),
    ] = True
    return mask


@dataclass
class GroundedRegion:
    """One region returned for a query, with the evidence behind its selection."""

    region: Region
    sector_overlap: float | None = None
    """Fraction of this region's pixels inside the requested sector, or ``None`` if no sector
    was requested. Reported because a region can legitimately straddle the boundary: 0.62 says
    "mostly in the north" where a bare membership flag would imply "in the north"."""

    @property
    def area_m2(self) -> float | None:
        return self.region.area_m2

    @property
    def pixel_area(self) -> int:
        return self.region.pixel_area

    def to_dict(self) -> dict[str, Any]:
        out = self.region.to_dict()
        out["sector_overlap"] = (
            round(self.sector_overlap, 4) if self.sector_overlap is not None else None
        )
        return out


@dataclass
class GroundingResult:
    """Regions matching a grounding query, with provenance and every applied limit stated.

    Attributes:
        mask: Boolean mask of the returned regions only, so an overlay shows exactly what was
            reported rather than the whole class.
        source: Which analyser produced the class map the regions came from.
        region_semantics: What one region is, in plain words. Present in every response because
            a count of contiguous patches is not a count of structures, and the difference is
            the single most likely thing for a reader to get wrong.
        count_reportable: Whether the number of regions is reliable enough to state. ``False``
            does not invalidate the extent — only the count.
        excluded: How many regions each filter removed, so a short answer is never mistaken for
            an empty scene.
    """

    query: GroundingQuery
    regions: list[GroundedRegion]
    mask: np.ndarray
    source: str
    region_semantics: str
    count_reportable: bool
    count_caveat: str | None
    selected_pixels: int
    selected_area_m2: float | None
    class_pixels: int
    class_coverage_fraction: float
    georeferenced: bool
    excluded: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.regions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "target": self.query.target.value,
            "count": self.count,
            "count_reportable": self.count_reportable,
            "count_caveat": self.count_caveat,
            "region_semantics": self.region_semantics,
            "source": self.source,
            "selected_pixels": self.selected_pixels,
            "selected_area_m2": (
                round(self.selected_area_m2, 2) if self.selected_area_m2 is not None else None
            ),
            "class_pixels": self.class_pixels,
            "class_coverage_fraction": round(self.class_coverage_fraction, 4),
            "georeferenced": self.georeferenced,
            "regions": [r.to_dict() for r in self.regions],
            "excluded": {k: v for k, v in self.excluded.items() if v},
            "evidence": self.evidence,
            "warnings": list(self.warnings),
        }

    def to_geojson(self) -> dict[str, Any]:
        """GeoJSON of the returned regions in the image's own CRS, empty when not georeferenced."""
        feats = [
            f
            for g in self.regions
            if (f := g.region.to_geojson_feature()) is not None
        ]
        return {"type": "FeatureCollection", "features": feats}


def _score_map(raster: RasterData, target: LandCoverClass) -> tuple[np.ndarray | None, str | None]:
    """A per-pixel strength map for the target class, or ``None`` when none applies.

    Reported as evidence only. Bare soil deliberately has no score: it is the cascade's
    remainder class, so there is no index whose magnitude means "more bare", and inventing one
    would attach a number to a region that measured nothing.
    """
    preference = {
        LandCoverClass.WATER: ("mndwi", "ndwi"),
        LandCoverClass.VEGETATION: ("ndvi",),
        LandCoverClass.BUILT_UP: ("ndbi",),
    }.get(target, ())
    if not preference:
        return None, None
    available = indices.compute_all_available(raster)
    for name in preference:
        if name in available:
            arr = available[name].array
            return np.clip((arr + 1.0) / 2.0, 0.0, 1.0).astype(np.float32), name
    return None, None


def _countability(
    target: LandCoverClass, fusion: FusionResult | None
) -> tuple[bool, str | None]:
    """Whether the number of regions of ``target`` can be stated, and why not when it cannot.

    See the module docstring, point 2. The rule follows the physics rather than the topology of
    a particular mask: no property of the returned regions distinguishes a countable result from
    a broken one. In the demo scene the largest region holds 100% of the bare-soil pixels when
    counting is perfect and 90% of the built-up pixels when counting is entirely broken, while
    water counts perfectly at 65% — so a "is one blob dominating?" heuristic would have gated
    the wrong cases in both directions.
    """
    if target not in RADAR_ARBITRATED_CLASSES:
        return True, None
    if fusion is not None and ScatteringRegime.DOUBLE_BOUNCE in fusion.sar.separated_regimes:
        return True, None

    shared = (
        "Built-up land and bare soil are both bright in the short-wave infrared, so the boundary "
        "between them is inferred from surface texture rather than measured directly, and "
        "neighbouring patches merge into one another wherever that texture cue is weak. The "
        "mapped extent and the total area are unaffected; only the number of separate regions is."
    )
    if fusion is None:
        return False, (
            "The number of separate areas cannot be stated reliably from this image alone. "
            + shared
            + " Radar measures that boundary directly, through the double-bounce return from "
            "vertical structure, so supplying a co-registered SAR image of the same area makes "
            "the count reportable."
        )
    return False, (
        "The number of separate areas cannot be stated reliably for this pair. The radar image "
        "contained no separable bright (double-bounce) population, so it could not draw the "
        "built-up boundary in this scene either. "
        + shared
    )


def _resolve_source(
    optical: RasterData,
    sar: RasterData | None,
    optical_result: LandCoverResult | None,
    fusion_result: FusionResult | None,
) -> tuple[np.ndarray, str, LandCoverResult, FusionResult | None, list[str]]:
    """Produce the class map to ground against, preferring the fused one when radar is present.

    Returns the class map, a source label, the optical classification, the fusion result if any,
    and the warnings the source itself raised — which are carried through rather than dropped,
    because a caveat about the classification is equally a caveat about the regions taken from it.
    """
    if fusion_result is not None:
        return (
            fusion_result.class_map,
            f"optical-sar-fusion ({fusion_result.method})",
            fusion_result.optical,
            fusion_result,
            list(fusion_result.warnings),
        )
    if sar is not None:
        fused = fuse_optical_sar(optical, sar, optical_result=optical_result)
        return (
            fused.class_map,
            f"optical-sar-fusion ({fused.method})",
            fused.optical,
            fused,
            list(fused.warnings),
        )
    optical_lc = optical_result if optical_result is not None else classify_land_cover(optical)
    return (
        optical_lc.class_map,
        f"optical-classification ({optical_lc.method})",
        optical_lc,
        None,
        list(optical_lc.warnings),
    )


def _check_geometry(query: GroundingQuery, meta: RasterMetadata) -> None:
    """Refuse the parts of a query the image's geometry cannot support.

    Raises:
        GeospatialError: A compass sector was requested on a grid that is not north-up, or a
            real-world area filter on an image with no known pixel size. Both would otherwise
            produce a confident answer about the wrong thing.
    """
    if (
        query.sector is not None
        and not query.frame_relative
        and query.sector.needs_north_up
        and not meta.is_north_up
    ):
        reason = (
            "is not georeferenced"
            if not meta.is_georeferenced
            else "is rotated relative to north"
        )
        raise GeospatialError(
            f'"{query.sector_phrase}" cannot be located because this image {reason}, so the '
            "top of the picture is not necessarily north. Asking for the "
            f'"{query.sector.frame_name}" of the image works on any picture.',
            context={"sector": query.sector.value, "north_up": meta.is_north_up},
        )
    if query.has_ground_unit_filter and not meta.is_georeferenced:
        raise GeospatialError(
            "A size limit in real-world units needs a georeferenced image, and this one carries "
            "no coordinate system, so its pixels have no known ground size. A limit in pixels "
            'works on any picture — for example "larger than 500 pixels".',
            context={"min_area_m2": query.min_area_m2, "max_area_m2": query.max_area_m2},
        )


def _apply_filters(
    vectors: VectorizeResult,
    query: GroundingQuery,
    shape: tuple[int, int],
) -> tuple[list[GroundedRegion], dict[str, int]]:
    """Apply the sector and size filters, counting what each one removed."""
    excluded = {
        "outside_sector": 0,
        "below_min_area": 0,
        "above_max_area": 0,
        "below_min_pixels": 0,
        "above_max_pixels": 0,
        "area_unknown": 0,
        "beyond_limit": 0,
    }
    sector = sector_mask(shape, query.sector) if query.sector is not None else None

    kept: list[GroundedRegion] = []
    for region in vectors.regions:
        overlap: float | None = None
        if sector is not None:
            pixels = vectors.pixels_of(region)
            overlap = float((pixels & sector).sum()) / float(max(pixels.sum(), 1))
            if overlap <= 0.5:
                excluded["outside_sector"] += 1
                continue
        if query.min_pixels is not None and region.pixel_area < query.min_pixels:
            excluded["below_min_pixels"] += 1
            continue
        if query.max_pixels is not None and region.pixel_area > query.max_pixels:
            excluded["above_max_pixels"] += 1
            continue
        if query.has_ground_unit_filter:
            if region.area_m2 is None:
                excluded["area_unknown"] += 1
                continue
            if query.min_area_m2 is not None and region.area_m2 < query.min_area_m2:
                excluded["below_min_area"] += 1
                continue
            if query.max_area_m2 is not None and region.area_m2 > query.max_area_m2:
                excluded["above_max_area"] += 1
                continue
        kept.append(GroundedRegion(region=region, sector_overlap=overlap))

    def size_key(g: GroundedRegion) -> float:
        return g.area_m2 if g.area_m2 is not None else float(g.pixel_area)

    kept.sort(key=size_key, reverse=query.superlative != "smallest")
    if query.limit is not None and len(kept) > query.limit:
        excluded["beyond_limit"] = len(kept) - query.limit
        kept = kept[: query.limit]
    return kept, excluded


def ground_query(
    query: str,
    optical: RasterData,
    *,
    sar: RasterData | None = None,
    optical_result: LandCoverResult | None = None,
    fusion_result: FusionResult | None = None,
) -> GroundingResult:
    """Locate the regions a text query names.

    Args:
        query: The user's question, in natural language.
        optical: The optical image to ground in.
        sar: An optional co-registered SAR image. When present the regions come from the fused
            class map, which is what makes the built-up and bare-soil counts reportable at all
            (module docstring, point 2).
        optical_result: A land-cover classification of ``optical``, if one has already been
            computed, to avoid classifying twice in an orchestrated run.
        fusion_result: A fusion of ``optical`` and a SAR image, if already computed. Takes
            precedence over ``sar``.

    Returns:
        A :class:`GroundingResult`. An understood query that matches nothing returns zero
        regions with a warning saying so — an empty answer is a finding, not an error.

    Raises:
        ValidationError: The query is empty or over the length limit.
        UnsupportedTaskError: The query cannot be parsed into a supported request.
        GeospatialError: The query needs geometry this image does not have.
    """
    parsed = parse_grounding_query(query)
    _check_geometry(parsed, optical.metadata)

    class_map, source, optical_lc, fusion, warnings = _resolve_source(
        optical, sar, optical_result, fusion_result
    )
    class_mask = class_map == parsed.target.code
    class_pixels = int(class_mask.sum())
    coverage = class_pixels / float(class_mask.size) if class_mask.size else 0.0

    score, score_name = _score_map(optical, parsed.target)
    min_pixels = pixels_for_ground_area(
        optical.metadata, MIN_REGION_AREA_M2, default_pixels=MIN_REGION_PIXELS
    )
    vectors = vectorize_mask(
        class_mask,
        optical.metadata,
        min_pixels=min_pixels,
        max_regions=MAX_REGIONS,
        score_map=score,
    )
    warnings.extend(vectors.warnings)

    regions, excluded = _apply_filters(vectors, parsed, class_mask.shape)

    selected = np.zeros_like(class_mask)
    for grounded in regions:
        selected |= vectors.pixels_of(grounded.region)
    areas = [g.area_m2 for g in regions if g.area_m2 is not None]
    selected_area = float(sum(areas)) if areas and vectors.georeferenced else None

    countable, count_caveat = _countability(parsed.target, fusion)
    if count_caveat is not None and (parsed.wants_count or regions):
        warnings.append(count_caveat)

    if class_pixels == 0:
        warnings.append(
            f"No {parsed.target.value.replace('_', ' ')} was detected anywhere in this image, "
            "so there is nothing to locate."
        )
    elif not regions:
        warnings.append(
            _no_match_reason(parsed, excluded, vectors.count, min_pixels, vectors.georeferenced)
        )

    result = GroundingResult(
        query=parsed,
        regions=regions,
        mask=selected,
        source=source,
        region_semantics=_region_semantics(parsed.target, optical.metadata),
        count_reportable=countable,
        count_caveat=count_caveat,
        selected_pixels=int(selected.sum()),
        selected_area_m2=selected_area,
        class_pixels=class_pixels,
        class_coverage_fraction=coverage,
        georeferenced=vectors.georeferenced,
        excluded=excluded,
        evidence=_evidence(
            parsed, optical_lc, fusion, vectors, score_name, min_pixels
        ),
        warnings=warnings,
    )
    logger.info(
        "grounding complete",
        extra={
            "target": parsed.target.value,
            "source": source,
            "regions_returned": result.count,
            "class_pixels": class_pixels,
            "count_reportable": countable,
        },
    )
    return result


def _no_match_reason(
    query: GroundingQuery,
    excluded: dict[str, int],
    candidates: int,
    min_pixels: int,
    georeferenced: bool,
) -> str:
    """Explain an empty result in terms of the filter that emptied it.

    "No regions found" is true but useless; whether a size limit removed forty candidates or
    the class simply has no patches above the noise floor changes what the user should do next.
    """
    if excluded.get("outside_sector"):
        where = (
            query.sector.frame_name
            if query.sector is not None and query.frame_relative
            else (query.sector_phrase or "that part of the image")
        )
        return (
            f"{candidates} separate areas of {query.target.value.replace('_', ' ')} were found "
            f"in this image, but none of them lies mainly in the {where}."
        )
    if excluded.get("below_min_area") or excluded.get("above_max_area"):
        return (
            f"{candidates} separate areas of {query.target.value.replace('_', ' ')} were found, "
            "but none of them falls inside the size limit in the question."
        )
    if excluded.get("below_min_pixels") or excluded.get("above_max_pixels"):
        return (
            f"{candidates} separate areas of {query.target.value.replace('_', ' ')} were found, "
            "but none of them matches the pixel-count limit in the question."
        )
    if excluded.get("area_unknown"):
        return (
            "A size limit in real-world units was requested, but the ground area of these "
            "regions could not be established, so none could be tested against it."
        )
    unit = f"{MIN_REGION_AREA_M2:.0f} m2" if georeferenced else f"{min_pixels} pixels"
    return (
        f"{query.target.value.replace('_', ' ').capitalize()} pixels are present, but no patch "
        f"of at least {unit} was found, so nothing is reported as a distinct area."
    )


def _region_semantics(target: LandCoverClass, meta: RasterMetadata) -> str:
    """State what one returned region is, at this image's resolution."""
    name = target.value.replace("_", " ")
    px = meta.pixel_size
    scale = ""
    if meta.is_georeferenced and px is not None:
        scale = f" mapped at about {px[0]:.0f} m per pixel"
    if target is LandCoverClass.BUILT_UP:
        return (
            f"Each region is one connected patch of built-up land{scale} — a block of "
            "development, not an individual building. Structures closer together than the pixel "
            "size cannot be separated."
        )
    return f"Each region is one connected patch of {name}{scale}."


def _evidence(
    query: GroundingQuery,
    optical_lc: LandCoverResult,
    fusion: FusionResult | None,
    vectors: VectorizeResult,
    score_name: str | None,
    min_pixels: int,
) -> dict[str, Any]:
    """The measurements behind the answer, for the confidence scorer and the execution trace."""
    out: dict[str, Any] = {
        "classification_method": optical_lc.method,
        "indices_used": list(optical_lc.indices_used),
        "class_separability": round(optical_lc.separability, 4),
        "region_score_index": score_name,
        "candidate_regions": vectors.count,
        "min_region_pixels": min_pixels,
        "min_region_area_m2": MIN_REGION_AREA_M2,
        "sector_selection": (
            None if query.sector is None else "majority overlap (more than half the region)"
        ),
    }
    if fusion is not None:
        out["fusion_method"] = fusion.method
        out["fusion_agreement_fraction"] = round(fusion.agreement_fraction, 4)
        out["fusion_corroborated_fraction"] = round(fusion.corroborated_fraction, 4)
        out["sar_regimes_separated"] = [r.value for r in fusion.sar.separated_regimes]
    return out

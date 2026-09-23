"""Core enumerations and value types shared across every layer.

These are deliberately dependency-free (stdlib only) so that the geospatial, service,
agent and API layers can all import them without creating cycles.
"""
from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    """Sensing modality of an input raster."""

    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


class InputMode(StrEnum):
    """Canonical input configuration of an analysis."""

    SINGLE_OPTICAL = "single_optical"
    SINGLE_SAR = "single_sar"
    OPTICAL_SAR_PAIR = "optical_sar_pair"
    BITEMPORAL_PAIR = "bitemporal_pair"


class QueryTask(StrEnum):
    """Tasks the router can dispatch to.

    ``UNKNOWN`` is a first-class outcome: an unrecognised query must produce an explicit
    "unsupported" response rather than a guessed workflow (brief §9, §28).
    """

    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    CHANGE_DETECTION = "change_detection"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_ANALYSIS = "optical_sar_analysis"
    LAND_COVER_ANALYSIS = "land_cover_analysis"
    REPORT_GENERATION = "report_generation"
    UNKNOWN = "unknown"


class ConfidenceLevel(StrEnum):
    """Bucketed confidence. Derived from a computed score, never from a language model."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"

    @classmethod
    def from_score(cls, score: float) -> ConfidenceLevel:
        """Map a 0..1 evidence score onto a level.

        The ``INSUFFICIENT`` band is what allows the system to decline to answer instead of
        guessing when evidence is too weak to support any claim.
        """
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.50:
            return cls.MEDIUM
        if score >= 0.25:
            return cls.LOW
        return cls.INSUFFICIENT


class LandCoverClass(StrEnum):
    """Land-cover classes the deterministic analysers can actually distinguish.

    Kept intentionally small: every member here must be separable by a real spectral or
    textural method that ships in this repo. Adding a class without an analyser would be a
    fabrication risk.
    """

    WATER = "water"
    VEGETATION = "vegetation"
    BUILT_UP = "built_up"
    BARE_SOIL = "bare_soil"
    UNCLASSIFIED = "unclassified"

    @property
    def code(self) -> int:
        """Integer this class takes in a class-map raster.

        One canonical mapping, because the same integers appear in four places that must not
        drift: the classifier's output raster, the demo generator's ground truth, the
        manifest's ``class_codes``, and any label raster a user uploads. ``0`` is reserved for
        "not classified" so an all-zero array reads as *nothing established* rather than as
        water.
        """
        return {
            LandCoverClass.UNCLASSIFIED: 0,
            LandCoverClass.WATER: 1,
            LandCoverClass.VEGETATION: 2,
            LandCoverClass.BUILT_UP: 3,
            LandCoverClass.BARE_SOIL: 4,
        }[self]

    @classmethod
    def from_code(cls, code: int) -> LandCoverClass:
        """Inverse of :attr:`code`; unrecognised integers become :attr:`UNCLASSIFIED`."""
        return {c.code: c for c in cls}.get(int(code), cls.UNCLASSIFIED)


class SpatialSector(StrEnum):
    """A named part of an image frame, for queries like "buildings in the north-east".

    Cardinal sectors are halves and diagonal ones quadrants, because that is what the words
    mean in ordinary use: the north of a scene is its upper half, not its upper third.
    :attr:`CENTRE` is the middle half of each axis.

    Deliberately coarse. A finer grid would imply a spatial precision the words do not carry,
    and a region straddling a boundary is the normal case rather than the exception — between
    25% and 100% of the ground-truth regions in the demo scene cross a quadrant line — so
    :mod:`app.services.grounding` selects by majority overlap and reports the overlap fraction
    instead of pretending each region sits inside exactly one cell.
    """

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTH_EAST = "north_east"
    NORTH_WEST = "north_west"
    SOUTH_EAST = "south_east"
    SOUTH_WEST = "south_west"
    CENTRE = "centre"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(row_start, row_end, col_start, col_end)`` as fractions of the frame in ``[0, 1]``.

        Rows increase downward, following raster convention, so :attr:`NORTH` is the *low* row
        range. That equivalence between "up" and "north" holds only for a north-up transform,
        which is why a caller must check the grid orientation before reading a compass name off
        these bounds (see :attr:`needs_north_up`).
        """
        return {
            SpatialSector.NORTH: (0.0, 0.5, 0.0, 1.0),
            SpatialSector.SOUTH: (0.5, 1.0, 0.0, 1.0),
            SpatialSector.EAST: (0.0, 1.0, 0.5, 1.0),
            SpatialSector.WEST: (0.0, 1.0, 0.0, 0.5),
            SpatialSector.NORTH_EAST: (0.0, 0.5, 0.5, 1.0),
            SpatialSector.NORTH_WEST: (0.0, 0.5, 0.0, 0.5),
            SpatialSector.SOUTH_EAST: (0.5, 1.0, 0.5, 1.0),
            SpatialSector.SOUTH_WEST: (0.5, 1.0, 0.0, 0.5),
            SpatialSector.CENTRE: (0.25, 0.75, 0.25, 0.75),
        }[self]

    @property
    def frame_name(self) -> str:
        """The image-space wording for the same area, e.g. "top left" for :attr:`NORTH_WEST`.

        Every sector is answerable in these terms on any image, georeferenced or not, so this
        is what a user-facing sentence falls back to when the compass reading is unavailable.
        """
        return {
            SpatialSector.NORTH: "top",
            SpatialSector.SOUTH: "bottom",
            SpatialSector.EAST: "right",
            SpatialSector.WEST: "left",
            SpatialSector.NORTH_EAST: "top right",
            SpatialSector.NORTH_WEST: "top left",
            SpatialSector.SOUTH_EAST: "bottom right",
            SpatialSector.SOUTH_WEST: "bottom left",
            SpatialSector.CENTRE: "centre",
        }[self]

    @property
    def needs_north_up(self) -> bool:
        """Whether reading this sector as a *compass* direction requires a north-up grid.

        ``False`` only for :attr:`CENTRE`: the middle of the pixel grid is the middle of the
        ground footprint under any rotation, so naming it involves no orientation assumption.
        For every other sector the claim "this is the north of the scene" is false on a rotated
        or ungeoreferenced image, and the frame-relative wording must be used instead.
        """
        return self is not SpatialSector.CENTRE


class ScatteringRegime(StrEnum):
    """What a SAR backscatter level says about the surface — a *mechanism*, not a land cover.

    Deliberately not :class:`LandCoverClass`. Radar measures surface roughness, geometry and
    dielectric constant, and several unrelated surfaces share one regime: a dark pixel is
    specular, which fits calm water but equally fits a dry tarmac apron, dune sand, or a
    radar shadow behind a hill. Naming these classes "water"/"built-up" would convert a
    physical measurement into a land-cover claim the single-channel data cannot support, so
    the regime is reported as measured and its candidate surfaces are listed, plural.
    """

    SMOOTH = "smooth"
    """Low backscatter: the surface reflects away from the sensor rather than back to it."""

    DIFFUSE = "diffuse"
    """Moderate backscatter from a rough surface. Vegetation and bare soil both live here and
    single-polarisation VV cannot separate them, so this regime is never subdivided."""

    DOUBLE_BOUNCE = "double_bounce"
    """High backscatter from a wall-and-ground corner geometry: buildings, ships, metal."""

    UNCLASSIFIED = "unclassified"
    """No regime could be established for this pixel — invalid data, or a refused threshold."""

    @property
    def candidate_surfaces(self) -> tuple[str, ...]:
        """Surfaces consistent with this regime, in likelihood order.

        Always more than one for the physical regimes. Callers must present these as
        candidates; picking one requires evidence radar alone does not provide, which is the
        motivation for optical/SAR fusion.
        """
        return {
            ScatteringRegime.SMOOTH: (
                "calm open water", "wet smooth ground", "dry sand or bare tarmac",
                "radar shadow",
            ),
            ScatteringRegime.DIFFUSE: ("vegetation", "bare soil", "rough natural terrain"),
            ScatteringRegime.DOUBLE_BOUNCE: (
                "buildings or other vertical structures", "metallic objects or vessels",
            ),
            ScatteringRegime.UNCLASSIFIED: (),
        }[self]

    @property
    def consistent_land_cover(self) -> frozenset[LandCoverClass]:
        """:attr:`candidate_surfaces` expressed in the classes this repo can measure.

        Each entry is one of the surfaces named above, mapped onto :class:`LandCoverClass`:
        "calm open water" and "wet smooth ground" are water; "dry sand or bare tarmac" and
        "rough natural terrain" are bare soil; "radar shadow" maps to nothing at all, because
        a shadow is an absence of measurement rather than a surface.

        Deliberately kept adjacent to the prose so the two cannot drift apart. Optical/SAR
        fusion decides whether two sensors corroborate or contradict each other from *this*
        set, so that the decision follows from the declared radar physics rather than from a
        separate hand-written compatibility table that nothing would keep honest.

        The sizes carry meaning of their own: only ``DOUBLE_BOUNCE`` maps to a single class, so
        it is the only regime that can name a land cover on its own. ``SMOOTH`` and
        ``DIFFUSE`` each admit two, which is why radar alone must decline there.
        """
        return {
            ScatteringRegime.SMOOTH: frozenset(
                {LandCoverClass.WATER, LandCoverClass.BARE_SOIL}
            ),
            ScatteringRegime.DIFFUSE: frozenset(
                {LandCoverClass.VEGETATION, LandCoverClass.BARE_SOIL}
            ),
            ScatteringRegime.DOUBLE_BOUNCE: frozenset({LandCoverClass.BUILT_UP}),
            ScatteringRegime.UNCLASSIFIED: frozenset(),
        }[self]

    @property
    def code(self) -> int:
        """Integer this regime takes in a regime-map raster; ``0`` means nothing established."""
        return {
            ScatteringRegime.UNCLASSIFIED: 0,
            ScatteringRegime.SMOOTH: 1,
            ScatteringRegime.DIFFUSE: 2,
            ScatteringRegime.DOUBLE_BOUNCE: 3,
        }[self]

    @classmethod
    def from_code(cls, code: int) -> ScatteringRegime:
        """Inverse of :attr:`code`; unrecognised integers become :attr:`UNCLASSIFIED`."""
        return {r.code: r for r in cls}.get(int(code), cls.UNCLASSIFIED)


class FusionEvidence(StrEnum):
    """Which sensors supported the class reported at a pixel by optical/SAR fusion.

    This is the per-pixel provenance of a fused map, and it exists so that a downstream
    confidence score and a user-facing explanation can both discount exactly the pixels that
    deserve it. A fused class map without it would present corroborated and arbitrated pixels
    as equally certain.
    """

    BOTH = "both"
    """Both sensors were established here and their readings are mutually consistent."""

    CONFLICT = "conflict"
    """Both were established and they disagree; an arbitration rule chose between them.

    Never silently averaged away — every conflict is reported with its counts, the rule that
    resolved it, and the candidate surfaces radar considered."""

    OPTICAL_ONLY = "optical_only"
    """Only the optical classifier reached a class; radar established no regime here."""

    SAR_ONLY = "sar_only"
    """Only radar established a regime. It names a class only when exactly one land cover is
    consistent with that regime (see :attr:`ScatteringRegime.consistent_land_cover`)."""

    NEITHER = "neither"
    """Neither sensor established anything — nodata on both sides, or both declined."""


class ChangeType(StrEnum):
    """What a land-cover transition means, in the terms a user asks questions in.

    Derived by lookup from an observed ``(from_class, to_class)`` pair, never inferred from
    the query. The mapping is keyed on the *destination* class first, which makes it total
    and unambiguous over the four real classes: becoming water is water gain, becoming
    vegetation is vegetation gain, becoming built-up is urban expansion, and becoming bare
    ground is a loss of whatever was there before. Every off-diagonal pair therefore has
    exactly one label, so no transition can be reported under two different names.
    """

    WATER_GAIN = "water_gain"
    WATER_LOSS = "water_loss"
    VEGETATION_GAIN = "vegetation_gain"
    VEGETATION_LOSS = "vegetation_loss"
    URBAN_EXPANSION = "urban_expansion"
    URBAN_LOSS = "urban_loss"
    SPECTRAL_ONLY = "spectral_only"
    """Measured spectral change with no land-cover class transition.

    A real and reportable observation — a crop growing, a reservoir dropping — kept separate
    so it is never presented as a land-cover conversion.
    """

    OTHER = "other"
    """A transition involving an unclassified pixel, so its meaning is not established."""

    @classmethod
    def from_transition(cls, before: LandCoverClass, after: LandCoverClass) -> ChangeType:
        """Label an observed class transition.

        Returns:
            :attr:`SPECTRAL_ONLY` when the class did not change, and :attr:`OTHER` when
            either side is unclassified — declining to name a change whose endpoints are
            not both established.
        """
        if before is after:
            return cls.SPECTRAL_ONLY
        if LandCoverClass.UNCLASSIFIED in (before, after):
            return cls.OTHER
        if after is LandCoverClass.WATER:
            return cls.WATER_GAIN
        if after is LandCoverClass.VEGETATION:
            return cls.VEGETATION_GAIN
        if after is LandCoverClass.BUILT_UP:
            return cls.URBAN_EXPANSION
        # after is BARE_SOIL: the informative half is what was lost.
        return {
            LandCoverClass.WATER: cls.WATER_LOSS,
            LandCoverClass.VEGETATION: cls.VEGETATION_LOSS,
            LandCoverClass.BUILT_UP: cls.URBAN_LOSS,
        }[before]


class StepStatus(StrEnum):
    """Outcome of a single orchestration step in the execution trace."""

    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class AnalysisStatus(StrEnum):
    """Terminal state of a whole analysis."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class EvidenceType(StrEnum):
    """What kind of thing a piece of evidence *is* (§15).

    A closed vocabulary rather than a free string, so the result page can render each kind
    correctly and a reader can tell a measured number from a prose observation. The distinction
    that matters most is :attr:`MEASUREMENT` versus :attr:`OBSERVATION`: the first is a figure the
    analysis computed, the second is a statement about the analysis. Only the first should ever be
    quoted as a result.
    """

    MEASUREMENT = "measurement"
    """A number the analysis computed (a class fraction, an area, a backscatter mean)."""

    CLASSIFICATION = "classification"
    """A per-pixel class assignment, summarised."""

    REGION = "region"
    """A located region, carrying a geometry."""

    CHANGE_MAP = "change_map"
    """A spatial map of detected change."""

    QUALITY = "quality"
    """An input-quality or metadata finding (nodata share, georeference presence, registration)."""

    OBSERVATION = "observation"
    """A plain-language observation about the analysis, with no single number attached."""


class BandRole(StrEnum):
    """Semantic role of a raster band.

    Resolved from band descriptions or band-count convention. ``UNKNOWN`` is preserved
    rather than guessed, because silently assuming a band is NIR would corrupt every index
    computed downstream.
    """

    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    NIR = "nir"
    SWIR1 = "swir1"
    SWIR2 = "swir2"
    VV = "vv"
    VH = "vh"
    GRAY = "gray"
    ALPHA = "alpha"
    UNKNOWN = "unknown"

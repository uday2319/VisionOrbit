"""Tests for the grounding tool — the query-driven, single-image capability (brief §2B, §18, §28).

Runs against the real demo optical scene (no mocks). It must (a) locate a countable class (water)
as regions with measured confidence; (b) map the *extent* of a class it cannot count from optical
alone (built-up), stating so instead of inventing a number, with the physical reason surfaced;
(c) refuse the inputs and queries it cannot honestly handle — a SAR image, the wrong image count, an
object it has no detector for ("ships"), a spatial relation ("near the river"), and an empty query.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agents.tools import GroundingTool, ToolContext
from app.agents.tools.base import ToolResult, ToolTier
from app.core.errors import ErrorCode, UnsupportedTaskError, ValidationError
from app.core.types import ConfidenceLevel, Modality, QueryTask, SpatialSector, StepStatus
from app.geospatial.raster import load_raster


@pytest.fixture(scope="module")
def optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def sar(demo_root):
    return load_raster(
        demo_root / "sar" / "scene_sar_vv.tif", max_edge=None, modality=Modality.SAR
    )


class TestLocateWater:
    """A countable class located across the whole image: regions, extent, and a reportable count."""

    @pytest.fixture(scope="class")
    def result(self, optical) -> ToolResult:
        return GroundingTool().run(ToolContext("where is the water?", [optical]))

    def test_identifies_itself_honestly(self, result):
        assert isinstance(result, ToolResult)
        assert result.task is QueryTask.GROUNDING
        assert result.tool_name == "grounding-spectral-cv"
        assert result.tier is ToolTier.CLASSICAL  # classifier cascade, not a grounding model

    def test_answer_names_regions_of_the_target(self, result):
        assert "water" in result.answer.lower()
        assert "region" in result.answer.lower()  # regions, not prose

    def test_water_is_countable_and_not_insufficient(self, result):
        assert result.data["count_reportable"] is True  # water needs no radar to count
        assert result.confidence.level is not ConfidenceLevel.INSUFFICIENT

    def test_status_is_consistent_with_its_inputs(self, result):
        if result.warnings or result.confidence.level is ConfidenceLevel.INSUFFICIENT:
            assert result.status is StepStatus.WARNING
        else:
            assert result.status is StepStatus.OK

    def test_evidence_discloses_provenance(self, result):
        joined = " ".join(result.evidence).lower()
        assert "separability" in joined  # classifier separability disclosed
        assert "candidate" in joined  # candidate-patch count disclosed

    def test_raw_present_but_not_serialised(self, result):
        assert result.raw is not None  # the GroundingResult, for rendering the region overlay
        payload = json.loads(json.dumps(result.to_dict()))
        assert "raw" not in payload
        assert "regions" in payload["data"]
        assert "count_reportable" in payload["data"]

    def test_describe(self):
        d = GroundingTool().describe()
        assert d["name"] == "grounding-spectral-cv"
        assert d["tier"] == "classical"
        assert d["task"] == QueryTask.GROUNDING.value
        assert d["summary"]


class TestBuiltUpCountNotReportableFromOpticalAlone:
    """Built-up land can be *mapped* from optical, but not *counted* — its boundary with bare soil
    is not spectrally measurable. The tool must decline the count, keep the extent, and surface the
    physical reason (radar) rather than emitting a confident but meaningless number (§18, §28).
    """

    @pytest.fixture(scope="class")
    def result(self, optical) -> ToolResult:
        return GroundingTool().run(ToolContext("how many buildings are there?", [optical]))

    def test_count_is_not_reportable(self, result):
        assert result.data["count_reportable"] is False

    def test_answer_declines_the_count_but_keeps_the_extent(self, result):
        lowered = result.answer.lower()
        assert "cannot be stated reliably" in lowered
        assert "mapped" in lowered  # the extent is still reported

    def test_reason_mentions_radar(self, result):
        joined = " ".join(result.warnings).lower()
        assert "radar" in joined or "sar" in joined  # the physical reason is surfaced


class TestScorelessTargetInSector:
    """Bare soil is the cascade's remainder class — it has no index score — and a compass sector
    exercises the sector-selection evidence line. Together these cover the two conditional evidence
    branches the whole-image water/built-up queries do not."""

    @pytest.fixture(scope="class")
    def result(self, optical) -> ToolResult:
        return GroundingTool().run(
            ToolContext("where is the bare soil in the north?", [optical])
        )

    def test_sector_selection_is_disclosed(self, result):
        joined = " ".join(result.evidence).lower()
        assert "sector selection" in joined  # a sector was applied and stated

    def test_scoreless_target_has_no_strength_line(self, result):
        # Bare soil has no index whose magnitude means "more bare", so no strength line is emitted
        # (rather than a fabricated score attached to a remainder class).
        joined = " ".join(result.evidence).lower()
        assert "region strength scored by" not in joined


class TestRejects:
    def test_sar_input_is_wrong_modality(self, sar):
        with pytest.raises(ValidationError) as ei:
            GroundingTool().run(ToolContext("where is the water?", [sar]))
        assert ei.value.code == ErrorCode.WRONG_MODALITY

    def test_no_image_is_rejected(self):
        with pytest.raises(ValidationError) as ei:
            GroundingTool().run(ToolContext("where is the water?", []))
        assert ei.value.code == ErrorCode.MISSING_SECOND_IMAGE

    def test_two_images_is_too_many(self, optical):
        with pytest.raises(ValidationError) as ei:
            GroundingTool().run(ToolContext("where is the water?", [optical, optical]))
        assert ei.value.code == ErrorCode.TOO_MANY_IMAGES

    def test_undetectable_object_is_unsupported(self, optical):
        """"Ships" has no detector; refuse by name rather than guess from land cover."""
        with pytest.raises(UnsupportedTaskError):
            GroundingTool().run(ToolContext("how many ships are in the harbour?", [optical]))

    def test_spatial_relation_is_unsupported(self, optical):
        """A relation between objects is not something this system computes."""
        with pytest.raises(UnsupportedTaskError):
            GroundingTool().run(ToolContext("show the buildings near the river", [optical]))

    def test_empty_query_is_rejected(self, optical):
        with pytest.raises(ValidationError) as ei:
            GroundingTool().run(ToolContext("   ", [optical]))
        assert ei.value.code == ErrorCode.EMPTY_QUERY


class TestSummaryPhrasing:
    """The phrasing branches of :meth:`GroundingTool._summarize` the demo scene does not exercise:
    a class absent from the image, an understood query that matches nothing, and the pixel-only
    extent / sector wording. Driven with duck-typed stubs so the branches are covered exactly."""

    @staticmethod
    def _query(value: str, sector=None, sector_phrase=None):
        return SimpleNamespace(
            target=SimpleNamespace(value=value), sector=sector, sector_phrase=sector_phrase
        )

    def test_absent_class_says_nothing_to_locate(self):
        stub = SimpleNamespace(query=self._query("water"), class_pixels=0)
        assert GroundingTool._summarize(stub).startswith("No water was detected")

    def test_present_but_no_matching_region(self):
        stub = SimpleNamespace(query=self._query("vegetation"), class_pixels=500, regions=[])
        sentence = GroundingTool._summarize(stub)
        assert "present in the image" in sentence
        assert "distinct region" in sentence

    def test_count_reportable_singular_with_area(self):
        stub = SimpleNamespace(
            query=self._query("water"),
            class_pixels=500,
            regions=[object()],
            selected_pixels=1000,
            selected_area_m2=2_000_000.0,
            count_reportable=True,
            count=1,
            region_semantics="Each region is one connected patch of water.",
        )
        sentence = GroundingTool._summarize(stub)
        assert "1 distinct water region," in sentence  # singular, no plural 's'
        assert "km²" in sentence  # georeferenced extent

    def test_count_not_reportable_pixel_extent_with_sector(self):
        stub = SimpleNamespace(
            query=self._query("built_up", sector=SpatialSector.NORTH, sector_phrase="north"),
            class_pixels=800,
            regions=[object(), object()],
            selected_pixels=800,
            selected_area_m2=None,  # ungeoreferenced -> pixel-only extent
            count_reportable=False,
            count=2,
            region_semantics="Each region is one connected patch of built-up land.",
        )
        sentence = GroundingTool._summarize(stub)
        assert "cannot be stated reliably" in sentence
        assert "in the north" in sentence  # sector wording echoed
        assert "px" in sentence and "km" not in sentence  # pixel-only extent, no invented area

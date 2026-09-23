"""Tests for the router: classifier verdict + input modalities -> a dispatch decision (§5, §9).

Five parts, mirroring what the router owns:

* **input defaults** — an ``UNKNOWN`` query is defaulted from the images (one optical -> land cover,
  one radar -> SAR, two of the same sensor family -> change, optical+SAR -> fusion), and declined for
  clarification when the images fit no analysis (none, radar beside an undetermined image, three or
  more);
* **recognised pass-through** — a task named by the text is dispatched as-is when the inputs suit it;
* **the lone-SAR override** — a general "what do you see / describe this" query whose only input is
  a single SAR image is re-pointed to SAR analysis, while grounding and land-cover queries are not;
* **the pair override** — a single-scene query asked of two images is re-pointed to that pair's
  analysis instead of dying at the contract gate on the image count;
* **the decision object** — source, reason, clarification flag and serialisation.

Inputs are real :class:`RasterData` built in memory (the router reads only ``.modality``), so the
tests use the same type the orchestrator passes rather than a stand-in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.agents.classifier import Classification, classify
from app.agents.modes import MAX_INPUTS
from app.agents.router import MAX_PAIR_INPUTS, RouteSource, route
from app.core.types import BandRole, Modality, QueryTask
from app.geospatial.raster import RasterData, RasterMetadata


# ---------------------------------------------------------------------------
# In-memory rasters of a chosen modality (only .modality is read by the router)
# ---------------------------------------------------------------------------
def _raster(modality: Modality) -> RasterData:
    meta = RasterMetadata(
        path=Path(f"{modality.value}.tif"),
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs_wkt=None,
        crs_epsg=None,
        transform=None,
        bounds=None,
        nodata=None,
    )
    return RasterData(
        data=np.zeros((1, 4, 4), dtype=np.float32),
        metadata=meta,
        band_roles=[BandRole.GRAY],
        modality=modality,
    )


def opt() -> RasterData:
    return _raster(Modality.OPTICAL)


def sar() -> RasterData:
    return _raster(Modality.SAR)


def unk() -> RasterData:
    return _raster(Modality.UNKNOWN)


# ---------------------------------------------------------------------------
# Input-aware defaults for an UNKNOWN query (§9)
# ---------------------------------------------------------------------------
class TestInputDefaults:
    def test_empty_query_one_optical_defaults_to_land_cover(self):
        d = route("", [opt()])
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.INPUT_DEFAULT
        assert d.is_resolved
        assert not d.needs_clarification

    def test_vague_query_one_optical_defaults_to_land_cover(self):
        # "hello there" carries no task signal -> UNKNOWN -> defaulted from the single optical image.
        d = route("hello there, nice picture", [opt()])
        assert d.classification.task is QueryTask.UNKNOWN  # the text alone recognised nothing
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.INPUT_DEFAULT

    def test_one_sar_defaults_to_sar_analysis(self):
        d = route("", [sar()])
        assert d.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert d.source is RouteSource.INPUT_DEFAULT

    def test_one_unknown_modality_is_treated_as_optical(self):
        d = route("", [unk()])
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.INPUT_DEFAULT

    def test_two_optical_defaults_to_change(self):
        d = route("", [opt(), opt()])
        assert d.task is QueryTask.CHANGE_DETECTION
        assert d.source is RouteSource.INPUT_DEFAULT

    def test_optical_and_unknown_defaults_to_change(self):
        # No radar present, so the pair is compared for change (an unknown side is treated optical).
        d = route("", [opt(), unk()])
        assert d.task is QueryTask.CHANGE_DETECTION

    def test_optical_and_sar_defaults_to_fusion(self):
        d = route("", [opt(), sar()])
        assert d.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert d.source is RouteSource.INPUT_DEFAULT

    def test_no_image_is_unresolved(self):
        d = route("", [])
        assert d.task is QueryTask.UNKNOWN
        assert d.source is RouteSource.UNRESOLVED
        assert d.needs_clarification
        assert not d.is_resolved

    def test_two_sar_images_default_to_change(self):
        # Two radar scenes of one area at two dates are a change comparison: the change tool
        # compares their shared backscatter band directly. It cannot name a land-cover transition
        # from backscatter, and the reason says so rather than promising an optical-grade result.
        d = route("", [sar(), sar()])
        assert d.task is QueryTask.CHANGE_DETECTION
        assert d.source is RouteSource.INPUT_DEFAULT
        assert d.is_resolved
        assert not d.needs_clarification
        assert "amplitude comparison" in d.reason
        assert "without naming a land-cover transition" in d.reason

    def test_sar_beside_unknown_is_unresolved(self):
        d = route("", [sar(), unk()])
        assert d.source is RouteSource.UNRESOLVED
        assert "undetermined type" in d.reason
        # The remedy is actionable: the sensor can be declared at upload time.
        assert "declare the second image's sensor" in d.reason

    def test_three_images_are_unresolved(self):
        d = route("", [opt(), opt(), opt()])
        assert d.source is RouteSource.UNRESOLVED
        assert d.needs_clarification
        assert "more than two" in d.reason


# ---------------------------------------------------------------------------
# Recognised tasks pass through; the text wins over the input default
# ---------------------------------------------------------------------------
class TestRecognisedPassThrough:
    def test_land_cover_query_passes_through(self):
        d = route("classify the land cover", [opt()])
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.CLASSIFIER
        assert "matched:" in d.reason  # the reason cites the evidence behind the task

    def test_recognised_task_beats_the_input_default_for_one_image(self):
        # One optical image would default to land cover anyway; what this pins is that the *text*
        # decided it (source=classifier), not the image count.
        d = route("classify the land cover", [opt()])
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.CLASSIFIER

    def test_single_scene_task_on_a_pair_is_not_dispatched_as_single_scene(self):
        # A land-cover query over two images used to pass straight through to the single-image
        # analyser and die at the contract gate with "works on exactly 1 image", which named the
        # wrong problem: the question is answerable, just by the pair analysis. See
        # TestPairInputOverride for what it is re-pointed to.
        d = route("classify the land cover", [opt(), opt()])
        assert d.task is not QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.INPUT_OVERRIDE

    def test_change_question_routes_to_change_vqa(self):
        d = route("what changed between the two dates?", [opt(), opt()])
        assert d.task is QueryTask.CHANGE_VQA
        assert d.source is RouteSource.CLASSIFIER

    def test_change_command_routes_to_change_detection(self):
        d = route("show me the change", [opt(), opt()])
        assert d.task is QueryTask.CHANGE_DETECTION
        assert d.source is RouteSource.CLASSIFIER

    def test_sar_query_on_sar_input_is_classifier_not_override(self):
        # The text already names SAR analysis, so this is a plain recognition, not an input override.
        d = route("analyse the backscatter", [sar()])
        assert d.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert d.source is RouteSource.CLASSIFIER

    def test_unimplemented_task_passes_through_for_the_registry_to_refuse(self):
        # The router does not refuse object detection itself; it routes it and the registry declines.
        d = route("detect all the ships and count them", [opt()])
        assert d.task is QueryTask.OBJECT_DETECTION
        assert d.source is RouteSource.CLASSIFIER
        assert d.is_resolved  # "resolved" means routed, not that a tool exists

    def test_report_generation_passes_through(self):
        d = route("generate a full analysis report", [opt()])
        assert d.task is QueryTask.REPORT_GENERATION
        assert d.source is RouteSource.CLASSIFIER


# ---------------------------------------------------------------------------
# The single input override: general description + lone SAR image -> SAR analysis
# ---------------------------------------------------------------------------
class TestInputOverride:
    def test_vqa_on_a_lone_sar_image_is_re_pointed_to_sar(self):
        d = route("what do you see in this image", [sar()])
        assert d.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert d.source is RouteSource.INPUT_OVERRIDE
        assert "SAR" in d.reason and "radar" in d.reason.lower()

    def test_captioning_on_a_lone_sar_image_is_re_pointed_to_sar(self):
        d = route("describe this image", [sar()])
        assert d.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert d.source is RouteSource.INPUT_OVERRIDE

    def test_grounding_on_sar_is_not_overridden(self):
        # A specific optical product is not silently swapped; it passes through to an honest refusal.
        d = route("where is the water", [sar()])
        assert d.task is QueryTask.GROUNDING
        assert d.source is RouteSource.CLASSIFIER

    def test_land_cover_on_sar_is_not_overridden(self):
        d = route("classify the land cover", [sar()])
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.CLASSIFIER

    def test_vqa_on_optical_is_not_overridden(self):
        d = route("what do you see in this image", [opt()])
        assert d.task is QueryTask.VQA
        assert d.source is RouteSource.CLASSIFIER

    def test_lone_sar_override_does_not_apply_to_a_pair(self):
        # The single-image SAR override is for one radar image. Two of them are a pair, so the
        # pair override applies instead and the question is answered from both dates.
        d = route("what do you see in this image", [sar(), sar()])
        assert d.task is QueryTask.CHANGE_DETECTION
        assert d.source is RouteSource.INPUT_OVERRIDE


# ---------------------------------------------------------------------------
# The pair input override: a single-scene query asked of two images
# ---------------------------------------------------------------------------
class TestPairInputOverride:
    def test_land_cover_on_an_optical_sar_pair_routes_to_fusion(self):
        # The failure the user hit: "Combine the optical and radar images to map land cover"
        # classifies as land cover, which analyses one scene, and was rejected for the image count.
        d = route("map the land cover using both sensors", [opt(), sar()])
        assert d.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert d.source is RouteSource.INPUT_OVERRIDE
        assert "two images were provided" in d.reason
        assert "where the two sensors disagree" in d.reason

    def test_land_cover_on_two_optical_routes_to_change(self):
        d = route("classify the land cover", [opt(), opt()])
        assert d.task is QueryTask.CHANGE_DETECTION
        assert d.source is RouteSource.INPUT_OVERRIDE
        assert "at each date" in d.reason

    def test_grounding_on_a_pair_routes_to_the_pair_analysis(self):
        # Grounding *is* overridden on a pair, unlike on a lone SAR image: both pair analyses read
        # the scene, so the "where is X" question is answered from both inputs.
        d = route("where is the water", [opt(), sar()])
        assert d.task is QueryTask.OPTICAL_SAR_ANALYSIS
        assert d.source is RouteSource.INPUT_OVERRIDE

    def test_captioning_on_a_bitemporal_pair_routes_to_change(self):
        d = route("describe this image", [opt(), opt()])
        assert d.task is QueryTask.CHANGE_DETECTION
        assert d.source is RouteSource.INPUT_OVERRIDE

    def test_reason_names_the_analysis_that_ran(self):
        # The substitution is stated, not implied (§49): the reason names both the query's reading
        # and the analysis actually dispatched.
        d = route("classify the land cover", [opt(), sar()])
        assert "land-cover analysis" in d.reason
        assert "optical/SAR fusion" in d.reason

    def test_object_detection_on_a_pair_is_not_overridden(self):
        # No pair analysis subsumes detection, so it passes through to the registry's honest
        # refusal rather than being swapped for something the user did not ask for.
        d = route("detect all the ships", [opt(), sar()])
        assert d.task is QueryTask.OBJECT_DETECTION
        assert d.source is RouteSource.CLASSIFIER

    def test_report_generation_on_a_pair_is_not_overridden(self):
        d = route("generate a full analysis report", [opt(), opt()])
        assert d.task is QueryTask.REPORT_GENERATION
        assert d.source is RouteSource.CLASSIFIER

    def test_change_query_on_a_pair_is_not_overridden(self):
        # Already a pair task; there is nothing to re-point.
        d = route("what changed between the two dates?", [opt(), opt()])
        assert d.task is QueryTask.CHANGE_VQA
        assert d.source is RouteSource.CLASSIFIER

    def test_sar_beside_unknown_is_not_overridden(self):
        # Not a resolvable pair: no recognised optical scene for fusion, and no established
        # same-sensor pair for change. The recognised task passes through to a downstream refusal.
        d = route("classify the land cover", [sar(), unk()])
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.CLASSIFIER

    def test_override_does_not_apply_to_three_images(self):
        d = route("classify the land cover", [opt(), opt(), opt()])
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.CLASSIFIER

    def test_pair_override_matches_the_unknown_query_default(self):
        # The two paths into a pair analysis must not drift apart: whatever an empty query defaults
        # to for a given pair, a scene-reading query on that same pair is re-pointed to.
        for pair in ([opt(), opt()], [opt(), sar()], [sar(), sar()]):
            assert route("classify the land cover", pair).task == route("", pair).task


# ---------------------------------------------------------------------------
# The decision object
# ---------------------------------------------------------------------------
class TestRouteDecision:
    def test_injected_classification_is_used_without_reclassifying(self):
        # A pre-computed UNKNOWN classification + one optical image still defaults to land cover,
        # proving route() honours the injected verdict rather than re-reading the (here mismatched) text.
        injected = Classification(QueryTask.UNKNOWN, 0.0)
        d = route("classify the land cover", [opt()], classification=injected)
        assert d.classification is injected
        assert d.task is QueryTask.LAND_COVER_ANALYSIS
        assert d.source is RouteSource.INPUT_DEFAULT

    def test_to_dict_carries_the_decision_and_the_classification(self):
        d = route("classify the land cover", [opt()])
        payload = d.to_dict()
        assert payload["task"] == QueryTask.LAND_COVER_ANALYSIS.value
        assert payload["source"] == RouteSource.CLASSIFIER.value
        assert payload["needs_clarification"] is False
        assert isinstance(payload["reason"], str) and payload["reason"]
        # The raw text verdict is nested for the trace.
        assert payload["classification"]["task"] == QueryTask.LAND_COVER_ANALYSIS.value
        assert payload["classification"]["recognized"] is True

    def test_routing_is_deterministic(self):
        first = route("what do you see", [sar()])
        second = route("what do you see", [sar()])
        assert first == second

    def test_default_classification_matches_classify(self):
        # With no injected classification, route() runs the classifier itself.
        query = "detect all the ships"
        d = route(query, [opt()])
        assert d.classification == classify(query)

    def test_pair_size_constant_matches_the_mode_normaliser(self):
        # The router keeps its own literal to stay clear of the geospatial import chain that
        # app.agents.modes pulls in; modes.MAX_INPUTS is the authority, so they must agree.
        assert MAX_PAIR_INPUTS == MAX_INPUTS

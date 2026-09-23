"""Tests for the evidence-based confidence engine (brief §27).

Confidence must be **computed from measurements**, never guessed and never produced by a language
model. These tests hold two things: that the aggregator's mechanics are what the module claims
(gates cap, contributors average, reasons come out weakest-first), and — the headline requirement
of §7.1 case 2 — that a pure-noise change pair resolves to ``INSUFFICIENT`` because the engine
consumes both ``bimodality`` and ``corroborated_fraction``, not a low-but-nonzero answer that would
present sensor noise as a finding.

Where a claim is about a real analysis, the input is a real analysis: the demo scenes are
classified/fused/grounded at test time and the resulting measured evidence is fed to the engine, so
a regression in an upstream detector shows up here too.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from rasterio.transform import Affine

from app.core.types import ConfidenceLevel
from app.geospatial.raster import load_raster
from app.geospatial.validate import assess_quality
from app.services import confidence, indices
from app.services.change import detect_change
from app.services.confidence import (
    ConfidenceFactor,
    ConfidenceReport,
    FactorKind,
    aggregate,
)
from app.services.fusion import fuse_optical_sar
from app.services.grounding import ground_query
from app.services.landcover import classify_land_cover
from app.services.sar import analyze_sar
from tests.conftest import (
    TEST_ORIGIN_X,
    TEST_ORIGIN_Y,
    TEST_PIXEL_M,
    write_test_raster,
)

OPTICAL_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
KNOWN_TRANSFORM = Affine(TEST_PIXEL_M, 0.0, TEST_ORIGIN_X, 0.0, -TEST_PIXEL_M, TEST_ORIGIN_Y)


def _f(name: str, value: float, weight: float = 1.0, *, gate: bool = False) -> ConfidenceFactor:
    """Terse factor builder for the pure-aggregator tests."""
    return ConfidenceFactor(
        name=name,
        value=value,
        weight=weight,
        reason=f"{name}={value}",
        kind=FactorKind.GATE if gate else FactorKind.CONTRIBUTOR,
    )


# ---------------------------------------------------------------------------
# The aggregator: pure mechanics, no I/O
# ---------------------------------------------------------------------------
class TestAggregate:
    def test_no_factors_declines_rather_than_asserting_confidence(self):
        """An empty evidence set is INSUFFICIENT, not a default HIGH — silence is not proof."""
        report = aggregate([])
        assert report.level is ConfidenceLevel.INSUFFICIENT
        assert report.score == 0.0
        assert report.limiting_factor is None
        assert any("no evidence" in r.lower() for r in report.reasons)

    def test_a_low_gate_caps_the_score_below_a_high_mean(self):
        """The reason gates exist: a single necessary measurement near zero sinks the answer.

        Every contributor is healthy, so a plain weighted mean would be comfortably HIGH. The
        gate at 0.1 forces the score down to 0.1 regardless.
        """
        report = aggregate(
            [_f("a", 0.9), _f("b", 0.9), _f("evidence", 0.1, gate=True)]
        )
        assert report.score == pytest.approx(0.1)
        assert report.level is ConfidenceLevel.INSUFFICIENT
        assert report.limiting_factor == "evidence"

    def test_a_healthy_gate_does_not_cap_and_weakest_contributor_leads(self):
        """When no gate binds, the limiting factor is the weakest contributor, not the gate."""
        report = aggregate(
            [_f("strong", 0.9), _f("weak", 0.4), _f("gate", 0.95, gate=True)]
        )
        # gate (0.95) exceeds the mean, so it does not cap; the weakest contributor is named.
        assert report.score < 0.95
        assert report.limiting_factor == "weak"

    def test_a_single_gate_factor_sets_the_base_mean(self):
        """A gate still participates in the mean; it is not cap-only."""
        report = aggregate([_f("only", 0.6, gate=True)])
        assert report.score == pytest.approx(0.6)

    def test_values_are_clamped_into_the_unit_interval(self):
        """A factor cannot smuggle a score above 1 or below 0 into the aggregate."""
        report = aggregate([_f("over", 1.5), _f("under", -0.3)])
        assert 0.0 <= report.score <= 1.0
        assert report.score == pytest.approx(0.5)  # clamped to (1.0 + 0.0) / 2

    def test_reasons_are_ordered_weakest_measurement_first(self):
        report = aggregate([_f("high", 0.9), _f("low", 0.2), _f("mid", 0.5)])
        assert report.reasons == ["low=0.2", "mid=0.5", "high=0.9"]

    def test_zero_total_weight_falls_back_to_a_plain_mean(self):
        """Weightless factors are still averaged, so a report is never silently empty."""
        report = aggregate([_f("a", 0.2, weight=0.0), _f("b", 0.8, weight=0.0)])
        assert report.score == pytest.approx(0.5)

    def test_report_is_json_serialisable_without_numpy_scalars(self):
        report = aggregate([_f("a", 0.9), _f("evidence", 0.3, gate=True)])
        payload = json.loads(json.dumps(report.to_dict()))  # raises on any numpy scalar
        assert payload["level"] == report.level.value
        assert payload["sufficient"] is (report.level is not ConfidenceLevel.INSUFFICIENT)
        assert payload["limiting_factor"] == "evidence"
        assert {f["name"] for f in payload["factors"]} == {"a", "evidence"}
        assert any(f["kind"] == "gate" for f in payload["factors"])


class TestBimodalitySignal:
    """The mapping from Sarle's coefficient onto a confidence, anchored to known values."""

    @pytest.mark.parametrize(
        ("bimodality", "expected"),
        [
            (1.0 / 3.0, 0.0),  # a normal distribution: one population
            (indices.BIMODALITY_THRESHOLD, 1.0),  # uniform: the two-population boundary
            (0.20, 0.0),  # below normal, clamped
            (0.90, 1.0),  # well past uniform, clamped
        ],
    )
    def test_anchors(self, bimodality, expected):
        assert confidence._bimodality_signal(bimodality) == pytest.approx(expected)

    def test_is_monotonic_between_the_anchors(self):
        lo = confidence._bimodality_signal(0.40)
        hi = confidence._bimodality_signal(0.50)
        assert 0.0 < lo < hi < 1.0


class TestSharedFactors:
    def test_unassessed_input_quality_yields_no_factor(self):
        """Quality is optional; when it was not measured it is absent, not assumed."""
        assert confidence._input_quality_factor(None) is None

    def test_warnings_penalise_but_are_floored(self):
        """Warnings nudge confidence down without ever dominating a genuine measurement."""
        assert confidence._warning_factor([]).value == pytest.approx(1.0)
        many = confidence._warning_factor([f"w{i}" for i in range(20)])
        assert many.value == pytest.approx(0.4)  # floored, not driven to zero
        assert many.kind is FactorKind.CONTRIBUTOR


# ---------------------------------------------------------------------------
# Change — the §7.1 case 2 requirement
# ---------------------------------------------------------------------------
class TestChangeConfidence:
    @pytest.fixture(scope="class")
    def noise_change(self, tmp_path_factory):
        """A pure-noise pair: no real change, only independent sensor noise on each date."""
        rng = np.random.default_rng(11)
        tmp = tmp_path_factory.mktemp("conf_noise")
        pair = []
        for name in ("n1.tif", "n2.tif"):
            arr = rng.normal(3000, 200, (6, 96, 96)).clip(1, 10000).astype(np.uint16)
            pair.append(
                load_raster(
                    write_test_raster(
                        tmp / name, arr, transform=KNOWN_TRANSFORM, descriptions=OPTICAL_BANDS
                    ),
                    max_edge=None,
                )
            )
        return detect_change(*pair)

    @pytest.fixture(scope="class")
    def real_change(self, demo_root):
        before = load_raster(demo_root / "temporal" / "scene_t1_optical.tif", max_edge=None)
        after = load_raster(demo_root / "temporal" / "scene_t2_optical.tif", max_edge=None)
        return detect_change(before, after)

    def test_pure_noise_resolves_to_insufficient(self, noise_change):
        """The headline: noise classified twice must not become a low-confidence finding.

        A weighted mean over the (perfectly clean) input image would report a comfortable score.
        The evidence gate — bound by the 0.00 corroborated fraction — forces INSUFFICIENT.
        """
        report = confidence.for_change(noise_change)
        assert report.level is ConfidenceLevel.INSUFFICIENT
        assert report.score == pytest.approx(0.0, abs=1e-9)
        assert report.limiting_factor == "evidence_strength"

    def test_the_evidence_reason_states_both_quantities_it_consumed(self, noise_change):
        """§7.1 case 2 names both channels; the reason must show both were read, not one."""
        report = confidence.for_change(noise_change)
        evidence = next(f for f in report.factors if f.name == "evidence_strength")
        assert evidence.kind is FactorKind.GATE
        assert "bimodality" in evidence.reason.lower()
        assert "corroborated" in evidence.reason.lower()

    def test_real_change_is_not_insufficient(self, real_change):
        """The engine must also be able to say the evidence is strong, or it is just a constant."""
        report = confidence.for_change(real_change)
        assert report.level is not ConfidenceLevel.INSUFFICIENT

    def test_corroboration_alone_can_sink_the_gate(self, real_change):
        """Both channels are necessary: zero corroboration sinks an otherwise-bimodal result."""
        no_corr = replace(real_change, corroborated_pixels=0)
        assert no_corr.corroborated_fraction == 0.0
        assert confidence.for_change(no_corr).level is ConfidenceLevel.INSUFFICIENT

    def test_unimodal_magnitude_alone_can_sink_the_gate(self, real_change):
        """And a unimodal magnitude sinks an otherwise well-corroborated result."""
        assert real_change.corroborated_fraction > 0.5  # the corroboration is genuinely there
        flat = replace(real_change, bimodality=0.20)  # below the normal-distribution anchor
        assert confidence.for_change(flat).level is ConfidenceLevel.INSUFFICIENT

    def test_poor_registration_is_a_second_gate(self, real_change):
        """A misregistered pair reports boundaries as change; registration must cap, not average."""
        bad_reg = replace(real_change.registration, score=0.05, ncc=0.05)
        report = confidence.for_change(replace(real_change, registration=bad_reg))
        assert report.score <= 0.05 + 1e-9
        assert report.limiting_factor == "registration"


# ---------------------------------------------------------------------------
# Land cover
# ---------------------------------------------------------------------------
class TestLandcoverConfidence:
    @pytest.fixture(scope="class")
    def demo_lc(self, demo_root):
        opt = load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        return classify_land_cover(opt), assess_quality(opt)

    def test_demo_scene_is_confident(self, demo_lc):
        result, quality = demo_lc
        report = confidence.for_landcover(result, quality)
        assert result.separability > 0.75
        assert report.level is ConfidenceLevel.HIGH

    def test_low_separability_gates_the_map_down(self, demo_lc):
        """Separability is the gate: indistinct classes cannot be certified by a clean image."""
        result, quality = demo_lc
        indistinct = replace(result, separability=0.10)
        report = confidence.for_landcover(indistinct, quality)
        assert report.score <= 0.10 + 1e-9
        assert report.limiting_factor == "class_separability"


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------
class TestSarConfidence:
    @pytest.fixture(scope="class")
    def demo_sar(self, demo_root):
        sar = load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)
        return analyze_sar(sar), assess_quality(sar)

    def test_demo_sar_is_confident(self, demo_sar):
        result, quality = demo_sar
        assert len(result.separated_regimes) >= 2
        assert confidence.for_sar(result, quality).level is ConfidenceLevel.HIGH

    def test_an_all_diffuse_scene_is_not_punished_to_insufficient(self, demo_sar):
        """Low mode prominence means "no extra regime to split", not "the diffuse reading is bad".

        Separation is therefore a contributor, not a gate: a genuinely uniform scene keeps a
        usable confidence instead of being forced to INSUFFICIENT for being uniform.
        """
        result, quality = demo_sar
        all_diffuse = replace(result, smooth_threshold_db=None, bright_threshold_db=None)
        assert [r.value for r in all_diffuse.separated_regimes] == ["diffuse"]
        report = confidence.for_sar(all_diffuse, quality)
        separation = next(f for f in report.factors if f.name == "regime_separation")
        assert separation.kind is FactorKind.CONTRIBUTOR
        assert report.level is not ConfidenceLevel.INSUFFICIENT


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
class TestFusionConfidence:
    @pytest.fixture(scope="class")
    def demo_fusion(self, demo_root):
        opt = load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        sar = load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)
        return fuse_optical_sar(opt, sar), assess_quality(opt)

    def test_demo_fusion_is_confident(self, demo_fusion):
        result, quality = demo_fusion
        assert result.corroborated_fraction > 0.75
        assert confidence.for_fusion(result, quality).level is ConfidenceLevel.HIGH

    def test_corroboration_is_the_gate(self, demo_fusion):
        """The share both sensors supported is the necessary measurement, carried as a gate."""
        result, quality = demo_fusion
        report = confidence.for_fusion(result, quality)
        corr = next(f for f in report.factors if f.name == "corroboration")
        assert corr.kind is FactorKind.GATE
        assert corr.value == pytest.approx(result.corroborated_fraction)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------
class TestGroundingConfidence:
    @pytest.fixture(scope="class")
    def demo_optical(self, demo_root):
        return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)

    def test_grounded_water_is_confident(self, demo_optical):
        result = ground_query("where is the water", demo_optical)
        report = confidence.for_grounding(result, assess_quality(demo_optical))
        assert result.count_reportable
        assert report.level is ConfidenceLevel.HIGH

    def test_class_separability_is_the_gate(self, demo_optical):
        result = ground_query("where is the water", demo_optical)
        report = confidence.for_grounding(result, assess_quality(demo_optical))
        gate = next(f for f in report.factors if f.name == "class_separability")
        assert gate.kind is FactorKind.GATE
        assert gate.value == pytest.approx(result.evidence["class_separability"])

    def test_a_withheld_count_does_not_lower_extent_confidence(self, demo_optical):
        """Withholding an unreliable count is correct handling, not a defect in the extent.

        The buildings count is not reportable from optical alone (§4.8), but the extent is a real
        answer; its confidence must not be dragged down by the withheld count. The withheld count
        is surfaced as its own weightless reason instead.
        """
        result = ground_query("show me the buildings", demo_optical)
        assert not result.count_reportable
        report = confidence.for_grounding(result, assess_quality(demo_optical))
        assert report.level is ConfidenceLevel.HIGH
        note = next(f for f in report.factors if f.name == "count_reportable")
        assert note.weight == 0.0  # present as a reason, absent from the arithmetic
        assert any("count is withheld" in r for r in report.reasons)

    def test_missing_separability_evidence_defaults_to_a_zero_gate(self, demo_optical):
        """If the evidence block lacks a separability, the gate reads 0 and declines to over-claim."""
        result = ground_query("where is the water", demo_optical)
        stripped = replace(result, evidence={})
        report = confidence.for_grounding(stripped)
        assert report.level is ConfidenceLevel.INSUFFICIENT


class TestReportContract:
    """Cross-cutting guarantees every adapter's output must satisfy."""

    def test_every_adapter_returns_a_serialisable_report(self, demo_root):
        opt = load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        report = confidence.for_landcover(classify_land_cover(opt), assess_quality(opt))
        assert isinstance(report, ConfidenceReport)
        payload = json.loads(json.dumps(report.to_dict()))
        assert set(payload) == {
            "level", "score", "sufficient", "limiting_factor", "reasons", "factors"
        }
        assert 0.0 <= payload["score"] <= 1.0

    def test_input_quality_is_included_only_when_it_was_measured(self, demo_root):
        """Quality is an optional contributor: present when assessed, absent (not assumed) when not.

        Exercised across every adapter both ways, so no adapter silently drops or fabricates the
        input-quality factor.
        """
        opt = load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)
        sar = load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)
        before = load_raster(demo_root / "temporal" / "scene_t1_optical.tif", max_edge=None)
        after = load_raster(demo_root / "temporal" / "scene_t2_optical.tif", max_edge=None)
        q = assess_quality(opt)

        pairs = [
            (
                confidence.for_change(detect_change(before, after), q),
                confidence.for_change(detect_change(before, after)),
            ),
            (
                confidence.for_landcover(classify_land_cover(opt), q),
                confidence.for_landcover(classify_land_cover(opt)),
            ),
            (
                confidence.for_sar(analyze_sar(sar), assess_quality(sar)),
                confidence.for_sar(analyze_sar(sar)),
            ),
            (
                confidence.for_fusion(fuse_optical_sar(opt, sar), q),
                confidence.for_fusion(fuse_optical_sar(opt, sar)),
            ),
            (
                confidence.for_grounding(ground_query("where is the water", opt), q),
                confidence.for_grounding(ground_query("where is the water", opt)),
            ),
        ]
        for with_quality, without_quality in pairs:
            assert any(f.name == "input_quality" for f in with_quality.factors)
            assert not any(f.name == "input_quality" for f in without_quality.factors)

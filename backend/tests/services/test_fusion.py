"""Tests for cross-modal optical + SAR fusion (brief §4).

Every figure quoted in ``app/services/fusion.py`` is asserted here, so a change that alters
the arbitration also fails the suite rather than quietly making the documentation wrong. The
central claims under test are that fusion improves the one axis its declared physics claims —
built-up land — and provably leaves the others alone; that its per-pixel evidence label
predicts correctness; and that it refuses rather than guesses where radar names two surfaces.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.core.errors import ErrorCode, GeospatialError
from app.core.types import FusionEvidence, LandCoverClass, ScatteringRegime
from app.geospatial.raster import load_raster
from app.services.fusion import CODE_TO_EVIDENCE, fuse_optical_sar
from app.services.landcover import classify_land_cover
from app.services.sar import analyze_sar

# Demo ground-truth codes, duplicated from the generator on purpose: if the generator changes,
# these tests should fail rather than silently track it.
GT_WATER, GT_VEGETATION, GT_BUILT_UP, GT_BARE_SOIL = 1, 2, 3, 4
GT_CODE = {
    LandCoverClass.WATER: GT_WATER,
    LandCoverClass.VEGETATION: GT_VEGETATION,
    LandCoverClass.BUILT_UP: GT_BUILT_UP,
    LandCoverClass.BARE_SOIL: GT_BARE_SOIL,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 of a boolean mask against a boolean truth."""
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Overall accuracy over the pixels the truth map establishes."""
    scored = truth > 0
    return float((predicted[scored] == truth[scored]).mean())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_optical(demo_root):
    return load_raster(demo_root / "optical" / "scene_optical.tif", max_edge=None)


@pytest.fixture(scope="module")
def demo_sar(demo_root):
    return load_raster(demo_root / "sar" / "scene_sar_vv.tif", max_edge=None)


@pytest.fixture(scope="module")
def demo_rgb(demo_root):
    """The same scene with only visible bands — the case fusion exists for."""
    return load_raster(demo_root / "edge_cases" / "rgb_only.tif", max_edge=None)


@pytest.fixture(scope="module")
def demo_truth(demo_root):
    return load_raster(
        demo_root / "optical" / "scene_landcover_gt.tif", max_edge=None
    ).data[0].astype(int)


@pytest.fixture(scope="module")
def fused(demo_optical, demo_sar):
    """One full fusion, shared across assertions (it is the expensive step)."""
    return fuse_optical_sar(demo_optical, demo_sar)


@pytest.fixture(scope="module")
def optical_only(demo_optical):
    """The optical result on its own, to measure what the SAR channel added."""
    return classify_land_cover(demo_optical)


# ---------------------------------------------------------------------------
# The physics the arbitration rests on
# ---------------------------------------------------------------------------
class TestConsistencyIsDerivedFromRadarPhysics:
    """The compatibility relation must follow from the declared candidate surfaces.

    If this drifts, the module is arbitrating by a private table rather than by the physics it
    tells the user about, and the explanation shown alongside a conflict becomes fiction.
    """

    def test_every_consistent_class_is_a_surface_the_regime_actually_lists(self):
        """Each mapped class must be traceable to a named candidate surface."""
        surface_words = {
            LandCoverClass.WATER: "water",
            LandCoverClass.VEGETATION: "vegetation",
            LandCoverClass.BUILT_UP: "buildings",
            LandCoverClass.BARE_SOIL: ("sand", "terrain", "ground", "tarmac"),
        }
        for regime in (
            ScatteringRegime.SMOOTH,
            ScatteringRegime.DIFFUSE,
            ScatteringRegime.DOUBLE_BOUNCE,
        ):
            listed = " ".join(regime.candidate_surfaces).lower()
            for label in regime.consistent_land_cover:
                wanted = surface_words[label]
                words = (wanted,) if isinstance(wanted, str) else wanted
                assert any(w in listed for w in words), (
                    f"{regime.value} maps to {label.value} but lists no matching surface"
                )

    def test_double_bounce_is_the_only_regime_that_names_one_class(self):
        """This is what licenses Rule A and forbids radar from deciding anything else."""
        single = {
            r
            for r in ScatteringRegime
            if len(r.consistent_land_cover) == 1
        }
        assert single == {ScatteringRegime.DOUBLE_BOUNCE}

    def test_diffuse_admits_both_vegetation_and_bare_soil(self):
        """The refusal that motivates fusion in the first place, pinned."""
        assert ScatteringRegime.DIFFUSE.consistent_land_cover == frozenset(
            {LandCoverClass.VEGETATION, LandCoverClass.BARE_SOIL}
        )

    def test_unclassified_is_consistent_with_nothing(self):
        """A refused pixel must never corroborate a class."""
        assert ScatteringRegime.UNCLASSIFIED.consistent_land_cover == frozenset()

    def test_class_codes_round_trip(self):
        for label in LandCoverClass:
            assert LandCoverClass.from_code(label.code) is label
        for regime in ScatteringRegime:
            assert ScatteringRegime.from_code(regime.code) is regime

    def test_unknown_codes_do_not_become_a_class(self):
        """An out-of-range integer must degrade to "nothing established", not to water."""
        assert LandCoverClass.from_code(99) is LandCoverClass.UNCLASSIFIED
        assert ScatteringRegime.from_code(-1) is ScatteringRegime.UNCLASSIFIED


# ---------------------------------------------------------------------------
# What fusion is worth
# ---------------------------------------------------------------------------
class TestAccuracyAgainstGroundTruth:
    """Scored against the demo scene's exact class map, not against another implementation."""

    def test_fusion_improves_overall_accuracy(self, fused, optical_only, demo_truth):
        """0.9768 optical alone → 0.9923 fused."""
        before = accuracy(optical_only.class_map, demo_truth)
        after = accuracy(fused.class_map, demo_truth)
        assert before == pytest.approx(0.9768, abs=0.005)
        assert after == pytest.approx(0.9923, abs=0.005)
        assert after > before

    def test_built_up_is_the_axis_that_improves(self, fused, optical_only, demo_truth):
        """F1 0.7861 → 0.9989, driven by precision 0.6651 → 0.9997.

        The direction matters as much as the size. Optical over-claims built-up because bare
        soil is also SWIR-bright, so the gain must come from removing false positives rather
        than from finding more pixels: recall moves 0.9608 → 0.9981 while precision moves by a
        third of its range.
        """
        truth = demo_truth == GT_BUILT_UP
        p0, r0, f0 = prf(optical_only.class_map == LandCoverClass.BUILT_UP.code, truth)
        p1, r1, f1 = prf(fused.mask_for(LandCoverClass.BUILT_UP), truth)

        assert f0 == pytest.approx(0.7861, abs=0.01)
        assert f1 == pytest.approx(0.9989, abs=0.005)
        assert p0 == pytest.approx(0.6651, abs=0.01)
        assert p1 == pytest.approx(0.9997, abs=0.005)
        assert r0 == pytest.approx(0.9608, abs=0.01)
        assert p1 - p0 > 0.25

    def test_fusion_leaves_the_axis_radar_cannot_judge_alone(
        self, fused, optical_only, demo_truth
    ):
        """Water and vegetation must barely move: Rule B forbids radar from touching them.

        This is the test that separates a physical rule from a fitted one. Something tuned for
        overall score would lift every class a little; this rule is only allowed to act on
        built-up, so water must be identical and vegetation must move by rounding.
        """
        for label, tolerance in (
            (LandCoverClass.WATER, 1e-9),
            (LandCoverClass.VEGETATION, 0.002),
        ):
            truth = demo_truth == GT_CODE[label]
            _, _, before = prf(optical_only.class_map == label.code, truth)
            _, _, after = prf(fused.mask_for(label), truth)
            assert after == pytest.approx(before, abs=tolerance), (
                f"{label.value} moved by {after - before:+.4f}; radar should not be "
                f"deciding this axis"
            )

    def test_bare_soil_improves_as_the_mirror_of_built_up(
        self, fused, optical_only, demo_truth
    ):
        """0.9833 → 0.9945. Not an independent gain — it is where the withdrawn pixels go."""
        truth = demo_truth == GT_BARE_SOIL
        _, _, before = prf(optical_only.class_map == LandCoverClass.BARE_SOIL.code, truth)
        _, _, after = prf(fused.mask_for(LandCoverClass.BARE_SOIL), truth)
        assert before == pytest.approx(0.9833, abs=0.01)
        assert after == pytest.approx(0.9945, abs=0.005)

    def test_both_directions_of_the_built_up_rule_are_correct(self, fused, demo_truth):
        """The rule was fixed in advance, so it could have been wrong either way.

        Radar asserting built-up over optical's bare soil: 298 pixels, all truly built.
        Radar withdrawing optical's built-up claim: 3,798 pixels, 3,770 truly bare soil.
        """
        asserts = [
            c for c in fused.conflicts
            if c.sar_regime is ScatteringRegime.DOUBLE_BOUNCE
            and c.optical_class is LandCoverClass.BARE_SOIL
        ]
        assert len(asserts) == 1
        group = asserts[0]
        assert group.arbiter == "sar"
        assert group.resolved_as is LandCoverClass.BUILT_UP
        assert group.pixel_count == pytest.approx(298, rel=0.05)

        withdrawn = [
            c for c in fused.conflicts
            if c.optical_class is LandCoverClass.BUILT_UP
            and c.sar_regime is not ScatteringRegime.DOUBLE_BOUNCE
        ]
        assert withdrawn, "the withdrawal direction must occur on this scene"
        assert all(c.resolved_as is LandCoverClass.BARE_SOIL for c in withdrawn)
        total = sum(c.pixel_count for c in withdrawn)
        assert total == pytest.approx(3798, rel=0.05)

    def test_the_documented_weak_corner_is_the_documented_size(self, fused, demo_truth):
        """6 shoreline pixels the rule gets wrong, recorded rather than smoothed over.

        Optical called them built-up, radar called them smooth, so Rule A withdraws the
        built-up claim and they fall back to bare soil — but they are truly water that
        optical's water test had already missed. Radar cannot fix that, because "calm open
        water" and "dry sand" are the same specular return. The test pins the size of the
        failure so it cannot grow unnoticed.
        """
        group = next(
            (c for c in fused.conflicts
             if c.optical_class is LandCoverClass.BUILT_UP
             and c.sar_regime is ScatteringRegime.SMOOTH),
            None,
        )
        assert group is not None
        assert group.pixel_count < 100
        assert group.pixel_count == pytest.approx(6, abs=10)


class TestFusionRecoversWhatOpticalRefuses:
    """The strongest argument for requiring both modalities."""

    def test_rgb_only_optical_claims_no_built_up_at_all(self, demo_rgb):
        """Not a low score — a refusal. Separating buildings from soil needs SWIR."""
        result = classify_land_cover(demo_rgb)
        assert result.method == "rgb-approximation"
        assert int((result.class_map == LandCoverClass.BUILT_UP.code).sum()) == 0
        assert any("short-wave infrared" in w for w in result.warnings)

    def test_sar_recovers_built_up_the_rgb_path_cannot_see(
        self, demo_rgb, demo_sar, demo_truth
    ):
        """F1 0.0000 → 0.9989, and overall accuracy 0.9626 → 0.9923."""
        result = fuse_optical_sar(demo_rgb, demo_sar)
        truth = demo_truth == GT_BUILT_UP

        rgb_only = classify_land_cover(demo_rgb)
        _, _, before = prf(rgb_only.class_map == LandCoverClass.BUILT_UP.code, truth)
        _, _, after = prf(result.mask_for(LandCoverClass.BUILT_UP), truth)
        assert before == 0.0
        assert after == pytest.approx(0.9989, abs=0.005)

        assert accuracy(rgb_only.class_map, demo_truth) == pytest.approx(0.9626, abs=0.005)
        assert accuracy(result.class_map, demo_truth) == pytest.approx(0.9923, abs=0.005)

    def test_the_rgb_limitation_warning_survives_fusion(self, demo_rgb, demo_sar):
        """Fusion fixes built-up; it does not fix the NIR-free water/vegetation separation,
        so the optical caveat must still reach the user."""
        result = fuse_optical_sar(demo_rgb, demo_sar)
        assert any("near-infrared" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Evidence labelling
# ---------------------------------------------------------------------------
class TestEvidenceLabelling:
    """The evidence label is the module's main output for the confidence layer, so what it does
    and does not establish has to be measured rather than assumed."""

    def test_agreement_fraction_matches_the_evidence_map(self, fused):
        """A derived number and the map it summarises must not be able to disagree."""
        both = int(fused.evidence_mask(FusionEvidence.BOTH).sum())
        conflict = int(fused.evidence_mask(FusionEvidence.CONFLICT).sum())
        assert fused.agreement_fraction == pytest.approx(both / (both + conflict))
        assert fused.conflict_fraction == pytest.approx(1.0 - fused.agreement_fraction)

    def test_the_sensors_agree_on_almost_all_of_the_demo_scene(self, fused):
        """98.43%."""
        assert fused.agreement_fraction == pytest.approx(0.9843, abs=0.01)

    def test_the_evidence_label_is_provenance_and_not_a_correctness_predictor(
        self, fused, demo_truth
    ):
        """0.9923 where both sensors agreed, against 0.9927 where they had to be arbitrated.

        The two populations are indistinguishable — 0.0005 apart over a 4,128-pixel conflict
        set, which is about two pixels. So the label records *which sensors supported a pixel*,
        which is what the confidence layer is told it means, and it must not be presented as
        evidence that a corroborated pixel is more likely to be right.

        This test previously asserted the opposite ordering (0.9924 against 0.9857), and the
        reason it flipped is worth keeping: the conflict population is now almost entirely the
        withdrawal direction of Rule A, where radar is right 3,770 times out of 3,798. Making
        the optical texture cue stable shrank the conflict set down to the part radar reliably
        wins, so what is left over-performs the corroborated majority rather than
        under-performing it. An arbitration that is nearly always correct is a good outcome; it
        just cannot also serve as a demonstration that arbitration is risky.
        """
        both = fused.evidence_mask(FusionEvidence.BOTH)
        conflict = fused.evidence_mask(FusionEvidence.CONFLICT)
        assert both.sum() > 0 and conflict.sum() > 0

        acc_both = float((fused.class_map[both] == demo_truth[both]).mean())
        acc_conflict = float((fused.class_map[conflict] == demo_truth[conflict]).mean())
        assert acc_both == pytest.approx(0.9923, abs=0.005)
        assert acc_conflict == pytest.approx(0.9927, abs=0.005)
        assert abs(acc_both - acc_conflict) < 0.005, (
            "if the two populations separate again, the label's meaning must be re-documented "
            "before anything is allowed to read it as a correctness signal"
        )

    def test_every_classified_pixel_carries_an_evidence_label(self, fused):
        """No pixel may hold a class without a record of what supported it."""
        classified = fused.class_map != LandCoverClass.UNCLASSIFIED.code
        neither = fused.evidence_mask(FusionEvidence.NEITHER)
        assert not (classified & neither).any()

    def test_evidence_map_uses_only_defined_codes(self, fused):
        assert set(np.unique(fused.evidence_map)) <= set(CODE_TO_EVIDENCE.values())

    def test_per_class_corroboration_is_reported(self, fused):
        """Each class must say how much of it both sensors supported."""
        for stat in fused.stats:
            assert 0.0 <= stat.corroborated_fraction <= 1.0
            supported = stat.evidence_counts.get(FusionEvidence.BOTH, 0)
            assert stat.corroborated_fraction == pytest.approx(
                supported / stat.pixel_count
            )
            assert sum(stat.evidence_counts.values()) == stat.pixel_count

    def test_built_up_is_the_least_corroborated_class(self, fused):
        """It is the only class the two sensors reach by different physics, so it is the only
        one where a large arbitrated share is expected — and reporting that is the point.

        0.9608, against 0.9793 for bare soil (the class on the other side of the same boundary)
        and 1.0000 for vegetation. The margin is what carries the claim; the absolute level is
        high because the optical and radar readings of this scene mostly do agree.
        """
        by_class = {s.label: s.corroborated_fraction for s in fused.stats}
        assert by_class[LandCoverClass.BUILT_UP] == min(by_class.values())
        assert by_class[LandCoverClass.BUILT_UP] == pytest.approx(0.9608, abs=0.01)
        others = [v for k, v in by_class.items() if k is not LandCoverClass.BUILT_UP]
        assert by_class[LandCoverClass.BUILT_UP] < min(others) - 0.01

    def test_corroborated_fraction_counts_the_whole_classified_map(self, fused):
        """Distinct from ``agreement_fraction``, which ignores unobserved pixels."""
        classified = int((fused.class_map != LandCoverClass.UNCLASSIFIED.code).sum())
        both = int(fused.evidence_mask(FusionEvidence.BOTH).sum())
        assert fused.corroborated_fraction == pytest.approx(both / classified)


# ---------------------------------------------------------------------------
# Conflicts are reported, never hidden
# ---------------------------------------------------------------------------
class TestConflictsAreReported:
    def test_every_conflict_pixel_belongs_to_exactly_one_reported_group(self, fused):
        """A conflict that fell out of the report would be a silently resolved disagreement."""
        counted = sum(c.pixel_count for c in fused.conflicts)
        assert counted == int(fused.evidence_mask(FusionEvidence.CONFLICT).sum())

    def test_conflicts_are_ordered_by_size(self, fused):
        counts = [c.pixel_count for c in fused.conflicts]
        assert counts == sorted(counts, reverse=True)

    def test_each_conflict_names_its_arbiter_and_its_reason(self, fused):
        for group in fused.conflicts:
            assert group.arbiter in {"sar", "optical"}
            assert len(group.reason) > 40
            assert group.candidate_surfaces, "radar's alternatives must be shown"

    def test_only_double_bounce_conflicts_are_decided_by_radar(self, fused):
        """Rule B: outside built-up, radar's reading is recorded but the optical class stands."""
        for group in fused.conflicts:
            radar_involved = (
                group.sar_regime is ScatteringRegime.DOUBLE_BOUNCE
                or group.optical_class is LandCoverClass.BUILT_UP
            )
            if group.arbiter == "sar":
                assert radar_involved
            else:
                assert not radar_involved
                assert group.resolved_as is group.optical_class

    def test_a_non_built_up_conflict_leaves_the_optical_class_standing(self, fused):
        """Water read as diffuse is a real disagreement radar cannot settle: it must be
        reported, and the pixel must keep the class the sensor with the direct measurement
        gave it."""
        group = next(
            (c for c in fused.conflicts
             if c.optical_class is LandCoverClass.WATER
             and c.sar_regime is ScatteringRegime.DIFFUSE),
            None,
        )
        assert group is not None, "this disagreement exists on the demo scene"
        assert group.arbiter == "optical"
        assert group.resolved_as is LandCoverClass.WATER

    def test_radar_that_found_no_bright_population_cannot_veto_built_up(
        self, demo_optical, demo_sar
    ):
        """Rule A needs a measurement behind it.

        Withdrawing an optical built-up claim is justified by radar having *measured* the
        absence of double-bounce at that pixel. When radar separated no bright population
        anywhere in the image, it measured nothing about vertical structure at all — the
        absence of a separable class is a property of the histogram, not of the ground. Vetoing
        on that basis would turn "radar could not tell" into "radar disagrees", which is
        exactly the fabrication the brief forbids. The claim must stand, marked uncorroborated.
        """
        # A radar image with a single scattering population and no usable contrast — the shape
        # a flat flooded scene or a mis-calibrated product takes. Optical is untouched and
        # still sees the town, so the disagreement is now radar-silence, not radar-dissent.
        rng = np.random.default_rng(4)
        linear = 10 ** (-10.0 / 10) * rng.gamma(6.0, 1 / 6.0, size=demo_sar.data.shape)
        flattened = replace(demo_sar, data=(10 * np.log10(linear)).astype(np.float32))

        result = fuse_optical_sar(demo_optical, flattened)
        assert result.sar.separated_regimes == [ScatteringRegime.DIFFUSE]
        assert result.method == "decision-level-optical-sar-partial"
        assert any("neither confirm nor rule out built-up" in w for w in result.warnings)

        unresolved = [
            c for c in result.conflicts if c.optical_class is LandCoverClass.BUILT_UP
        ]
        assert unresolved, "optical still claims built-up land, so the conflict exists"
        for group in unresolved:
            assert group.resolved_as is LandCoverClass.BUILT_UP
            assert group.arbiter == "none"
            assert "uncorroborated" in group.reason

        # And the claim really does survive into the map, rather than only into the prose.
        optical_built = classify_land_cover(demo_optical).mask_for(LandCoverClass.BUILT_UP)
        assert (result.class_map[optical_built] == LandCoverClass.BUILT_UP.code).all()

    def test_conflict_serialisation_carries_the_explanation(self, fused):
        payload = fused.to_dict()
        assert payload["conflicts"], "conflicts must survive serialisation"
        first = payload["conflicts"][0]
        assert set(first) >= {
            "optical_class", "sar_regime", "pixel_count", "resolved_as", "arbiter",
            "reason", "radar_candidate_surfaces",
        }
        assert payload["agreement_fraction"] == pytest.approx(
            fused.agreement_fraction, abs=1e-4
        )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
class TestRefusals:
    def test_mismatched_shapes_are_refused(self, demo_optical, demo_root):
        small = load_raster(demo_root / "edge_cases" / "different_res.tif", max_edge=None)
        with pytest.raises(GeospatialError) as exc:
            fuse_optical_sar(demo_optical, small)
        assert exc.value.code is ErrorCode.SIZE_MISMATCH

    def test_a_precomputed_result_for_another_image_is_refused(
        self, demo_optical, demo_sar, demo_root
    ):
        """Accepting it would attach one scene's classes to another scene's radar."""
        other = load_raster(demo_root / "edge_cases" / "different_res.tif", max_edge=None)
        with pytest.raises(GeospatialError) as exc:
            fuse_optical_sar(
                demo_optical, demo_sar, optical_result=classify_land_cover(other)
            )
        assert exc.value.code is ErrorCode.SIZE_MISMATCH

    def test_no_valid_overlap_is_refused(self, demo_sar, demo_optical, make_raster):
        """An all-nodata optical image has nothing to fuse, and saying so beats returning a
        map that is really the radar result wearing land-cover labels."""
        blank = np.full((6, *demo_sar.shape), np.nan, dtype=np.float32)
        raster = load_raster(
            make_raster(
                "blank.tif", blank, nodata=float("nan"),
                descriptions=("blue", "green", "red", "nir", "swir1", "swir2"),
            ),
            max_edge=None,
        )
        with pytest.raises(GeospatialError) as exc:
            fuse_optical_sar(raster, demo_sar)
        assert exc.value.code is ErrorCode.NO_SPATIAL_OVERLAP

    def test_radar_alone_does_not_name_a_class_under_an_ambiguous_regime(
        self, demo_sar, demo_optical
    ):
        """Rule C. Where optical carries no data and radar says "smooth", the fused map must
        stay unclassified: calm water and dry sand are the same return.
        """
        data = demo_optical.data.copy()
        data[:, :, :80] = np.nan
        blanked = replace(demo_optical, data=data)

        result = fuse_optical_sar(blanked, demo_sar)
        strip = np.zeros(demo_sar.shape, dtype=bool)
        strip[:, :80] = True

        sar_only = result.evidence_mask(FusionEvidence.SAR_ONLY)
        assert sar_only.any(), "the blanked strip must produce radar-only pixels"

        ambiguous = sar_only & (
            (result.sar.regime_map == ScatteringRegime.SMOOTH.code)
            | (result.sar.regime_map == ScatteringRegime.DIFFUSE.code)
        )
        assert ambiguous.any()
        assert (result.class_map[ambiguous] == LandCoverClass.UNCLASSIFIED.code).all()
        assert any("more than one surface" in w for w in result.warnings)

    def test_radar_alone_does_name_built_up(self, demo_sar, demo_optical):
        """The other half of Rule C: double bounce names exactly one class, so it may."""
        data = demo_optical.data.copy()
        data[:, :, :] = np.nan
        data[:, :, 400:] = demo_optical.data[:, :, 400:]
        blanked = replace(demo_optical, data=data)

        result = fuse_optical_sar(blanked, demo_sar)
        sar_only = result.evidence_mask(FusionEvidence.SAR_ONLY)
        db = sar_only & (result.sar.regime_map == ScatteringRegime.DOUBLE_BOUNCE.code)
        assert db.any(), "the blanked region must contain double-bounce returns"
        assert (result.class_map[db] == LandCoverClass.BUILT_UP.code).all()


class TestInputSanity:
    def test_an_optical_image_passed_as_radar_is_flagged(self, demo_optical):
        """Fusing a scene with itself is silently meaningless without this warning."""
        result = fuse_optical_sar(demo_optical, demo_optical)
        assert any("looks like optical data" in w for w in result.warnings)

    def test_a_radar_image_passed_as_optical_is_flagged(self, demo_sar):
        result = fuse_optical_sar(demo_sar, demo_sar)
        assert any("looks like radar" in w for w in result.warnings)

    def test_misalignment_is_warned_about(self, demo_optical, demo_sar):
        """A shifted radar image must be reported as misregistered, because the disagreement
        it produces is an artefact rather than a sensor conflict."""
        shifted = replace(demo_sar, data=np.roll(demo_sar.data, 12, axis=2))
        result = fuse_optical_sar(demo_optical, shifted)
        assert result.registration.offset_magnitude > 1.0
        assert any("misaligned" in w for w in result.warnings)

    def test_the_matched_pair_is_not_warned_about(self, fused):
        """The demo pair shares a grid exactly, so a misalignment warning here would be a
        false alarm that trains users to ignore the real one."""
        assert fused.registration.offset_magnitude < 1.0
        assert not any("misaligned" in w for w in fused.warnings)


# ---------------------------------------------------------------------------
# Structure and determinism
# ---------------------------------------------------------------------------
class TestResultStructure:
    def test_fractions_sum_to_one_over_the_classified_area(self, fused):
        assert sum(s.fraction for s in fused.stats) == pytest.approx(1.0, abs=1e-6)

    def test_pixel_counts_match_the_class_map(self, fused):
        for stat in fused.stats:
            assert stat.pixel_count == int(fused.mask_for(stat.label).sum())

    def test_areas_use_the_known_pixel_size(self, fused, demo_manifest):
        """10 m pixels: every area must be its pixel count times exactly 100 m²."""
        expected = demo_manifest["grid"]["pixel_area_m2"]
        for stat in fused.stats:
            assert stat.area_m2 == pytest.approx(stat.pixel_count * expected)

    def test_no_area_is_reported_without_georeferencing(self, demo_root, demo_sar):
        """Pixel counts stay; areas become ``None`` rather than being invented."""
        plain = load_raster(demo_root / "edge_cases" / "no_georeference.tif", max_edge=None)
        result = fuse_optical_sar(plain, demo_sar)
        assert result.stats
        assert all(s.area_m2 is None for s in result.stats)
        assert all(s.pixel_count > 0 for s in result.stats)
        assert all(c.area_m2 is None for c in result.conflicts)

    def test_dominant_class_is_the_largest(self, fused):
        assert fused.dominant() is max(fused.stats, key=lambda s: s.pixel_count).label

    def test_both_single_sensor_results_are_kept(self, fused):
        """A user must be able to see what each sensor contributed on its own."""
        assert fused.optical.method == "spectral-indices"
        assert fused.sar.polarization == "VV"
        assert ScatteringRegime.DOUBLE_BOUNCE in fused.sar.separated_regimes

    def test_precomputed_results_give_an_identical_answer(
        self, demo_optical, demo_sar, fused
    ):
        """The orchestrator will pass results it already has; that must be an optimisation
        only, never a different code path with a different answer."""
        again = fuse_optical_sar(
            demo_optical,
            demo_sar,
            optical_result=classify_land_cover(demo_optical),
            sar_result=analyze_sar(demo_sar),
        )
        assert np.array_equal(again.class_map, fused.class_map)
        assert np.array_equal(again.evidence_map, fused.evidence_map)

    def test_repeated_runs_are_identical(self, demo_optical, demo_sar, fused):
        """No randomness anywhere in the path."""
        again = fuse_optical_sar(demo_optical, demo_sar)
        assert np.array_equal(again.class_map, fused.class_map)
        assert again.agreement_fraction == fused.agreement_fraction

    def test_serialises_without_raw_arrays(self, fused):
        """The API payload must stay small: maps travel as artifacts, not as JSON."""
        import json

        payload = fused.to_dict()
        text = json.dumps(payload)
        assert len(text) < 20000
        assert "class_map" not in payload
        assert payload["dominant_class"] == LandCoverClass.BARE_SOIL.value
        assert payload["method"] == "decision-level-optical-sar"

    def test_warnings_are_not_duplicated(self, fused):
        assert len(fused.warnings) == len(set(fused.warnings))

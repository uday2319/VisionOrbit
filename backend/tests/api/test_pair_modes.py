"""End-to-end tests for the two-image paths: optical+SAR fusion and bi-temporal change (§5, §6).

These cover the failure a user actually hit — selecting a two-image mode, uploading two scenes and
getting a refusal instead of an analysis — at the level it was reported, through the HTTP API, with
rasters written to disk. Three separate defects met there, so each is pinned:

* **the declared sensor was thrown away.** :func:`app.geospatial.raster.infer_modality` reads only
  filename tokens and band descriptions, so a real SAR scene exported as ``subset_1.tif`` with an
  unlabelled band is undetermined. The client can declare it (``modality_hint``), but the analysis
  route re-opened the raster without the modality persisted at upload time, so the declaration was
  lost between the two requests and the pair normalised to ``bitemporal_pair``.
* **a single-scene query on a pair was rejected for its image count.** "Map the land cover using both
  sensors" classifies as land-cover analysis, which the contract gate then refused with "works on
  exactly 1 image" — naming the wrong problem, since the fusion tool answers exactly that question
  from both inputs.
* **an undetermined sensor surfaced as a band complaint.** The run reached the change service and
  failed with "no bands in common", which describes the symptom rather than the cause the user can
  act on.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.transform import Affine

from app.main import app
from tests.conftest import TEST_ORIGIN_X, TEST_ORIGIN_Y, TEST_PIXEL_M, write_test_raster

_OPTICAL_BANDS = ("blue", "green", "red", "nir")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _transform() -> Affine:
    return Affine(TEST_PIXEL_M, 0.0, TEST_ORIGIN_X, 0.0, -TEST_PIXEL_M, TEST_ORIGIN_Y)


@pytest.fixture
def scenes(tmp_path):
    """Co-registered scenes written with *neutral* filenames and no sensor metadata.

    The naming is the point. Every scene is ``scene_a.tif`` / ``scene_b.tif``, so nothing in the
    file tells the backend which sensor produced it — the situation a real export from SNAP or a
    subsetting tool produces, and the one the declared-modality path exists to serve. A test using
    ``S1A_IW_GRDH_vv.tif`` would pass on the filename token alone and prove nothing.
    """

    def _optical(name: str, *, shift: float = 0.0) -> str:
        rng = np.random.default_rng(11)
        arr = np.empty((4, 64, 64), dtype=np.uint16)
        # Vegetation: low red, high NIR. The shifted copy turns a block into water (NIR collapses),
        # so a bi-temporal pair has real, measurable change rather than noise.
        arr[:] = np.array([600, 900, 700, 4200], dtype=np.uint16)[:, None, None]
        arr = (arr + rng.integers(0, 60, arr.shape)).astype(np.uint16)
        if shift:
            arr[3, 16:48, 16:48] = 300
            arr[1, 16:48, 16:48] = 1400
        return str(
            write_test_raster(
                tmp_path / name, arr, transform=_transform(), descriptions=_OPTICAL_BANDS
            )
        )

    def _sar(name: str) -> str:
        rng = np.random.default_rng(12)
        # Speckled backscatter, one band, and deliberately *no* band description.
        gamma = rng.gamma(shape=4.0, scale=1.0, size=(1, 64, 64)) / 4.0
        arr = (10.0 * np.log10(gamma) - 14.0).astype(np.float32)
        arr[0, 16:48, 16:48] = -3.0  # a bright block: built-up / double bounce
        return str(write_test_raster(tmp_path / name, arr, transform=_transform()))

    return {"optical": _optical, "sar": _sar}


def _upload(client: TestClient, path: str, *, hint: str | None = None) -> dict:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    with open(path, "rb") as handle:
        data = {"modality_hint": hint} if hint else None
        response = client.post(
            "/api/upload", files={"file": (name, handle, "image/tiff")}, data=data
        )
    assert response.status_code == 201, response.text
    return response.json()


def _analyze(client: TestClient, query: str, image_ids: list[str]):
    return client.post("/api/analyze", json={"query": query, "image_ids": image_ids})


class TestModalityHintOnUpload:
    def test_declared_sar_is_reported_back(self, client, scenes):
        body = _upload(client, scenes["sar"]("scene_b.tif"), hint="sar")
        assert body["modality"] == "sar"

    def test_without_a_hint_the_same_file_is_undetermined(self, client, scenes):
        # The premise of the whole hint mechanism: nothing in this file identifies the sensor.
        body = _upload(client, scenes["sar"]("scene_b.tif"))
        assert body["modality"] == "unknown"

    def test_unknown_hint_is_rejected_not_ignored(self, client, scenes):
        path = scenes["sar"]("scene_b.tif")
        with open(path, "rb") as handle:
            response = client.post(
                "/api/upload",
                files={"file": ("scene_b.tif", handle, "image/tiff")},
                data={"modality_hint": "radar"},
            )
        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == "WRONG_MODALITY"
        # The message lists what would have been accepted instead of failing blankly.
        assert "optical" in body["message"] and "sar" in body["message"]


class TestOpticalSarPair:
    """A declared optical+SAR pair must run the fusion the user asked for."""

    @pytest.fixture
    def pair(self, client, scenes):
        optical = _upload(client, scenes["optical"]("scene_a.tif"), hint="optical")
        sar = _upload(client, scenes["sar"]("scene_b.tif"), hint="sar")
        assert (optical["modality"], sar["modality"]) == ("optical", "sar")
        return [optical["image_id"], sar["image_id"]]

    def test_fusion_runs(self, client, pair):
        response = _analyze(client, "Combine the optical and radar images to map land cover.", pair)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == "optical_sar_pair"
        assert body["task"] == "optical_sar_analysis"
        assert body["tool"] == "fusion-decision-cv"

    def test_declared_modality_survives_into_the_analysis(self, client, pair):
        # The regression: the analysis route used to re-open each raster without the modality
        # recorded at upload, so a hinted SAR scene reverted to undetermined and the pair was
        # mislabelled bi-temporal — a false provenance claim in the trace and the database row.
        body = _analyze(client, "Combine both sensors.", pair).json()
        assert body["modalities"] == ["optical", "sar"]
        assert body["input_count"] == 2

    def test_single_scene_query_is_answered_by_the_pair_analysis(self, client, pair):
        # Previously HTTP 422 TOOL_INPUT_CONTRACT_ERROR: "works on exactly 1 image(s)".
        response = _analyze(client, "Map the land cover using both sensors.", pair)
        assert response.status_code == 200, response.text
        assert response.json()["tool"] == "fusion-decision-cv"

    def test_the_substitution_is_visible_in_the_trace(self, client, pair):
        # An overridden route must be stated, not implied (§49): the routing step's own text has to
        # say which analysis ran and why, since the user asked in single-scene words.
        body = _analyze(client, "What land cover is present?", pair).json()
        routing = " ".join(
            step["details"] for step in body["execution_trace"]["steps"] if step["details"]
        )
        assert "optical/SAR fusion" in routing
        assert "two images were provided" in routing

    def test_answer_is_not_a_contract_error_dressed_as_weak_evidence(self, client, pair):
        # §27/§0: a tool-level failure must never surface as "insufficient evidence".
        body = _analyze(client, "Combine the optical and radar images.", pair).json()
        assert "insufficient evidence" not in body["answer"].lower()
        assert body["answer"].strip()


class TestConfidenceReportIsPublished:
    """The measured confidence report must reach the client, not just its score (§27).

    Only ``confidence_score`` and ``confidence_level`` used to be published, so the UI had no
    measured reasons to show and no sufficiency verdict to read. It invented both: three fixed
    reason strings, and its own ``score >= 0.5`` rule — which headlined a measured ``low`` answer as
    "Insufficient evidence for a reliable conclusion" while the same card showed a Low badge.
    """

    @pytest.fixture
    def body(self, client, scenes):
        optical = _upload(client, scenes["optical"]("scene_a.tif"), hint="optical")
        sar = _upload(client, scenes["sar"]("scene_b.tif"), hint="sar")
        response = _analyze(
            client,
            "Combine the optical and radar images to map land cover.",
            [optical["image_id"], sar["image_id"]],
        )
        assert response.status_code == 200, response.text
        return response.json()

    def test_report_carries_the_measured_factors(self, body):
        report = body["confidence"]
        assert report["factors"], "no measured factors were published"
        # Every factor states what it measured, rather than being a single synthetic score.
        for factor in report["factors"]:
            assert factor["reason"].strip()
            assert 0.0 <= factor["value"] <= 1.0
            assert factor["kind"] in ("contributor", "gate")

    def test_reasons_are_the_factor_reasons_weakest_first(self, body):
        report = body["confidence"]
        assert report["reasons"] == [
            f["reason"] for f in sorted(report["factors"], key=lambda f: f["value"])
        ]

    def test_the_limiting_factor_is_named(self, body):
        report = body["confidence"]
        assert report["limiting_factor"] in {f["name"] for f in report["factors"]}

    def test_sufficiency_is_the_backends_verdict_not_a_threshold(self, body):
        # A measured 'low' answer is sufficient to state with caveats; only 'insufficient' is not.
        report = body["confidence"]
        assert report["sufficient"] is (report["level"] != "insufficient")

    def test_score_and_level_agree_with_the_flat_fields(self, body):
        report = body["confidence"]
        assert report["score"] == pytest.approx(body["confidence_score"], abs=1e-4)
        assert report["level"] == body["confidence_level"]

    def test_the_report_survives_a_re_read(self, client, body):
        # The detail endpoint reads the persisted trace, so the factors must still be there.
        reread = client.get(f"/api/analysis/{body['analysis_id']}")
        assert reread.status_code == 200, reread.text
        assert reread.json()["confidence"] == body["confidence"]

    def test_the_tool_that_ran_is_named_at_the_top_level(self, body):
        # The UI read execution_trace.steps[1].tool_name and so credited 'ModeNormalizer'; the
        # response's own field is the authority and must name the specialist.
        assert body["tool"] == "fusion-decision-cv"
        assert body["tier"] == "classical"
        step_tools = [s["tool_name"] for s in body["execution_trace"]["steps"]]
        assert "ModeNormalizer" in step_tools  # the step really is there — just not the analyst


class TestBitemporalPair:
    """Two dates of one sensor must run change analysis, hint or no hint."""

    @pytest.fixture
    def pair(self, client, scenes):
        t1 = _upload(client, scenes["optical"]("scene_a.tif"))
        t2 = _upload(client, scenes["optical"]("scene_b.tif", shift=1.0))
        return [t1["image_id"], t2["image_id"]]

    def test_change_runs(self, client, pair):
        # "What changed…?" is a *question* about change, so it routes to change VQA rather than to
        # the change *report*. Both rest on the same measurement; they differ in what the answer
        # addresses, which is the separation the change-VQA tool exists to make.
        response = _analyze(client, "What changed between the two dates?", pair)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == "bitemporal_pair"
        assert body["task"] == "change_vqa"
        assert body["tool"] == "change-vqa-cv"

    def test_show_me_the_change_runs_the_change_report(self, client, pair):
        """An imperative mood asks for the report, not an answer to a question."""
        body = _analyze(client, "Show me the change between the two dates.", pair).json()
        assert body["mode"] == "bitemporal_pair"
        assert body["task"] == "change_detection"
        assert body["tool"] == "change-cva-cv"

    def test_measured_change_is_reported(self, client, pair):
        # The second scene has a genuine 32x32 vegetation-to-water block, so a real measurement
        # must come back rather than a "no change" default.
        body = _analyze(client, "What changed between the two dates?", pair).json()
        assert "%" in body["answer"]
        assert body["data"]["changed_pixels"] > 0

    def test_single_scene_query_is_answered_by_the_change_analysis(self, client, pair):
        response = _analyze(client, "What land cover is present?", pair)
        assert response.status_code == 200, response.text
        assert response.json()["task"] == "change_detection"

    def test_two_declared_sar_scenes_are_compared(self, client, scenes):
        # Bi-temporal mode offers SAR in the interface, so two radar dates must not dead-end.
        # The query carries no task signal, so the mode is settled from the images alone — the
        # input-default path, not the classifier. (An empty string is rejected by the request
        # schema, which requires at least one character.)
        a = _upload(client, scenes["sar"]("scene_a.tif"), hint="sar")
        b = _upload(client, scenes["sar"]("scene_b.tif"), hint="sar")
        response = _analyze(client, "hello there", [a["image_id"], b["image_id"]])
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == "bitemporal_pair"
        assert body["modalities"] == ["sar", "sar"]
        assert body["tool"] == "change-cva-cv"


class TestUndeterminedSensorIsExplained:
    """With no hint, an optical+SAR upload cannot be recognised as one — so say what to do."""

    def test_refusal_names_the_cause_and_the_remedy(self, client, scenes):
        optical = _upload(client, scenes["optical"]("scene_a.tif"))
        undetermined = _upload(client, scenes["sar"]("scene_b.tif"))
        assert undetermined["modality"] == "unknown"
        response = _analyze(
            client,
            "Combine the optical and radar images to map land cover.",
            [optical["image_id"], undetermined["image_id"]],
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == "MISSING_BAND"
        # Names which image, why, and how to fix it — not just "no bands in common".
        assert "image 2" in body["message"]
        assert "could not be determined" in body["message"]
        assert "modality_hint" in body["message"]

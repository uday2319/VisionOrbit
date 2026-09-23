"""End-to-end tests for the imagery a result page shows.

A live run used to return none at all: the renderer existed only in ``scripts/
generate_frontend_fixtures.py``, so a recorded demo case displayed a classified overlay while the
very same tool run through the API displayed "No rendered imagery for this analysis." These tests
pin the whole path — the PNGs are written under the storage root, served by the ``/storage`` mount,
described in the analyse response, and still described when the analysis is re-read later.

What is *not* asserted is any particular answer or confidence: this is about the pictures, and the
tools' measurements are tested where they are made.
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from app.api.routes.analysis import _ARTIFACT_DIRNAME, _stored_list
from app.config import get_settings
from app.main import app

TRANSFORM = from_origin(500000, 3000000, 10, 10)
OPTICAL_BANDS = ("blue", "green", "red", "nir")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _write(path, array, *, epsg: int | None = 32643, descriptions=None):
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count, dtype=array.dtype,
        crs=f"EPSG:{epsg}" if epsg else None,
        transform=TRANSFORM if epsg else rasterio.Affine.identity(),
    ) as dst:
        dst.write(array)
        for i, name in enumerate(descriptions or (), start=1):
            dst.set_band_description(i, name)
    return path


def _optical(shift: int = 0) -> np.ndarray:
    """A 4-band scene: vegetation everywhere, water in the north-west block."""
    arr = np.zeros((4, 60, 60), dtype=np.uint16)
    arr[0], arr[1], arr[2], arr[3] = 500, 700, 450, 4500
    arr[3, 10 + shift : 30 + shift, 10:30] = 300  # NIR collapse => water/non-vegetation
    arr[1, 10 + shift : 30 + shift, 10:30] = 900
    return arr


def _sar() -> np.ndarray:
    arr = np.full((1, 60, 60), -12.0, dtype=np.float32)
    arr[0, 10:30, 10:30] = 3.0  # bright double-bounce block
    return arr


def _upload(client, path) -> str:
    with open(path, "rb") as fh:
        res = client.post("/api/upload", files={"file": (path.name, fh, "image/tiff")})
    assert res.status_code in (200, 201), res.text
    return res.json()["image_id"]


def _analyze(client, query: str, image_ids: list[str]) -> dict:
    res = client.post("/api/analyze", json={"query": query, "image_ids": image_ids})
    assert res.status_code == 200, res.text
    return res.json()


def _run_land_cover(client, tmp_path) -> dict:
    path = _write(tmp_path / "imagery_optical.tif", _optical(), descriptions=OPTICAL_BANDS)
    return _analyze(
        client, "What land cover types are present in this scene?", [_upload(client, path)]
    )


class TestRenderedImagery:
    def test_a_land_cover_run_returns_a_base_and_its_class_map(self, client, tmp_path):
        body = _run_land_cover(client, tmp_path)
        kinds = [a["kind"] for a in body["artifacts"]]
        assert kinds == ["base", "class_map"]
        base, overlay = body["artifacts"]
        assert base["label"] == "Input image"
        assert overlay["label"] == "Land-cover classification"
        assert overlay["legend"], "the overlay must ship the key it was drawn from"
        assert all(e["color"].startswith("#") for e in overlay["legend"])

    def test_the_urls_point_into_this_analysis_own_directory(self, client, tmp_path):
        body = _run_land_cover(client, tmp_path)
        prefix = f"/storage/{_ARTIFACT_DIRNAME}/{body['analysis_id']}/"
        assert all(a["url"].startswith(prefix) for a in body["artifacts"])

    def test_the_files_exist_and_are_served_by_the_storage_mount(self, client, tmp_path):
        """The URL is not a claim about a file — the file is there, and the mount returns it."""
        body = _run_land_cover(client, tmp_path)
        root = get_settings().storage_root
        for artifact in body["artifacts"]:
            on_disk = root / artifact["url"].removeprefix("/storage/")
            assert on_disk.is_file(), on_disk
            served = client.get(artifact["url"])
            assert served.status_code == 200
            assert served.headers["content-type"] == "image/png"
            assert served.content == on_disk.read_bytes()

    def test_the_two_images_differ(self, client, tmp_path):
        """A base and an overlay that are byte-identical would mean nothing was drawn."""
        body = _run_land_cover(client, tmp_path)
        base, overlay = (client.get(a["url"]).content for a in body["artifacts"])
        assert base != overlay

    def test_a_sar_run_returns_the_scattering_regimes(self, client, tmp_path):
        path = _write(tmp_path / "imagery_sar_vv.tif", _sar(), descriptions=("vv",))
        body = _analyze(client, "Describe the SAR backscatter in this image.", [_upload(client, path)])
        # One SAR scene: the backscatter analyser, under the shared radar task name.
        assert body["tool"] == "sar-backscatter-cv"
        assert [a["kind"] for a in body["artifacts"]] == ["base", "regime_map"]
        assert {e["label"] for e in body["artifacts"][1]["legend"]} == {
            "Smooth", "Diffuse", "Double bounce",
        }

    def test_a_bitemporal_run_returns_the_change_mask(self, client, tmp_path):
        t1 = _write(tmp_path / "imagery_t1.tif", _optical(), descriptions=OPTICAL_BANDS)
        t2 = _write(tmp_path / "imagery_t2.tif", _optical(shift=15), descriptions=OPTICAL_BANDS)
        body = _analyze(
            client, "What changed between the two dates?",
            [_upload(client, t1), _upload(client, t2)],
        )
        assert body["mode"] == "bitemporal_pair"
        assert [a["kind"] for a in body["artifacts"]] == ["base", "change_mask"]

    def test_a_grounding_run_returns_the_grounded_regions(self, client, tmp_path):
        path = _write(tmp_path / "imagery_ground.tif", _optical(), descriptions=OPTICAL_BANDS)
        body = _analyze(client, "Where is the water?", [_upload(client, path)])
        assert body["task"] == "grounding"
        assert [a["kind"] for a in body["artifacts"]] == ["base", "grounding"]


class TestInputMetadata:
    def test_inputs_report_the_raster_as_the_analysis_read_it(self, client, tmp_path):
        """The loader's own resolution — band roles, driver, transform — not the upload's echo.

        The upload response cannot supply these, which is why the Inputs panel had to make do with
        the little it does carry, and had nothing at all after a reload.
        """
        body = _run_land_cover(client, tmp_path)
        assert len(body["inputs"]) == 1
        meta = body["inputs"][0]
        assert meta["filename"] == "imagery_optical.tif"
        assert meta["modality"] == "optical"
        assert meta["band_roles"] == list(OPTICAL_BANDS)
        assert meta["band_descriptions"] == list(OPTICAL_BANDS)
        assert meta["driver"] == "GTiff"
        assert meta["georeferenced"] is True
        assert meta["crs_epsg"] == 32643
        assert meta["transform"][:2] == [10.0, 0.0]
        assert meta["bounds_wgs84"]["south"] < meta["bounds_wgs84"]["north"]

    def test_inputs_are_reported_in_caller_order(self, client, tmp_path):
        t1 = _write(tmp_path / "order_t1.tif", _optical(), descriptions=OPTICAL_BANDS)
        t2 = _write(tmp_path / "order_t2.tif", _optical(shift=15), descriptions=OPTICAL_BANDS)
        body = _analyze(
            client, "What changed between the two dates?",
            [_upload(client, t1), _upload(client, t2)],
        )
        assert [m["filename"] for m in body["inputs"]] == ["order_t1.tif", "order_t2.tif"]


class TestUngeoreferencedInput:
    def test_no_bounds_are_claimed_and_the_imagery_still_renders(self, client, tmp_path):
        """§8: a picture with no CRS gets ``bounds_wgs84: null``, which the UI reads as pixel view."""
        path = _write(tmp_path / "no_crs.tif", _optical(), epsg=None, descriptions=OPTICAL_BANDS)
        body = _analyze(
            client, "What land cover types are present in this scene?", [_upload(client, path)]
        )
        assert body["artifacts"], "imagery is rendered whether or not the input is georeferenced"
        assert all(a["bounds_wgs84"] is None for a in body["artifacts"])
        assert body["inputs"][0]["georeferenced"] is False
        assert body["inputs"][0]["bounds_wgs84"] is None

    def test_a_file_that_names_no_band_is_still_a_valid_response(self, client, tmp_path):
        """Unnamed bands report as nulls, positionally — and must not fail response validation.

        ``band_descriptions`` was typed ``list[str]`` on the wire, so the first unnamed band turned
        a perfectly good analysis into a 500.
        """
        path = _write(tmp_path / "unnamed.tif", _optical(), epsg=None)
        body = _analyze(
            client, "What land cover types are present in this scene?", [_upload(client, path)]
        )
        assert body["inputs"][0]["band_descriptions"] == [None, None, None, None]


class TestReReadAnalysis:
    def test_a_stored_analysis_keeps_its_imagery_and_inputs(self, client, tmp_path):
        """Reloading a result page must not empty it.

        The descriptors ride in the trace blob, so this is also what proves the read side: nothing
        is re-rendered and nothing is reconstructed, the run's own record is returned.
        """
        body = _run_land_cover(client, tmp_path)
        again = client.get(f"/api/analysis/{body['analysis_id']}")
        assert again.status_code == 200
        stored = again.json()
        assert stored["artifacts"] == body["artifacts"]
        assert stored["inputs"] == body["inputs"]
        # And the copy stored with the trace is the same one.
        assert stored["execution_trace"]["artifacts"] == body["artifacts"]
        assert stored["execution_trace"]["inputs"] == body["inputs"]
        assert client.get(stored["artifacts"][0]["url"]).status_code == 200


class TestStoredList:
    """``_stored_list`` is the read side: it returns what was recorded, or nothing.

    A row persisted before this field existed has no imagery, and a run whose render failed has
    none either. Neither gap is filled in: an invented URL would 404, and invented input metadata
    would describe a file the analysis may never have read.
    """

    def test_returns_the_recorded_entries(self):
        assert _stored_list({"artifacts": [{"kind": "base"}]}, "artifacts") == [{"kind": "base"}]

    @pytest.mark.parametrize(
        "trace",
        [{}, {"artifacts": None}, {"artifacts": "base.png"}, {"artifacts": {}}],
        ids=["absent", "null", "string", "object"],
    )
    def test_anything_that_is_not_a_list_reads_as_nothing_recorded(self, trace):
        assert _stored_list(trace, "artifacts") == []

    def test_non_dict_entries_are_dropped_rather_than_passed_on(self):
        assert _stored_list({"inputs": [{"filename": "a.tif"}, "b.tif", None]}, "inputs") == [
            {"filename": "a.tif"}
        ]

"""Integration tests for FastAPI REST API endpoints."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data


def test_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    tools = response.json()
    assert isinstance(tools, list)
    assert len(tools) > 0


def test_history_endpoint_empty(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data


def test_upload_invalid_format(client):
    file_content = b"fake image content"
    files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["error_code"] == "UNSUPPORTED_FORMAT"


def test_models_endpoint_lists_the_registered_components(client):
    """/api/models lists the registered tools, under the names they actually have.

    This test used to require names like ``OpticalLandcoverSpecialist`` and ``FusionSpecialist``.
    No such component exists — nothing in the codebase has ever been called that. They named a
    model line-up the project does not have, which is exactly the claim §26 forbids, and the test
    failed against the endpoint that reports the truth. The registry entries below are the tools
    the router can actually dispatch to; each declares the input contract it is gated on.

    The count went from seven to eight when change VQA stopped being an alias of change detection
    and became ``change-vqa-cv``. The two answer different questions from the same measurement — a
    report versus an answer to what was asked — so the registry now dispatches them separately and
    the endpoint reports both rather than crediting one for the other's work.

    It went from eight to nine when ``satquery-rs-visual-v1`` — the EuroSAT-adapted ResNet-18 — was
    registered as a PREFERRED rung above classical captioning. That is the one non-classical entry,
    and it is the only one whose ``available`` may read false: it is gated on a real checkpoint, the
    learned-models setting and an importable torch, so this asserts its tier and provenance rather
    than pinning its readiness on the host that runs the test.
    """
    response = client.get("/api/models")
    assert response.status_code == 200
    tools = response.json()
    names = [t["name"] for t in tools]

    assert sorted(names) == sorted(
        [
            "landcover-spectral-cv",
            "grounding-spectral-cv",
            "vqa-landcover-cv",
            "captioning-landcover-cv",
            "change-cva-cv",
            "change-vqa-cv",
            "sar-backscatter-cv",
            "fusion-decision-cv",
            "satquery-rs-visual-v1",
        ]
    )
    by_name = {t["name"]: t for t in tools}
    learned = by_name.pop("satquery-rs-visual-v1")

    # Every remaining tool is a classical computer-vision analyser with no learned weights, so none
    # may be advertised as `preferred` or `fallback` (§49), and each is always runnable.
    assert {t["tier"] for t in by_name.values()} == {"classical"}
    assert all(t.get("available") is True for t in by_name.values())
    assert all(t.get("runtime") is None for t in by_name.values())

    # The learned rung earns its tier from a real trained checkpoint, and reports the artefact.
    assert learned["tier"] == "preferred"
    assert learned["runtime"]["model_id"] == "satquery-rs-visual-v1"
    assert learned["runtime"]["mode"] in {"LIVE", "UNAVAILABLE"}
    assert learned["status"] == ("ready" if learned["available"] else "unavailable")
    assert (learned["runtime"]["mode"] == "LIVE") is bool(learned["available"])

    # The contract each tool is gated on is published, not implied (§6).
    for tool in tools:
        contract = tool["contract"]
        assert contract["supported_modes"], tool["name"]
        assert contract["min_images"] >= 1
        assert contract["max_images"] >= contract["min_images"]


def test_evaluation_endpoint(client):
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "benchmarks" in data or "timestamp" in data


def test_live_single_optical_pipeline_unseen_geotiff(client, tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    opt_file = tmp_path / "unseen_optical.tif"
    data = np.zeros((4, 60, 60), dtype=np.uint16)
    data[0] = 500  # Blue
    data[1] = 700  # Green
    data[2] = 450  # Red
    data[3] = 4500 # NIR (high vegetation)
    transform = from_origin(500000, 3000000, 10, 10)

    with rasterio.open(opt_file, "w", driver="GTiff", height=60, width=60, count=4, dtype=np.uint16, crs="EPSG:32643", transform=transform) as dst:
        dst.write(data)

    with open(opt_file, "rb") as f:
        up = client.post("/api/upload", files={"file": ("unseen_optical.tif", f, "image/tiff")})
    assert up.status_code in (200, 201)
    img_id = up.json()["image_id"]

    res = client.post("/api/analyze", json={
        "query": "Describe the land-cover and major objects visible in this image.",
        "image_ids": [img_id],
    })
    assert res.status_code == 200
    ans = res.json()
    assert ans["mode"] == "single_optical"
    assert "confidence_level" in ans
    assert len(ans["execution_trace"]["steps"]) >= 3


def test_live_grounding_pipeline_unseen_geotiff(client, tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    opt_file = tmp_path / "unseen_grounding.tif"
    data = np.zeros((4, 60, 60), dtype=np.uint16)
    data[0] = 500
    data[1] = 700
    data[2] = 450
    data[3] = 4500
    # Water pond in north
    data[0, :20, :] = 900
    data[1, :20, :] = 800
    data[2, :20, :] = 500
    data[3, :20, :] = 300
    transform = from_origin(500000, 3000000, 10, 10)

    with rasterio.open(opt_file, "w", driver="GTiff", height=60, width=60, count=4, dtype=np.uint16, crs="EPSG:32643", transform=transform) as dst:
        dst.write(data)

    with open(opt_file, "rb") as f:
        up = client.post("/api/upload", files={"file": ("unseen_grounding.tif", f, "image/tiff")})
    assert up.status_code in (200, 201)
    img_id = up.json()["image_id"]

    res = client.post("/api/analyze", json={
        "query": "Where is the water in the north?",
        "image_ids": [img_id],
    })
    assert res.status_code == 200
    ans = res.json()
    assert ans["mode"] == "single_optical"
    assert ans["task"] == "grounding"


def test_live_bitemporal_pipeline_unseen_geotiff(client, tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    t1_file = tmp_path / "unseen_t1.tif"
    t2_file = tmp_path / "unseen_t2.tif"
    data_t1 = np.full((4, 60, 60), 1000, dtype=np.uint16)
    data_t1[3] = 4000 # High NIR (veg)
    data_t2 = data_t1.copy()
    data_t2[3, 20:40, 20:40] = 800 # Vegetation lost in center
    transform = from_origin(500000, 3000000, 10, 10)

    with rasterio.open(t1_file, "w", driver="GTiff", height=60, width=60, count=4, dtype=np.uint16, crs="EPSG:32643", transform=transform) as dst:
        dst.write(data_t1)
    with rasterio.open(t2_file, "w", driver="GTiff", height=60, width=60, count=4, dtype=np.uint16, crs="EPSG:32643", transform=transform) as dst:
        dst.write(data_t2)

    with open(t1_file, "rb") as f1, open(t2_file, "rb") as f2:
        up1 = client.post("/api/upload", files={"file": ("unseen_t1.tif", f1, "image/tiff")})
        up2 = client.post("/api/upload", files={"file": ("unseen_t2.tif", f2, "image/tiff")})

    assert up1.status_code in (200, 201)
    assert up2.status_code in (200, 201)

    res = client.post("/api/analyze", json={
        "query": "What changed between these two dates?",
        "image_ids": [up1.json()["image_id"], up2.json()["image_id"]],
    })
    assert res.status_code == 200
    ans = res.json()
    assert ans["mode"] == "bitemporal_pair"
    assert ans["task"] in ("change_detection", "change_vqa")


def test_live_optical_sar_fusion_pipeline_unseen_geotiff(client, tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    opt_file = tmp_path / "unseen_opt.tif"
    sar_file = tmp_path / "unseen_sar.tif"
    transform = from_origin(500000, 3000000, 10, 10)

    opt_data = np.full((4, 60, 60), 1200, dtype=np.uint16)
    sar_data = np.full((1, 60, 60), -12.0, dtype=np.float32)
    sar_data[0, 10:30, 10:30] = 3.0 # High double bounce

    with rasterio.open(opt_file, "w", driver="GTiff", height=60, width=60, count=4, dtype=np.uint16, crs="EPSG:32643", transform=transform) as dst:
        dst.write(opt_data)
    with rasterio.open(sar_file, "w", driver="GTiff", height=60, width=60, count=1, dtype=np.float32, crs="EPSG:32643", transform=transform) as dst:
        dst.write(sar_data)

    with open(opt_file, "rb") as f1, open(sar_file, "rb") as f2:
        up1 = client.post("/api/upload", files={"file": ("unseen_opt.tif", f1, "image/tiff")})
        up2 = client.post("/api/upload", files={"file": ("unseen_sar.tif", f2, "image/tiff")})

    assert up1.status_code in (200, 201)
    assert up2.status_code in (200, 201)

    res = client.post("/api/analyze", json={
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "image_ids": [up1.json()["image_id"], up2.json()["image_id"]],
    })
    assert res.status_code == 200
    ans = res.json()
    assert ans["mode"] == "optical_sar_pair"
    assert ans["task"] == "optical_sar_analysis"

    # Verify /api/history retains canonical modes
    hist = client.get("/api/history").json()
    assert hist["total"] > 0
    modes = [it["mode"] for it in hist["items"]]
    assert "optical_sar_pair" in modes
    assert "bitemporal_pair" in modes


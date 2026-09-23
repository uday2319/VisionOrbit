"""Tests for the real EuroSAT → ResNet-18 adaptation pipeline in ``ml/`` (brief §18, §49).

Eight areas the brief asks to be covered live here or in ``tests/agent/test_scene_tool.py``: dataset
loading, preprocessing, checkpoint saving, checkpoint loading, inference, invalid-image handling
(this file), and model registry plus live backend integration (the tool test).

Two deliberate properties:

* **Nothing is mocked.** Where a test needs images it reads the actual EuroSAT JPEGs recorded by
  ``python -m ml.adaptation.prepare_data``; where it needs a checkpoint it either uses the trained one
  or trains a fresh miniature one on real images. A mocked dataset would prove the test harness works
  and nothing about the pipeline.
* **Absent data skips, it does not fail.** A fresh clone has neither the dataset nor the checkpoint,
  and that is a supported state — the backend falls back to the deterministic analysers, which
  ``test_scene_tool.py`` asserts. Skips here mean "not prepared", never "broken".

The split manifests record repo-root-relative paths, because the ``ml`` CLIs are documented to run
from the repo root. These tests therefore chdir there rather than rewriting the manifest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
if str(REPO_ROOT) not in sys.path:  # the ml package is not importable from backend/ by default
    sys.path.insert(0, str(REPO_ROOT))

SUBSET_DIR = REPO_ROOT / "ml" / "datasets" / "processed" / "eurosat_subset"
CHECKPOINT_DIR = REPO_ROOT / "ml" / "checkpoints" / "satquery-rs-visual-v1"


def _torch_available() -> bool:
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("torchvision") is not None


requires_torch = pytest.mark.skipif(not _torch_available(), reason="torch/torchvision not installed")
requires_dataset = pytest.mark.skipif(
    not (SUBSET_DIR / "manifest.json").is_file(),
    reason="EuroSAT subset not prepared; run `python -m ml.adaptation.prepare_data`",
)
requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.pt").is_file(),
    reason="no trained checkpoint; run `python -m ml.adaptation.train`",
)

@pytest.fixture(autouse=True)
def _at_repo_root(monkeypatch):
    """Split manifests store repo-root-relative image paths; resolve them the documented way."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not (SUBSET_DIR / "manifest.json").is_file():
        pytest.skip("EuroSAT subset not prepared")
    return json.loads((SUBSET_DIR / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------


@requires_dataset
class TestDatasetLoading:
    def test_manifest_records_a_real_open_dataset_not_a_synthetic_one(self, manifest):
        """The prohibition that matters most: the recorded provenance must be the real thing."""
        assert manifest["dataset"] == "EuroSAT (RGB)"
        assert manifest["is_synthetic"] is False
        assert "eurosat" in manifest["source_url"].lower()
        assert manifest["licence"] == "MIT"
        assert manifest["citation"]
        # Nothing in this pipeline touched BigEarthNet, so nothing may name it.
        assert "bigearthnet" not in json.dumps(manifest).lower()

    def test_subset_sizes_are_inside_the_brief_s_bounds(self, manifest):
        splits = manifest["splits"]
        assert 500 <= splits["train"] <= 2000
        assert 100 <= splits["val"] <= 400
        assert 100 <= splits["test"] <= 400
        assert manifest["seed"] == 1337  # the fixed seed the subset is reproducible from

    def test_ten_classes_are_recorded_in_a_stable_order(self, manifest):
        from ml.datasets.eurosat_adapter import EUROSAT_CLASSES

        assert manifest["num_classes"] == 10
        assert tuple(manifest["class_names"]) == EUROSAT_CLASSES

    def test_every_split_loads_with_labels_in_range_and_files_on_disk(self, manifest):
        from ml.datasets.eurosat_adapter import load_split

        for split, expected in manifest["splits"].items():
            samples = load_split(split, SUBSET_DIR)
            assert len(samples) == expected
            assert {s.label for s in samples} == set(range(10))
            for sample in samples:
                assert sample.class_name == manifest["class_names"][sample.label]
            # Spot-check existence rather than stat()ing 2800 files under OneDrive sync.
            for sample in samples[:20]:
                assert Path(sample.path).is_file(), sample.path

    def test_splits_are_disjoint_so_the_test_set_is_genuinely_held_out(self):
        """The accuracy claim is void if a test image was trained on."""
        from ml.datasets.eurosat_adapter import load_split

        train = {s.path for s in load_split("train", SUBSET_DIR)}
        val = {s.path for s in load_split("val", SUBSET_DIR)}
        test = {s.path for s in load_split("test", SUBSET_DIR)}
        assert not train & val
        assert not train & test
        assert not val & test

    def test_class_balance_is_even_across_every_split(self, manifest):
        from collections import Counter

        from ml.datasets.eurosat_adapter import load_split

        for split, per_class in manifest["per_class_per_split"].items():
            counts = Counter(s.label for s in load_split(split, SUBSET_DIR))
            assert set(counts.values()) == {per_class}


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------


@requires_torch
@requires_dataset
class TestPreprocessing:
    @pytest.fixture(scope="class")
    def sample_path(self) -> str:
        from ml.datasets.eurosat_adapter import load_split

        return load_split("test", SUBSET_DIR)[0].path

    def test_a_real_eurosat_tile_loads_as_rgb(self, sample_path):
        from ml.datasets.eurosat_adapter import load_image

        image = load_image(sample_path)
        assert image.mode == "RGB"
        assert image.size == (64, 64)  # EuroSAT tiles are 64x64; the transform resizes them

    def test_eval_transform_produces_the_tensor_the_model_expects(self, sample_path):
        from ml.datasets.eurosat_adapter import INPUT_SIZE, build_transform, load_image

        tensor = build_transform(train=False)(load_image(sample_path))
        assert tuple(tensor.shape) == (3, INPUT_SIZE, INPUT_SIZE)
        assert str(tensor.dtype) == "torch.float32"

    def test_normalization_is_imagenet_statistics_actually_applied(self, sample_path):
        """The pretrained backbone was fitted on ImageNet-normalised input; a raw [0,1] tensor
        would quietly shift the distribution and cost accuracy for no visible reason."""
        from ml.datasets.eurosat_adapter import (
            IMAGENET_MEAN,
            IMAGENET_STD,
            build_transform,
            load_image,
        )

        image = load_image(sample_path)
        normalised = build_transform(train=False)(image)
        # Invert the normalisation and recover a valid [0,1] image.
        for c in range(3):
            channel = normalised[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
            assert float(channel.min()) >= -1e-4
            assert float(channel.max()) <= 1.0 + 1e-4
        # And it is not merely ToTensor: at least one channel must leave [0,1].
        assert float(normalised.min()) < 0.0

    def test_train_transform_augments_while_eval_transform_is_deterministic(self, sample_path):
        import torch
        from ml.datasets.eurosat_adapter import build_transform, load_image

        image = load_image(sample_path)
        evaluation = build_transform(train=False)
        assert torch.equal(evaluation(image), evaluation(image))

        torch.manual_seed(0)
        training = build_transform(train=True)
        draws = [training(image) for _ in range(12)]
        assert any(not torch.equal(draws[0], d) for d in draws), "flips never fired"

    def test_dataset_yields_tensor_label_pairs(self):
        from ml.datasets.eurosat_adapter import build_dataset

        ds = build_dataset("test", SUBSET_DIR)
        tensor, label = ds[0]
        assert tuple(tensor.shape) == (3, 224, 224)
        assert 0 <= int(label) <= 9
        assert len(ds) == 400


# ---------------------------------------------------------------------------
# 3. Invalid image / missing artefact handling
# ---------------------------------------------------------------------------


class TestInvalidInputHandling:
    def test_missing_image_raises_file_not_found_naming_the_path(self, tmp_path):
        from ml.datasets.eurosat_adapter import load_image

        missing = tmp_path / "absent.jpg"
        with pytest.raises(FileNotFoundError, match="Image not found"):
            load_image(missing)

    def test_a_non_image_file_raises_a_clear_value_error(self, tmp_path):
        from ml.datasets.eurosat_adapter import load_image

        bogus = tmp_path / "not-an-image.jpg"
        bogus.write_bytes(b"this is text, not a JPEG")
        with pytest.raises(ValueError, match="Not a readable image"):
            load_image(bogus)

    def test_a_truncated_image_is_rejected_rather_than_silently_padded(self, tmp_path):
        from ml.datasets.eurosat_adapter import load_image

        truncated = tmp_path / "truncated.png"
        truncated.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        with pytest.raises(ValueError):
            load_image(truncated)

    def test_an_unprepared_subset_directory_says_how_to_prepare_it(self, tmp_path):
        from ml.datasets.eurosat_adapter import DatasetNotPreparedError, load_split

        with pytest.raises(DatasetNotPreparedError, match="prepare_data"):
            load_split("test", tmp_path)

    def test_an_unknown_split_name_is_a_value_error(self):
        from ml.datasets.eurosat_adapter import load_split

        with pytest.raises(ValueError, match="Unknown split"):
            load_split("holdout", SUBSET_DIR)

    @requires_torch
    def test_inference_on_a_missing_checkpoint_raises_checkpoint_not_found(self, tmp_path):
        """A missing checkpoint must be an explicit outage — never an untrained random model."""
        from ml.adaptation.inference import AdaptedModelInference, CheckpointNotFoundError

        with pytest.raises(CheckpointNotFoundError):
            AdaptedModelInference(tmp_path / "nope")


# ---------------------------------------------------------------------------
# 4. The base model really is a pretrained torchvision ResNet-18
# ---------------------------------------------------------------------------


@requires_torch
class TestBaseModel:
    def test_it_is_torchvision_resnet18_with_a_fresh_ten_class_head(self):
        from ml.adaptation.train import build_model
        from torchvision.models.resnet import ResNet

        model = build_model(10)
        assert isinstance(model, ResNet)
        assert model.fc.out_features == 10
        assert model.fc.in_features == 512

    def test_the_backbone_carries_the_actual_imagenet_weights(self):
        """The prohibition on calling a hand-rolled Conv2D stack "ResNet-18", tested directly.

        Compares the built model's first convolution against the weights torchvision itself
        distributes for ``IMAGENET1K_V1``. Equal tensors mean the pretrained checkpoint was loaded;
        a bespoke or randomly initialised network could not match it.
        """
        import torch
        from ml.adaptation.train import build_model
        from torchvision.models import ResNet18_Weights, resnet18

        reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        built = build_model(10)
        assert torch.equal(built.conv1.weight, reference.conv1.weight)
        assert torch.equal(built.layer4[1].conv2.weight, reference.layer4[1].conv2.weight)

    def test_only_the_classification_head_is_trainable(self):
        from ml.adaptation.train import build_model, count_parameters

        model = build_model(10)
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable_names == {"fc.weight", "fc.bias"}
        trainable, total = count_parameters(model)
        assert trainable == 512 * 10 + 10 == 5130
        assert total == 11_181_642

    def test_batchnorm_stays_in_eval_mode_during_training(self):
        """Freezing ``requires_grad`` does not freeze BatchNorm's running statistics; a training-mode
        backbone would drift the very features the probe was fitted on."""
        import torch.nn as nn
        from ml.adaptation.train import build_model, set_backbone_eval

        model = build_model(10)
        set_backbone_eval(model)
        assert model.fc.training is True
        assert all(
            not m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
        )


# ---------------------------------------------------------------------------
# 5. Checkpoint saving — a real (miniature) training run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def miniature_run(tmp_path_factory) -> dict:
    """Train one epoch on a small slice of the *real* subset and return (metadata, output_dir).

    Small enough to run inside a test suite, real enough to prove the save path: the images are
    actual EuroSAT tiles, the backbone is the actual pretrained ResNet-18, and the checkpoint written
    is the same artefact ``python -m ml.adaptation.train`` produces. It is never used as the shipped
    checkpoint or as a source of reported accuracy.
    """
    if not _torch_available() or not (SUBSET_DIR / "manifest.json").is_file():
        pytest.skip("torch or the EuroSAT subset is unavailable")

    import os

    os.chdir(REPO_ROOT)  # module-scoped: the autouse chdir fixture is function-scoped
    from ml.adaptation.train import train_adapter

    sizes = {"train": 20, "val": 10, "test": 10}
    subset = tmp_path_factory.mktemp("mini_subset")
    manifest = json.loads((SUBSET_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["splits"] = dict(sizes)
    manifest["per_class_per_split"] = {k: v // 10 for k, v in sizes.items()}
    (subset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for split, size in sizes.items():
        rows = json.loads((SUBSET_DIR / f"{split}.json").read_text(encoding="utf-8"))
        # Take a stride so the slice spans classes rather than landing in one folder.
        stride = max(1, len(rows) // size)
        (subset / f"{split}.json").write_text(
            json.dumps(rows[::stride][:size]), encoding="utf-8"
        )

    output = tmp_path_factory.mktemp("mini_checkpoint")
    metadata = train_adapter(
        subset_dir=subset, output_dir=output, epochs=1, batch_size=8, seed=1337
    )
    return {"metadata": metadata, "dir": output, "sizes": sizes}


@pytest.mark.slow
class TestCheckpointSaving:
    def test_the_run_writes_weights_and_metadata(self, miniature_run):
        directory = miniature_run["dir"]
        assert (directory / "model.pt").is_file()
        assert (directory / "metadata.json").is_file()
        assert (directory / "model.pt").stat().st_size > 1_000_000  # real ResNet-18 weights

    def test_metadata_records_every_field_the_brief_asks_for(self, miniature_run):
        meta = miniature_run["metadata"]
        for field in (
            "dataset", "dataset_source", "num_train_images", "num_val_images", "num_test_images",
            "base_model", "pretrained_weights", "adaptation_method", "trainable_parameters",
            "total_parameters", "epochs", "learning_rate", "batch_size", "train_loss", "val_loss",
            "timestamp", "checkpoint_file", "history", "training_seconds", "seed",
        ):
            assert field in meta, field
        assert meta["dataset"] == "EuroSAT (RGB)"
        assert meta["base_model"] == "torchvision.models.resnet18"
        assert meta["pretrained_weights"] == "IMAGENET1K_V1"
        assert meta["trainable_parameters"] == 5130
        assert meta["training_completed"] is True
        assert meta["num_train_images"] == miniature_run["sizes"]["train"]

    def test_metadata_reports_no_test_accuracy_until_evaluation_runs(self, miniature_run):
        """An unevaluated checkpoint must not advertise an accuracy it never measured."""
        meta = miniature_run["metadata"]
        assert meta["test_accuracy"] is None
        assert meta["test_evaluation"] is None

    def test_the_saved_payload_describes_itself(self, miniature_run):
        import torch

        payload = torch.load(miniature_run["dir"] / "model.pt", map_location="cpu",
                             weights_only=True)
        assert payload["model_id"] == "satquery-rs-visual-v1"
        assert payload["architecture"] == "resnet18"
        assert payload["pretrained_weights"] == "IMAGENET1K_V1"
        assert len(payload["class_names"]) == 10
        assert payload["state_dict"]["fc.weight"].shape == (10, 512)

    def test_training_actually_updated_the_head_and_nothing_else(self, miniature_run):
        """One epoch on 20 images will not move the needle on accuracy — but it must move the
        weights it claims to train, and only those."""
        import torch
        from ml.adaptation.train import build_model

        payload = torch.load(miniature_run["dir"] / "model.pt", map_location="cpu",
                             weights_only=True)
        fresh = build_model(10).state_dict()
        trained = payload["state_dict"]
        assert not torch.equal(trained["fc.weight"], fresh["fc.weight"])
        assert torch.equal(trained["conv1.weight"], fresh["conv1.weight"])
        assert torch.equal(trained["layer4.1.bn2.running_mean"], fresh["layer4.1.bn2.running_mean"])


# ---------------------------------------------------------------------------
# 6. Checkpoint loading and inference
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFreshCheckpointLoads:
    """The load test the brief asks for: a separate loader, not the training process's own model."""

    def test_a_just_written_checkpoint_loads_and_predicts(self, miniature_run):
        from ml.adaptation.inference import AdaptedModelInference
        from ml.datasets.eurosat_adapter import load_split

        engine = AdaptedModelInference(miniature_run["dir"])
        sample = load_split("test", SUBSET_DIR)[0]
        prediction = engine.predict(sample.path)
        assert prediction.class_name in engine.class_names
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.probabilities) == 10


@requires_torch
@requires_checkpoint
@requires_dataset
class TestShippedCheckpointInference:
    """The artefact the backend actually serves, loaded independently of the training module."""

    @pytest.fixture(scope="class")
    def engine(self):
        from ml.adaptation.inference import AdaptedModelInference

        return AdaptedModelInference(CHECKPOINT_DIR)

    def test_describe_reads_its_provenance_from_the_checkpoint(self, engine):
        described = engine.describe()
        assert described["model_id"] == "satquery-rs-visual-v1"
        assert described["architecture"] == "resnet18"
        assert described["pretrained_weights"] == "IMAGENET1K_V1"
        assert "linear probe" in described["adaptation_method"]
        assert len(engine.class_names) == 10

    def test_predictions_on_unseen_test_images_are_real_distributions(self, engine):
        from ml.datasets.eurosat_adapter import load_split

        samples = load_split("test", SUBSET_DIR)[::40][:10]
        for sample in samples:
            prediction = engine.predict(sample.path)
            assert prediction.class_name == engine.class_names[prediction.label]
            assert prediction.confidence == pytest.approx(
                max(prediction.probabilities.values()), abs=1e-6
            )
            assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    def test_it_is_better_than_chance_on_held_out_images(self, engine):
        """A weak but decisive check that the head learned *something* real: a random 10-class head
        would score about 10%. The reported figure comes from ``evaluate``, not from here."""
        from ml.datasets.eurosat_adapter import load_split

        samples = load_split("test", SUBSET_DIR)[::10][:40]
        correct = sum(engine.predict(s.path).label == s.label for s in samples)
        assert correct / len(samples) > 0.5

    def test_the_same_image_predicts_identically_twice(self, engine):
        """eval() mode and no augmentation at inference: the answer must be reproducible."""
        from ml.datasets.eurosat_adapter import load_split

        path = load_split("test", SUBSET_DIR)[3].path
        first, second = engine.predict(path), engine.predict(path)
        assert first.class_name == second.class_name
        assert first.confidence == pytest.approx(second.confidence, abs=1e-9)


# ---------------------------------------------------------------------------
# 7. Evaluation — measured on the held-out split, labelled honestly
# ---------------------------------------------------------------------------


@requires_torch
class TestConfusionMatrix:
    def test_it_counts_truth_rows_against_predicted_columns(self):
        from ml.adaptation.evaluate import confusion_matrix

        matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 0], num_classes=3)
        assert matrix == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
        assert sum(sum(row) for row in matrix) == 4


@requires_torch
@requires_checkpoint
@requires_dataset
class TestAdaptationEvaluation:
    @pytest.fixture(scope="class")
    def report(self, tmp_path_factory) -> dict:
        """Re-measure on a 100-image slice of the held-out split, into a temp directory.

        A slice keeps the test quick; the number it produces is not the reported figure and is not
        written next to the shipped checkpoint. ``evaluation.json`` in the repo comes from the full
        400-image run of ``python -m ml.adaptation.evaluate``.
        """
        import shutil

        from ml.adaptation.evaluate import evaluate_adapted_model

        scratch = tmp_path_factory.mktemp("eval_checkpoint")
        for name in ("model.pt", "metadata.json"):
            shutil.copy2(CHECKPOINT_DIR / name, scratch / name)
        return evaluate_adapted_model(
            checkpoint_dir=scratch, subset_dir=SUBSET_DIR, split="test", limit=100
        )

    def test_it_is_labelled_as_an_adaptation_evaluation_not_an_sih_benchmark(self, report):
        """§49 and the user's explicit instruction: this number is not the prescribed benchmark."""
        from ml.adaptation.evaluate import EVALUATION_LABEL

        assert report["evaluation"] == EVALUATION_LABEL == "Remote-sensing adaptation evaluation"
        assert report["is_sih_benchmark"] is False
        assert "not the SIH prescribed benchmark" in report["note"]

    def test_accuracy_is_derived_from_the_counted_samples(self, report):
        assert report["num_test_samples"] == 100
        assert report["num_correct"] <= report["num_test_samples"]
        assert report["test_accuracy"] == pytest.approx(
            report["num_correct"] / report["num_test_samples"], abs=1e-9
        )
        assert report["test_accuracy"] > 0.5  # far above the 10% a random head would score

    def test_the_confusion_matrix_accounts_for_every_sample(self, report):
        matrix = report["confusion_matrix"]
        assert len(matrix) == 10 and all(len(row) == 10 for row in matrix)
        assert sum(sum(row) for row in matrix) == report["num_test_samples"]
        diagonal = sum(matrix[i][i] for i in range(10))
        assert diagonal == report["num_correct"]
        assert "row" in report["confusion_matrix_orientation"].lower()

    def test_it_records_which_artefact_and_which_split_produced_the_number(self, report):
        assert report["model_id"] == "satquery-rs-visual-v1"
        assert report["split"] == "test"
        assert report["dataset"] == "EuroSAT (RGB)"
        assert report["timestamp"]

    def test_it_backfills_the_measured_accuracy_into_the_checkpoint_metadata(
        self, report, tmp_path_factory
    ):
        """The backend reads ``test_accuracy`` from the checkpoint's sidecars, so evaluation has to
        write it there — otherwise the live component could never state its measured accuracy."""
        import shutil

        from ml.adaptation.evaluate import evaluate_adapted_model

        scratch = tmp_path_factory.mktemp("backfill_checkpoint")
        for name in ("model.pt", "metadata.json"):
            shutil.copy2(CHECKPOINT_DIR / name, scratch / name)
        evaluate_adapted_model(
            checkpoint_dir=scratch, subset_dir=SUBSET_DIR, split="test", limit=20
        )
        meta = json.loads((scratch / "metadata.json").read_text(encoding="utf-8"))
        assert meta["test_accuracy"] is not None
        assert (scratch / "evaluation.json").is_file()


@requires_checkpoint
class TestShippedEvaluationArtefact:
    """The numbers the docs and the UI quote must be the ones on disk, measured on 400 images."""

    def test_the_recorded_evaluation_is_a_full_held_out_run(self):
        path = CHECKPOINT_DIR / "evaluation.json"
        if not path.is_file():
            pytest.skip("evaluation.json absent; run `python -m ml.adaptation.evaluate`")
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["is_sih_benchmark"] is False
        assert report["num_test_samples"] == 400
        assert report["test_accuracy"] == pytest.approx(report["num_correct"] / 400, abs=1e-9)
        assert "bigearthnet" not in json.dumps(report).lower()

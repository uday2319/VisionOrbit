"""Real remote-sensing adaptation: EuroSAT (RGB) -> pretrained ResNet-18 linear probe.

The four modules form one pipeline, each runnable on its own::

    python -m ml.adaptation.prepare_data   # download EuroSAT, record a seeded subset
    python -m ml.adaptation.train          # adapt the pretrained ResNet-18, save a checkpoint
    python -m ml.adaptation.evaluate       # measure accuracy on the held-out test split
    python -m ml.adaptation.inference      # load the checkpoint and predict on unseen images

Attribute access is resolved lazily. Importing the submodules here eagerly would make
``python -m ml.adaptation.train`` warn that the module was already in ``sys.modules`` before being
executed as ``__main__`` — and each module is a CLI entry point, so that path is the normal one.
"""
from typing import Any

_EXPORTS: dict[str, str] = {
    "ADAPTATION_METHOD": "train",
    "BASE_MODEL": "train",
    "DEFAULT_CHECKPOINT_DIR": "train",
    "MODEL_ID": "train",
    "PRETRAINED_WEIGHTS": "train",
    "build_model": "train",
    "train_adapter": "train",
    "EVALUATION_LABEL": "evaluate",
    "evaluate_adapted_model": "evaluate",
    "AdaptedModelInference": "inference",
    "CheckpointNotFoundError": "inference",
    "ScenePrediction": "inference",
    "prepare_eurosat_subset": "prepare_data",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module_name}", __name__), name)


def __dir__() -> list[str]:
    return list(__all__)

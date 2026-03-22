from pathlib import Path

from essay_labeler.config import load_config


def test_load_config_supports_overrides() -> None:
    config = load_config(
        Path("configs/base.yaml"),
        overrides=["training.folds=3", "model.backbone_name=distilroberta-base"],
    )
    assert config.training.folds == 3
    assert config.model.backbone_name == "distilroberta-base"

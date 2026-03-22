__all__ = [
    "ExperimentConfig",
    "TokenClassificationModel",
    "SpanThresholds",
    "build_training_frame",
    "build_tokenizer",
    "load_config",
    "load_essays",
    "post_process_predictions",
    "predict_from_logits",
]


def __getattr__(name: str):
    if name in {"ExperimentConfig", "load_config"}:
        from essay_labeler.config import ExperimentConfig, load_config

        return {"ExperimentConfig": ExperimentConfig, "load_config": load_config}[name]
    if name in {"build_training_frame", "load_essays"}:
        from essay_labeler.data import build_training_frame, load_essays

        return {"build_training_frame": build_training_frame, "load_essays": load_essays}[name]
    if name in {"TokenClassificationModel", "build_tokenizer"}:
        from essay_labeler.modeling import TokenClassificationModel, build_tokenizer

        return {"TokenClassificationModel": TokenClassificationModel, "build_tokenizer": build_tokenizer}[name]
    if name in {"SpanThresholds", "post_process_predictions"}:
        from essay_labeler.postprocess import SpanThresholds, post_process_predictions

        return {
            "SpanThresholds": SpanThresholds,
            "post_process_predictions": post_process_predictions,
        }[name]
    if name == "predict_from_logits":
        from essay_labeler.pipeline import predict_from_logits

        return predict_from_logits
    raise AttributeError(name)

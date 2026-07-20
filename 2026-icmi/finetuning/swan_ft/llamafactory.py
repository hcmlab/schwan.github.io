from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import yaml

from .config import ModelSpec, RunSpec
from .paths import resolver_for


CURRENT_SWAN_MODEL_FAMILY: str | None = None


DEFAULT_EARLY_STOPPING_POLICY = {
    "do_eval": True,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "early_stopping_steps": 5,
    "save_total_limit": 2,
}


def _wandb_enabled() -> bool:
    value = os.environ.get("SWAN_WANDB_ENABLED", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def _maybe_apply_wandb(config: dict[str, Any], run_spec: RunSpec, fold_id: int, predict: bool = False) -> dict[str, Any]:
    if not _wandb_enabled():
        return config

    resolver = resolver_for(run_spec)
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    run_kind = "predict" if predict else "train"
    run_name = f"{model_spec.output_name}_{resolver.variant_id(dataset_spec, run_spec.options)}_{run_kind}_fold_{fold_id}"

    config["report_to"] = "wandb"
    config["run_name"] = run_name
    config.setdefault("logging_steps", 10)

    project = os.environ.get("SWAN_WANDB_PROJECT")
    entity = os.environ.get("SWAN_WANDB_ENTITY")
    group = os.environ.get("SWAN_WANDB_GROUP") or f"{model_spec.output_name}_{resolver.variant_id(dataset_spec, run_spec.options)}"
    tags = os.environ.get("SWAN_WANDB_TAGS")

    if project:
        os.environ.setdefault("WANDB_PROJECT", project)
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)
    if group:
        os.environ.setdefault("WANDB_RUN_GROUP", group)
    if tags:
        os.environ.setdefault("WANDB_TAGS", tags)

    return config


def _prediction_metadata_mode(model_family: str | None) -> str:
    if model_family == "vision_glm":
        return "strip_all_metadata"
    if model_family in {"vision_qwen", "vision_internvl", "vision_minicpm"}:
        return "strip_video_metadata"
    return "preserve"



def _maybe_patch_prediction_trainer() -> None:
    try:
        from llamafactory.train.sft import trainer as sft_trainer
    except Exception:
        return

    original = sft_trainer.CustomSeq2SeqTrainer.prediction_step
    if getattr(original, "_swan_prediction_family_patch", False):
        return

    def patched(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        metadata_mode = _prediction_metadata_mode(CURRENT_SWAN_MODEL_FAMILY)
        if metadata_mode == "strip_all_metadata":
            for key in list(inputs.keys()):
                if key.endswith("_metadata"):
                    inputs.pop(key, None)
        elif metadata_mode == "strip_video_metadata":
            inputs.pop("video_metadata", None)
        return original(self, model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs)

    patched._swan_prediction_family_patch = True
    sft_trainer.CustomSeq2SeqTrainer.prediction_step = patched


def _maybe_patch_glm46v_video_processor() -> None:
    if CURRENT_SWAN_MODEL_FAMILY != "vision_glm":
        return
    try:
        import transformers
        import transformers.video_processing_utils as video_processing_utils
    except Exception:
        return

    def _wrap_call(cls) -> None:
        original = getattr(cls, "__call__", None)
        if original is None or getattr(original, "_swan_glm46v_patch", False):
            return

        def patched(self, *args, **kwargs):
            if self.__class__.__name__ == "Glm46VVideoProcessor":
                kwargs = dict(kwargs)
                if kwargs.get("images", object()) is None:
                    kwargs.pop("images", None)
                video_metadata = kwargs.get("video_metadata")
                if isinstance(video_metadata, list):
                    normalized = []
                    for item in video_metadata:
                        if isinstance(item, dict):
                            item = dict(item)
                            total_num_frames = item.get("total_num_frames")
                            total_frames = item.pop("total_frames", None)
                            if total_num_frames is None and total_frames is not None:
                                item["total_num_frames"] = total_frames
                        normalized.append(item)
                    kwargs["video_metadata"] = normalized
            return original(self, *args, **kwargs)

        patched._swan_glm46v_patch = True
        cls.__call__ = patched

    for cls_name in ("Glm46VVideoProcessor", "Glm4VVideoProcessor"):
        cls = getattr(transformers, cls_name, None)
        if cls is not None:
            _wrap_call(cls)

    for base_name in ("BaseVideoProcessor", "VideoProcessingMixin", "VideoProcessorMixin"):
        cls = getattr(video_processing_utils, base_name, None)
        if cls is not None:
            _wrap_call(cls)


def _apply_option_overrides(config: dict[str, Any], run_spec: RunSpec) -> dict[str, Any]:
    options = run_spec.options
    if options.quantization is False:
        config.pop("quantization_bit", None)
        config.pop("quantization_method", None)
    if options.per_device_train_batch_size is not None:
        config["per_device_train_batch_size"] = options.per_device_train_batch_size
    if options.per_device_eval_batch_size is not None:
        config["per_device_eval_batch_size"] = options.per_device_eval_batch_size
    if options.gradient_accumulation_steps is not None:
        config["gradient_accumulation_steps"] = options.gradient_accumulation_steps
    if options.num_train_epochs is not None:
        config["num_train_epochs"] = options.num_train_epochs
    if options.eval_steps is not None:
        config["eval_steps"] = options.eval_steps
    if options.save_steps is not None:
        config["save_steps"] = options.save_steps
    if options.early_stopping_steps is not None:
        config["early_stopping_steps"] = options.early_stopping_steps
    if options.load_best_model_at_end is not None:
        config["load_best_model_at_end"] = options.load_best_model_at_end
    if config.get("load_best_model_at_end"):
        strategy = config.get("eval_strategy", config.get("evaluation_strategy", "steps"))
        config["eval_strategy"] = strategy
        config["save_strategy"] = strategy
        if strategy == "steps":
            config["save_steps"] = config.get("save_steps", config.get("eval_steps", 100))
            config["eval_steps"] = config.get("eval_steps", config.get("save_steps", 100))
    return config


def _base_config(model_spec: ModelSpec) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_name_or_path": model_spec.base_model,
        "template": model_spec.template,
        "trust_remote_code": model_spec.trust_remote_code,
        "swan_model_family": model_spec.family,
    }
    config.update(model_spec.extra_config)
    return config


def render_train_config(run_spec: RunSpec, fold_id: int) -> Path:
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    resolver = resolver_for(run_spec)
    config = _base_config(model_spec)
    config.update(DEFAULT_EARLY_STOPPING_POLICY)
    config.update(model_spec.train_defaults)
    config.update(
        {
            "stage": "sft",
            "do_train": True,
            "dataset": f"{resolver.variant_id(dataset_spec, run_spec.options)}_fold_{fold_id}_train",
            "eval_dataset": f"{resolver.variant_id(dataset_spec, run_spec.options)}_fold_{fold_id}_val",
            "dataset_dir": str(resolver.variant_root(dataset_spec, run_spec.options)),
            "tokenized_path": str(resolver.tokenized_root(model_spec, dataset_spec, fold_id, run_spec.options)),
            "output_dir": str(resolver.fold_output_dir(model_spec, dataset_spec, fold_id, run_spec.options)),
        }
    )
    config = _apply_option_overrides(config, run_spec)
    config = _maybe_apply_wandb(config, run_spec, fold_id, predict=False)
    output_path = resolver.rendered_train_config(model_spec, dataset_spec, fold_id, run_spec.options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def render_predict_config(run_spec: RunSpec, fold_id: int) -> Path:
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    resolver = resolver_for(run_spec)
    config = _base_config(model_spec)
    config.update(model_spec.predict_defaults)
    config.update(
        {
            "stage": "sft",
            "finetuning_type": model_spec.train_defaults.get("finetuning_type", "lora"),
            "adapter_name_or_path": str(resolver.fold_output_dir(model_spec, dataset_spec, fold_id, run_spec.options)),
            "do_train": False,
            "do_predict": True,
            "predict_with_generate": True,
            "eval_dataset": f"{resolver.variant_id(dataset_spec, run_spec.options)}_fold_{fold_id}_val",
            "dataset_dir": str(resolver.variant_root(dataset_spec, run_spec.options)),
            "output_dir": str(resolver.fold_prediction_dir(model_spec, dataset_spec, fold_id, run_spec.options)),
        }
    )
    config = _apply_option_overrides(config, run_spec)
    config = _maybe_apply_wandb(config, run_spec, fold_id, predict=True)
    output_path = resolver.rendered_predict_config(model_spec, dataset_spec, fold_id, run_spec.options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def render_final_train_config(run_spec: RunSpec) -> Path:
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    resolver = resolver_for(run_spec)
    config = _base_config(model_spec)
    config.update(model_spec.train_defaults)
    config.update(
        {
            "stage": "sft",
            "do_train": True,
            "do_eval": False,
            "dataset": f"{resolver.variant_id(dataset_spec, run_spec.options)}_train_dev",
            "dataset_dir": str(resolver.variant_root(dataset_spec, run_spec.options)),
            "tokenized_path": str(resolver.final_tokenized_root(model_spec, dataset_spec, run_spec.options)),
            "output_dir": str(resolver.final_output_dir(model_spec, dataset_spec, run_spec.options)),
            "load_best_model_at_end": False,
        }
    )
    for key in (
        "eval_dataset",
        "eval_strategy",
        "evaluation_strategy",
        "eval_steps",
        "metric_for_best_model",
        "greater_is_better",
        "early_stopping_steps",
    ):
        config.pop(key, None)
    config = _apply_option_overrides(config, run_spec)
    config["do_eval"] = False
    config["load_best_model_at_end"] = False
    output_path = resolver.rendered_final_train_config(model_spec, dataset_spec, run_spec.options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def render_test_predict_config(run_spec: RunSpec) -> Path:
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    resolver = resolver_for(run_spec)
    config = _base_config(model_spec)
    config.update(model_spec.predict_defaults)
    config.update(
        {
            "stage": "sft",
            "finetuning_type": model_spec.train_defaults.get("finetuning_type", "lora"),
            "adapter_name_or_path": str(resolver.final_output_dir(model_spec, dataset_spec, run_spec.options)),
            "do_train": False,
            "do_predict": True,
            "predict_with_generate": True,
            "eval_dataset": f"{resolver.variant_id(dataset_spec, run_spec.options)}_test",
            "dataset_dir": str(resolver.variant_root(dataset_spec, run_spec.options)),
            "output_dir": str(resolver.final_prediction_dir(model_spec, dataset_spec, run_spec.options)),
        }
    )
    config = _apply_option_overrides(config, run_spec)
    config = _maybe_apply_wandb(config, run_spec, fold_id=999, predict=True)
    output_path = resolver.rendered_test_predict_config(model_spec, dataset_spec, run_spec.options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def write_run_manifest(run_spec: RunSpec) -> Path:
    resolver = resolver_for(run_spec)
    model_spec = run_spec.model_spec
    dataset_spec = run_spec.dataset_spec
    payload = {
        "profile_id": run_spec.profile.id,
        "dataset_spec_id": resolver.variant_id(dataset_spec, run_spec.options),
        "model_id": model_spec.id,
        "folds": list(run_spec.options.folds),
        "role_mode": dataset_spec.effective_role_mode(run_spec.options),
        "excluded_labels": list(dataset_spec.effective_excluded_labels(run_spec.options)),
        "model_family": model_spec.family,
        "paths": {
            "data_root": str(run_spec.profile.data_root),
            "variant_root": str(resolver.variant_root(dataset_spec, run_spec.options)),
            "model_root": str(resolver.model_root(model_spec, dataset_spec, run_spec.options)),
        },
    }
    output_path = resolver.run_manifest(model_spec, dataset_spec, run_spec.options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_llamafactory(config_path: Path, predict: bool = False) -> None:
    global CURRENT_SWAN_MODEL_FAMILY
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    CURRENT_SWAN_MODEL_FAMILY = config.pop("swan_model_family", None)
    if predict:
        _maybe_patch_prediction_trainer()
    _maybe_patch_glm46v_video_processor()
    from llamafactory.train.tuner import run_exp

    try:
        run_exp(args=config)
    except Exception:
        traceback.print_exc()
        raise

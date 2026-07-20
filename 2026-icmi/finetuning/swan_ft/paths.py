from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetSpec, ModelSpec, RunOptions, RunSpec, RuntimeProfile


@dataclass(frozen=True)
class PathResolver:
    profile: RuntimeProfile

    def _profile_env_path(self, env_name: str) -> Path | None:
        profile_token = self.profile.id.upper().replace("-", "_")
        value = os.environ.get(f"SWAN_{profile_token}_{env_name}") or os.environ.get(f"SWAN_{env_name}")
        if not value:
            return None
        return Path(os.path.expanduser(os.path.expandvars(value)))

    def data_root(self) -> Path:
        return self.profile.data_root

    def session_root(self) -> Path | None:
        return self.profile.session_root

    def output_root(self) -> Path:
        return self.profile.output_root

    def variants_root(self) -> Path:
        configured = getattr(self.profile, "variants_root", None)
        env_path = self._profile_env_path("VARIANTS_ROOT")
        return configured or env_path or (self.data_root() / "variants")

    def logs_root(self) -> Path:
        return self.profile.logs_root

    def temp_root(self) -> Path:
        return self.profile.temp_root

    def cache_root(self) -> Path | None:
        return self.profile.cache_root

    def base_dataset_json(self, dataset_spec: DatasetSpec) -> Path:
        return self.data_root() / dataset_spec.base_dataset_file

    def base_dataset_info(self, dataset_spec: DatasetSpec) -> Path:
        return self.data_root() / dataset_spec.base_dataset_info_file

    def folds_json(self, dataset_spec: DatasetSpec) -> Path:
        return self.data_root() / dataset_spec.base_folds_file

    def clips_root(self, dataset_spec: DatasetSpec) -> Path:
        return self.data_root() / dataset_spec.clip_subdir

    def variant_id(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> str:
        options = options or RunOptions()
        return dataset_spec.variant_id(options)

    def variant_root(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variants_root() / self.variant_id(dataset_spec, options)

    def variant_dataset_json(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "dataset.json"

    def variant_dataset_info(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "dataset_info.json"

    def variant_fold_json(self, dataset_spec: DatasetSpec, fold_id: int, split: str, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / f"fold_{fold_id}_{split}.json"

    def variant_train_dev_json(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "train_dev.json"

    def variant_test_json(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "test.json"

    def audio_root(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / dataset_spec.audio_subdir

    def variant_labels_path(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "labels.json"

    def variant_manifest_path(self, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "dataset_manifest.json"

    def model_root(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.output_root() / model_spec.output_name / self.variant_id(dataset_spec, options)

    def fold_output_dir(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, fold_id: int, options: RunOptions | None = None) -> Path:
        return self.model_root(model_spec, dataset_spec, options) / f"fold_{fold_id}"

    def final_output_dir(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.model_root(model_spec, dataset_spec, options) / "final_model"

    def fold_prediction_dir(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, fold_id: int, options: RunOptions | None = None) -> Path:
        return self.fold_output_dir(model_spec, dataset_spec, fold_id, options) / "predictions"

    def final_prediction_dir(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.final_output_dir(model_spec, dataset_spec, options) / "predictions"

    def fold_eval_report(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, fold_id: int, options: RunOptions | None = None) -> Path:
        return self.model_root(model_spec, dataset_spec, options) / f"fold_{fold_id}_eval_report.json"

    def final_test_eval_report(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.model_root(model_spec, dataset_spec, options) / f"{model_spec.output_name}_{self.variant_id(dataset_spec, options)}_test_report.json"

    def cv_summary_path(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.model_root(model_spec, dataset_spec, options) / f"{model_spec.output_name}_{self.variant_id(dataset_spec, options)}_cv_summary.json"

    def tokenized_root(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, fold_id: int, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "tokenized" / model_spec.output_name / f"fold_{fold_id}"

    def final_tokenized_root(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.variant_root(dataset_spec, options) / "tokenized" / model_spec.output_name / "final_model"

    def rendered_config_dir(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.model_root(model_spec, dataset_spec, options) / "rendered_configs"

    def rendered_train_config(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, fold_id: int, options: RunOptions | None = None) -> Path:
        return self.rendered_config_dir(model_spec, dataset_spec, options) / f"train_fold_{fold_id}.yaml"

    def rendered_predict_config(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, fold_id: int, options: RunOptions | None = None) -> Path:
        return self.rendered_config_dir(model_spec, dataset_spec, options) / f"predict_fold_{fold_id}.yaml"

    def rendered_final_train_config(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.rendered_config_dir(model_spec, dataset_spec, options) / "train_final.yaml"

    def rendered_test_predict_config(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.rendered_config_dir(model_spec, dataset_spec, options) / "predict_test.yaml"

    def run_manifest(self, model_spec: ModelSpec, dataset_spec: DatasetSpec, options: RunOptions | None = None) -> Path:
        return self.model_root(model_spec, dataset_spec, options) / "run_manifest.json"

    def slurm_script(self, job_name: str) -> Path:
        return self.logs_root() / "slurm_scripts" / f"{job_name}.sh"


def resolver_for(run_spec: RunSpec) -> PathResolver:
    return PathResolver(run_spec.profile)

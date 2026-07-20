from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
FT_ROOT = PACKAGE_DIR.parent
REPO_ROOT = FT_ROOT.parent
CONFIG_ROOT = FT_ROOT / "configs"
PROFILE_DIR = CONFIG_ROOT / "profiles"
DATASET_SPEC_DIR = CONFIG_ROOT / "dataset_specs"
MODEL_SPEC_DIR = CONFIG_ROOT / "model_specs"


VALID_ROLE_MODES = ("joint", "infant", "caregiver")


def _load_env_file() -> None:
    env_file = os.environ.get("SWAN_ENV_FILE")
    if env_file:
        env_path = Path(env_file)
    else:
        candidates = [
            Path.cwd() / ".env",
            REPO_ROOT / ".env",
            FT_ROOT / ".env",
        ]
        env_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if env_path is None or not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expand_path(value: str | None, base_dir: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or "%" in expanded:
        return None
    path = Path(expanded)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _expand_value(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or "%" in expanded:
        return None
    return expanded


def _profile_env_name(profile_id: str, env_name: str) -> str:
    if not env_name.startswith("SWAN_"):
        return env_name
    profile_token = profile_id.upper().replace("-", "_")
    return f"SWAN_{profile_token}_{env_name.removeprefix('SWAN_')}"


def _env_value(profile_id: str, env_name: str) -> str | None:
    return os.environ.get(_profile_env_name(profile_id, env_name)) or os.environ.get(env_name)


def _path_from_env_or_value(
    profile_id: str,
    env_name: str,
    value: str | None,
    base_dir: Path,
    fallback: Path | None = None,
) -> Path | None:
    env_path = _expand_path(_env_value(profile_id, env_name), base_dir)
    if env_path is not None:
        return env_path
    value_path = _expand_path(value, base_dir)
    return value_path if value_path is not None else fallback


def _value_from_env_or_value(profile_id: str, env_name: str, value: str | None) -> str | None:
    env_value = _expand_value(_env_value(profile_id, env_name))
    if env_value is not None:
        return env_value
    return _expand_value(value)


@dataclass(frozen=True)
class SlurmConfig:
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    gres: str | None = None
    cpus_per_task: int | None = None
    mem: str | None = None
    time: str | None = None
    array_parallelism: int | None = None
    python_bin: str = "python"
    extra_sbatch_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SlurmConfig":
        data = data or {}
        return cls(
            partition=data.get("partition"),
            account=data.get("account"),
            qos=data.get("qos"),
            gres=data.get("gres"),
            cpus_per_task=data.get("cpus_per_task"),
            mem=data.get("mem"),
            time=data.get("time"),
            array_parallelism=data.get("array_parallelism"),
            python_bin=data.get("python_bin", "python"),
            extra_sbatch_args=list(data.get("extra_sbatch_args", [])),
        )


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    launcher: str
    work_root: Path | None
    session_root: Path | None
    data_root: Path
    output_root: Path
    variants_root: Path | None
    cache_root: Path | None
    temp_root: Path
    logs_root: Path
    env_name: str
    python_bin: str
    conda_activate: str | None
    local_gpus: int | None
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    @classmethod
    def from_dict(cls, profile_id: str, data: dict[str, Any]) -> "RuntimeProfile":
        base_dir = FT_ROOT
        work_root = _path_from_env_or_value(profile_id, "SWAN_WORK_ROOT", data.get("work_root"), base_dir)
        data_root = _path_from_env_or_value(profile_id, "SWAN_DATA_ROOT", data.get("data_root"), base_dir, fallback=work_root or Path("."))
        output_root = _path_from_env_or_value(
            profile_id,
            "SWAN_OUTPUT_ROOT",
            data.get("output_root"),
            base_dir,
            fallback=(work_root / "output" if work_root is not None else data_root / "output"),
        )
        cache_root = _path_from_env_or_value(
            profile_id,
            "SWAN_CACHE_ROOT",
            data.get("cache_root"),
            base_dir,
            fallback=(work_root / "cache" if work_root is not None else data_root / "cache"),
        )
        temp_root = _path_from_env_or_value(
            profile_id,
            "SWAN_TEMP_ROOT",
            data.get("temp_root"),
            base_dir,
            fallback=(work_root / "tmp" if work_root is not None else data_root / "tmp"),
        ) or Path(tempfile.gettempdir())
        logs_root = _path_from_env_or_value(
            profile_id,
            "SWAN_LOGS_ROOT",
            data.get("logs_root"),
            base_dir,
            fallback=(work_root / "logs" / profile_id if work_root is not None else data_root / "logs" / profile_id),
        ) or output_root
        return cls(
            id=profile_id,
            launcher=data["launcher"],
            work_root=work_root,
            session_root=_path_from_env_or_value(profile_id, "SWAN_SESSION_ROOT", data.get("session_root"), base_dir),
            data_root=data_root,
            output_root=output_root,
            variants_root=_path_from_env_or_value(profile_id, "SWAN_VARIANTS_ROOT", data.get("variants_root"), base_dir),
            cache_root=cache_root,
            temp_root=temp_root,
            logs_root=logs_root,
            env_name=data.get("env_name", "swan"),
            python_bin=_value_from_env_or_value(profile_id, "SWAN_PYTHON_BIN", data.get("python_bin")) or "python",
            conda_activate=_value_from_env_or_value(profile_id, "SWAN_CONDA_ACTIVATE", data.get("conda_activate")),
            local_gpus=data.get("local_gpus"),
            slurm=SlurmConfig.from_dict(data.get("slurm")),
        )


def _ensure_runtime_dirs(profile: RuntimeProfile) -> RuntimeProfile:
    for path in (profile.output_root, profile.logs_root, profile.temp_root):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    if profile.variants_root is not None:
        try:
            profile.variants_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    if profile.cache_root is not None:
        try:
            profile.cache_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return profile


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    modalities: tuple[str, ...]
    role_mode: str
    include_background: bool
    prompt_template: str
    base_dataset_file: str
    base_dataset_info_file: str
    base_folds_file: str
    dataset_subdir: str
    clip_subdir: str
    audio_subdir: str
    reuse_video_container_for_audio: bool
    reuse_existing_clips: bool
    build_from_existing_root: bool
    excluded_labels: tuple[str, ...]

    @classmethod
    def from_dict(cls, spec_id: str, data: dict[str, Any]) -> "DatasetSpec":
        role_mode = data.get("role_mode", "joint")
        if role_mode not in VALID_ROLE_MODES:
            raise ValueError(f"Invalid role_mode in dataset spec {spec_id}: {role_mode}")
        return cls(
            id=spec_id,
            modalities=tuple(data["modalities"]),
            role_mode=role_mode,
            include_background=bool(data["include_background"]),
            prompt_template=data["prompt_template"],
            base_dataset_file=data.get("base_dataset_file", "dataset.json"),
            base_dataset_info_file=data.get("base_dataset_info_file", "dataset_info.json"),
            base_folds_file=data.get("base_folds_file", "folds.json"),
            dataset_subdir=data.get("dataset_subdir", spec_id),
            clip_subdir=data.get("clip_subdir", "clips"),
            audio_subdir=data.get("audio_subdir", "audio_clips"),
            reuse_video_container_for_audio=bool(data.get("reuse_video_container_for_audio", False)),
            reuse_existing_clips=bool(data.get("reuse_existing_clips", True)),
            build_from_existing_root=bool(data.get("build_from_existing_root", True)),
            excluded_labels=tuple(sorted(set(label.lower() for label in data.get("excluded_labels", [])))),
        )

    def effective_role_mode(self, options: "RunOptions") -> str:
        return options.role_mode or self.role_mode

    def effective_excluded_labels(self, options: "RunOptions") -> tuple[str, ...]:
        return tuple(sorted(set(self.excluded_labels).union(options.exclude_labels)))

    def variant_id(self, options: "RunOptions") -> str:
        parts = [self.dataset_subdir or self.id]
        role_mode = self.effective_role_mode(options)
        if role_mode != self.role_mode and role_mode not in parts[0]:
            parts.append(role_mode)
        excluded = self.effective_excluded_labels(options)
        if excluded:
            suffix = "_".join(excluded)
            if f"no_{suffix}" not in parts[0] and f"excl_{suffix}" not in parts[0]:
                parts.append(f"excl_{suffix}")
        return "_".join(parts)


def _infer_model_family(model_id: str, data: dict[str, Any]) -> str:
    base_model = str(data.get("base_model", "")).lower()
    template = str(data.get("template", "")).lower()
    modality = str(data.get("modality", "")).lower()
    combined = f"{model_id.lower()} {base_model} {template}"

    if modality == "audio":
        return "audio_qwen" if "qwen" in combined else "audio"
    if modality == "omni":
        return "omni_qwen" if "qwen" in combined else "omni"
    if "glm" in combined:
        return "vision_glm"
    if "internvl" in combined:
        return "vision_internvl"
    if "minicpm" in combined:
        return "vision_minicpm"
    if modality == "video" and "qwen" in combined:
        return "vision_qwen"
    return modality or "generic"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    base_model: str
    template: str
    modality: str
    family: str
    output_name: str
    trust_remote_code: bool
    allow_quantization: bool
    h100_full_precision_default: bool
    train_defaults: dict[str, Any]
    predict_defaults: dict[str, Any]
    extra_config: dict[str, Any]

    @classmethod
    def from_dict(cls, model_id: str, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            id=model_id,
            base_model=data["base_model"],
            template=data["template"],
            modality=data["modality"],
            family=str(data.get("family") or _infer_model_family(model_id, data)),
            output_name=data.get("output_name", model_id),
            trust_remote_code=bool(data.get("trust_remote_code", True)),
            allow_quantization=bool(data.get("allow_quantization", False)),
            h100_full_precision_default=bool(data.get("h100_full_precision_default", True)),
            train_defaults=dict(data.get("train_defaults", {})),
            predict_defaults=dict(data.get("predict_defaults", {})),
            extra_config=dict(data.get("extra_config", {})),
        )


@dataclass(frozen=True)
class RunOptions:
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    overwrite: bool = False
    dry_run: bool = False
    submit: bool = False
    quantization: bool | None = None
    per_device_train_batch_size: int | None = None
    per_device_eval_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    num_train_epochs: float | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    early_stopping_steps: int | None = None
    load_best_model_at_end: bool | None = None
    role_mode: str | None = None
    exclude_labels: tuple[str, ...] = ()
    execute_local: bool = False


@dataclass(frozen=True)
class RunSpec:
    profile: RuntimeProfile
    dataset_spec: DatasetSpec
    model_spec: ModelSpec | None
    options: RunOptions

    def with_options(self, **kwargs: Any) -> "RunSpec":
        return replace(self, options=replace(self.options, **kwargs))


def validate_model_dataset_compatibility(run_spec: RunSpec) -> None:
    model_spec = run_spec.model_spec
    if model_spec is None:
        return

    modalities = set(run_spec.dataset_spec.modalities)
    model_modality = model_spec.modality

    if model_modality == "audio":
        if modalities != {"audio"}:
            raise ValueError(
                f"Model {model_spec.id} requires an audio-only dataset spec, but "
                f"{run_spec.dataset_spec.id} has modalities {sorted(modalities)}."
            )
        return

    if model_modality == "video":
        if "video" not in modalities or "audio" in modalities:
            raise ValueError(
                f"Model {model_spec.id} requires a video-only dataset spec, but "
                f"{run_spec.dataset_spec.id} has modalities {sorted(modalities)}."
            )
        return

    if model_modality == "omni":
        if modalities != {"video", "audio"}:
            raise ValueError(
                f"Model {model_spec.id} requires a joint video+audio dataset spec, but "
                f"{run_spec.dataset_spec.id} has modalities {sorted(modalities)}."
            )
        return


def _resolve_probe_path(candidate_value: str, profile: RuntimeProfile) -> Path:
    candidate = Path(str(candidate_value).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    return profile.data_root / candidate



def _collect_probe_targets_from_samples(samples: list[dict[str, Any]], profile: RuntimeProfile, limit: int = 8) -> list[Path]:
    probe_targets: list[Path] = []
    for sample in samples:
        videos = sample.get("videos") or []
        if not videos:
            continue
        probe_targets.append(_resolve_probe_path(videos[0], profile))
        if len(probe_targets) >= limit:
            break
    return probe_targets



def _load_probe_targets_from_dataset(path: Path, profile: RuntimeProfile, limit: int = 8) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    return _collect_probe_targets_from_samples(samples, profile, limit=limit)



def _variant_root_for_validation(run_spec: RunSpec) -> Path:
    variants_root = run_spec.profile.variants_root or (run_spec.profile.data_root / "variants")
    return variants_root / run_spec.dataset_spec.variant_id(run_spec.options)



def validate_omni_media_requirements(run_spec: RunSpec) -> None:
    model_spec = run_spec.model_spec
    if model_spec is None or model_spec.family != "omni_qwen":
        return
    if not run_spec.dataset_spec.reuse_video_container_for_audio:
        return

    variant_root = _variant_root_for_validation(run_spec)
    variant_candidates = [variant_root / f"fold_{fold_id}_val.json" for fold_id in run_spec.options.folds] + [
        variant_root / "test.json",
        variant_root / "train_dev.json",
        variant_root / "dataset.json",
    ]

    dataset_path: Path | None = None
    probe_targets: list[Path] = []
    for candidate in variant_candidates:
        if candidate.exists():
            dataset_path = candidate
            probe_targets = _load_probe_targets_from_dataset(candidate, run_spec.profile, limit=8)
            if probe_targets:
                break

    if not probe_targets:
        base_dataset_path = run_spec.profile.data_root / run_spec.dataset_spec.base_dataset_file
        dataset_path = base_dataset_path
        if not base_dataset_path.exists():
            raise FileNotFoundError(
                f"Base dataset not found at {base_dataset_path}. Build or mount the base dataset before running Omni jobs."
            )
        probe_targets = _load_probe_targets_from_dataset(base_dataset_path, run_spec.profile, limit=8)

    if not probe_targets:
        raise ValueError(
            f"Dataset spec {run_spec.dataset_spec.id} is configured for Omni, but no source video clips were found in {dataset_path}."
        )

    if not _ffprobe_has_audio_stream(probe_targets):
        raise ValueError(
            f"Dataset spec {run_spec.dataset_spec.id} reuses clip .mp4 containers as audio inputs, but the sampled "
            "clips do not expose an embedded audio stream. Run `python check_mp4_audio.py <clips_root>` and either "
            "fix the clips or switch to a dataset spec that materializes standalone audio files."
        )



def _ffprobe_has_audio_stream(paths: list[Path]) -> bool:
    for path in paths:
        if not path.exists():
            continue
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffprobe is required to validate embedded audio streams for Omni dataset specs."
            ) from exc
        if result.returncode != 0:
            continue
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            continue
        if payload.get("streams"):
            return True
    return False


def load_profile(profile_id: str) -> RuntimeProfile:
    path = PROFILE_DIR / f"{profile_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown profile: {profile_id}")
    return _ensure_runtime_dirs(RuntimeProfile.from_dict(profile_id, _read_json(path)))


def load_dataset_spec(spec_id: str) -> DatasetSpec:
    path = DATASET_SPEC_DIR / f"{spec_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown dataset spec: {spec_id}")
    return DatasetSpec.from_dict(spec_id, _read_json(path))


def load_model_spec(model_id: str) -> ModelSpec:
    path = MODEL_SPEC_DIR / f"{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown model: {model_id}")
    return ModelSpec.from_dict(model_id, _read_json(path))


def list_named_configs(directory: Path) -> list[str]:
    return sorted(path.stem for path in directory.glob("*.json"))

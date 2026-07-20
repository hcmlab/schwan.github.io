from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .baseline.config import BASELINE_SPEC_DIR, list_baseline_specs, load_baseline_spec
from .config import (
    DATASET_SPEC_DIR,
    MODEL_SPEC_DIR,
    PROFILE_DIR,
    RunOptions,
    RunSpec,
    list_named_configs,
    load_dataset_spec,
    load_model_spec,
    load_profile,
    validate_model_dataset_compatibility,
    validate_omni_media_requirements,
)
from .dataset import build_dataset_variant, inspect_dataset_variant
from .evaluation import score_cv, score_fold, score_test
from .folds import write_folds
from .launcher import LocalLauncher, SlurmLauncher, launcher_for
from .llamafactory import render_final_train_config, render_predict_config, render_test_predict_config, render_train_config, run_llamafactory, write_run_manifest
from .paths import resolver_for


def _add_common_run_args(parser: argparse.ArgumentParser, require_model: bool = True) -> None:
    parser.add_argument("--profile", required=True, choices=list_named_configs(PROFILE_DIR))
    parser.add_argument("--dataset-spec", required=True, choices=list_named_configs(DATASET_SPEC_DIR))
    if require_model:
        parser.add_argument("--model", required=True, choices=list_named_configs(MODEL_SPEC_DIR))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--quantization", choices=["on", "off"], default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--early-stopping-steps", type=int, default=None)
    parser.add_argument("--disable-best-model-load", action="store_true")
    parser.add_argument("--role-mode", choices=["joint", "infant", "caregiver"], default=None)
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--exclude-labels", default=None)


def _run_spec_from_args(args: argparse.Namespace, require_model: bool = True) -> RunSpec:
    profile = load_profile(args.profile)
    dataset_spec = load_dataset_spec(args.dataset_spec)
    model_spec = load_model_spec(args.model) if require_model else None
    quantization = None
    if args.quantization == "on":
        quantization = True
    elif args.quantization == "off":
        quantization = False
    options = RunOptions(
        folds=tuple(args.folds),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        submit=args.submit,
        quantization=quantization,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        early_stopping_steps=args.early_stopping_steps,
        load_best_model_at_end=False if args.disable_best_model_load else None,
        role_mode=args.role_mode,
        exclude_labels=tuple(
            sorted(
                {
                    label.strip().lower()
                    for label in ([*(args.exclude_label or [])] + ((args.exclude_labels or "").split(",") if args.exclude_labels else []))
                    if label and label.strip()
                }
            )
        ),
        execute_local=args.execute_local,
    )
    run_spec = RunSpec(profile=profile, dataset_spec=dataset_spec, model_spec=model_spec, options=options)
    if require_model:
        validate_model_dataset_compatibility(run_spec)
        should_validate_media = not (profile.launcher == "slurm" and args.submit and not args.execute_local)
        if should_validate_media:
            validate_omni_media_requirements(run_spec)
    return run_spec


def _apply_hf_cache_env(run_spec: RunSpec) -> None:
    cache_root = run_spec.profile.cache_root
    temp_root = run_spec.profile.temp_root
    dataset_cache_root = temp_root / "hf_datasets_cache"
    dataset_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(dataset_cache_root)

    if cache_root is None:
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    env_paths = {
        "HF_HOME": cache_root,
        "HUGGINGFACE_HUB_CACHE": cache_root / "hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
    }
    for key, value in env_paths.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)


def _icep_definitions_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[2] / "icep_definitions.json"
    return candidate if candidate.exists() else None


def cmd_folds_create(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    output_path = Path(args.output) if args.output else profile.data_root / load_dataset_spec(args.dataset_spec).base_folds_file
    session_root = profile.session_root
    if session_root is None or not session_root.exists():
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            print(json.dumps({
                "output": str(output_path),
                "total_sessions": payload.get("total_sessions"),
                "train_dev_count": payload.get("train_dev_count"),
                "test_count": payload.get("test_count"),
                "reused_existing": True,
            }, indent=2))
            return 0
        if session_root is None:
            raise ValueError(f"Profile {profile.id} has no session_root configured and no existing folds file at {output_path}")
        raise FileNotFoundError(f"Session root not found: {session_root}. No existing folds file found at {output_path}.")
    payload = write_folds(session_root, output_path, n_folds=args.n_folds, test_fraction=args.test_fraction, seed=args.seed)
    print(json.dumps({"output": str(output_path), "total_sessions": payload["total_sessions"], "train_dev_count": payload.get("train_dev_count"), "test_count": payload.get("test_count")}, indent=2))
    return 0


def cmd_dataset_build(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args, require_model=False)
    result = build_dataset_variant(run_spec, icep_definitions_path=_icep_definitions_path())
    print(json.dumps(result, indent=2))
    return 0


def cmd_dataset_inspect(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args, require_model=False)
    print(json.dumps(inspect_dataset_variant(run_spec), indent=2))
    return 0


def _ensure_local_execution(run_spec: RunSpec, area: str, action: str, fold_id: int | None = None) -> int:
    launch = launcher_for(run_spec)
    if isinstance(launch, LocalLauncher):
        return launch.run(area, action, extra=[str(fold_id)] if fold_id is not None else [])
    if not run_spec.options.submit:
        raise ValueError("SLURM profile selected without --submit. Add --submit or use a local profile.")
    return 0


def cmd_train_fold(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    fold_id = args.folds[0]
    if not run_spec.options.execute_local and run_spec.profile.launcher == "slurm":
        if not run_spec.options.submit:
            raise ValueError("Add --submit to submit this fold to SLURM")
        job_id = SlurmLauncher(run_spec.with_options(folds=(fold_id,))).submit("train", "fold")
        print(json.dumps({"job_id": job_id, "fold": fold_id}, indent=2))
        return 0
    config_path = render_train_config(run_spec.with_options(folds=(fold_id,)), fold_id)
    write_run_manifest(run_spec.with_options(folds=(fold_id,)))
    _apply_hf_cache_env(run_spec)
    if not run_spec.options.dry_run:
        run_llamafactory(config_path, predict=False)
    print(json.dumps({"config": str(config_path), "fold": fold_id}, indent=2))
    return 0


def cmd_train_cv(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    if not run_spec.options.execute_local and run_spec.profile.launcher == "slurm":
        if not run_spec.options.submit:
            raise ValueError("Add --submit to submit CV training to SLURM")
        job_ids = []
        for fold_id in run_spec.options.folds:
            fold_spec = run_spec.with_options(folds=(fold_id,))
            job_ids.append(SlurmLauncher(fold_spec).submit("train", "fold"))
        print(json.dumps({"job_ids": job_ids}, indent=2))
        return 0
    for fold_id in run_spec.options.folds:
        fold_spec = run_spec.with_options(folds=(fold_id,), execute_local=True)
        cmd_train_fold(argparse.Namespace(**{**vars(args), "folds": [fold_id], "execute_local": True}))
    return 0


def cmd_train_final(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    config_path = render_final_train_config(run_spec)
    write_run_manifest(run_spec)
    _apply_hf_cache_env(run_spec)
    if not run_spec.options.dry_run:
        run_llamafactory(config_path, predict=False)
    print(json.dumps({"config": str(config_path), "mode": "final_train"}, indent=2))
    return 0


def cmd_predict_fold(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    fold_id = args.folds[0]
    if not run_spec.options.execute_local and run_spec.profile.launcher == "slurm":
        if not run_spec.options.submit:
            raise ValueError("Add --submit to submit this fold to SLURM")
        job_id = SlurmLauncher(run_spec.with_options(folds=(fold_id,))).submit("predict", "fold")
        print(json.dumps({"job_id": job_id, "fold": fold_id}, indent=2))
        return 0
    config_path = render_predict_config(run_spec.with_options(folds=(fold_id,)), fold_id)
    _apply_hf_cache_env(run_spec)
    if not run_spec.options.dry_run:
        run_llamafactory(config_path, predict=True)
    print(json.dumps({"config": str(config_path), "fold": fold_id}, indent=2))
    return 0


def cmd_predict_cv(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    if not run_spec.options.execute_local and run_spec.profile.launcher == "slurm":
        if not run_spec.options.submit:
            raise ValueError("Add --submit to submit CV prediction to SLURM")
        job_ids = []
        for fold_id in run_spec.options.folds:
            fold_spec = run_spec.with_options(folds=(fold_id,))
            job_ids.append(SlurmLauncher(fold_spec).submit("predict", "fold"))
        print(json.dumps({"job_ids": job_ids}, indent=2))
        return 0
    for fold_id in run_spec.options.folds:
        cmd_predict_fold(argparse.Namespace(**{**vars(args), "folds": [fold_id], "execute_local": True}))
    return 0


def cmd_predict_test(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    config_path = render_test_predict_config(run_spec)
    _apply_hf_cache_env(run_spec)
    if not run_spec.options.dry_run:
        run_llamafactory(config_path, predict=True)
    print(json.dumps({"config": str(config_path), "mode": "test_predict"}, indent=2))
    return 0


def cmd_report_fold(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    fold_id = args.folds[0]
    report = score_fold(run_spec.with_options(folds=(fold_id,)), fold_id)
    print(json.dumps({"fold": fold_id, "report_path": str(resolver_for(run_spec).fold_eval_report(run_spec.model_spec, run_spec.dataset_spec, fold_id, run_spec.options)), "parse_errors": report["parse_errors"]}, indent=2))
    return 0


def cmd_report_cv(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    summary = score_cv(run_spec)
    print(json.dumps({"summary_path": str(resolver_for(run_spec).cv_summary_path(run_spec.model_spec, run_spec.dataset_spec, run_spec.options)), "folds": summary["folds_evaluated"]}, indent=2))
    return 0


def cmd_report_test(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    report = score_test(run_spec)
    print(json.dumps({"report_path": str(resolver_for(run_spec).final_test_eval_report(run_spec.model_spec, run_spec.dataset_spec, run_spec.options)), "test_samples": report["test_samples"]}, indent=2))
    return 0


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    run_spec = _run_spec_from_args(args)
    if run_spec.profile.launcher == "slurm" and not run_spec.options.execute_local:
        if not run_spec.options.submit:
            raise ValueError("Add --submit to run the pipeline on SLURM")
        train_jobs = []
        for fold_id in run_spec.options.folds:
            fold_spec = run_spec.with_options(folds=(fold_id,))
            train_jobs.append(SlurmLauncher(fold_spec).submit("train", "fold"))
        dependency = ":".join(train_jobs)
        predict_jobs = []
        for fold_id in run_spec.options.folds:
            fold_spec = run_spec.with_options(folds=(fold_id,))
            predict_jobs.append(SlurmLauncher(fold_spec).submit("predict", "fold", dependency=dependency))
        report_job = SlurmLauncher(run_spec).submit("report", "cv", dependency=":".join(predict_jobs))
        print(json.dumps({"train_jobs": train_jobs, "predict_jobs": predict_jobs, "report_job": report_job}, indent=2))
        return 0

    cmd_train_cv(argparse.Namespace(**{**vars(args), "execute_local": True}))
    cmd_predict_cv(argparse.Namespace(**{**vars(args), "execute_local": True}))
    cmd_report_cv(argparse.Namespace(**vars(args)))
    return 0


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, choices=list_named_configs(PROFILE_DIR))
    parser.add_argument("--dataset-spec", required=True, choices=list_named_configs(DATASET_SPEC_DIR))
    parser.add_argument("--baseline-spec", required=True, choices=list_baseline_specs())
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--modality-combo", choices=["video", "audio", "omni"], default=None,
                        help="Run a single modality combo (for array jobs). If omitted, runs all combos from the baseline spec.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--role-mode", choices=["joint", "infant", "caregiver"], default=None)
    parser.add_argument("--exclude-label", action="append", default=[])
    parser.add_argument("--exclude-labels", default=None)
    parser.add_argument("--device", default="cuda")


def _baseline_run_spec(args: argparse.Namespace) -> RunSpec:
    profile = load_profile(args.profile)
    dataset_spec = load_dataset_spec(args.dataset_spec)
    options = RunOptions(
        folds=tuple(args.folds),
        overwrite=args.overwrite,
        submit=getattr(args, "submit", False),
        execute_local=getattr(args, "execute_local", False),
        role_mode=args.role_mode,
        exclude_labels=tuple(
            sorted(
                {
                    label.strip().lower()
                    for label in ([*(args.exclude_label or [])] + ((args.exclude_labels or "").split(",") if args.exclude_labels else []))
                    if label and label.strip()
                }
            )
        ),
    )
    return RunSpec(profile=profile, dataset_spec=dataset_spec, model_spec=None, options=options)


def _baseline_slurm_submit(run_spec: RunSpec, baseline_spec_id: str, action: str, device: str = "cuda", dependency: str | None = None) -> str:
    """Submit slurm/baseline.sh with env vars, like the other SLURM scripts."""
    import subprocess

    from .config import FT_ROOT

    profile = run_spec.profile
    slurm = profile.slurm
    resolver = resolver_for(run_spec)
    job_name = f"swan_baseline_{action}"
    script_path = FT_ROOT / "slurm" / "baseline.sh"

    folds_csv = ",".join(str(f) for f in run_spec.options.folds)
    exclude_csv = ",".join(run_spec.options.exclude_labels)

    sbatch_args = ["sbatch", "--parsable", f"--job-name={job_name}"]
    if slurm.partition:
        sbatch_args.append(f"--partition={slurm.partition}")
    if slurm.account:
        sbatch_args.append(f"--account={slurm.account}")
    if slurm.qos:
        sbatch_args.append(f"--qos={slurm.qos}")
    if slurm.gres:
        sbatch_args.append(f"--gres={slurm.gres}")
    if slurm.cpus_per_task:
        sbatch_args.append(f"--cpus-per-task={slurm.cpus_per_task}")
    if slurm.mem:
        sbatch_args.append(f"--mem={slurm.mem}")
    if slurm.time:
        sbatch_args.append(f"--time={slurm.time}")
    if dependency:
        sbatch_args.append(f"--dependency=afterok:{dependency}")
    output_path = resolver.logs_root() / f"{job_name}_%j.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sbatch_args.append(f"--output={output_path}")
    sbatch_args.extend(slurm.extra_sbatch_args)

    # Pass config via --export so baseline.sh picks them up
    export_vars = [
        f"FT_DIR={FT_ROOT}",
        f"PROFILE={profile.id}",
        f"DATASET_SPEC={run_spec.dataset_spec.id}",
        f"BASELINE_SPEC={baseline_spec_id}",
        f"FOLDS_CSV={folds_csv}",
        f"DEVICE={device}",
        f"ACTION={action}",
        f"OVERWRITE={'1' if run_spec.options.overwrite else '0'}",
    ]
    if run_spec.options.role_mode:
        export_vars.append(f"ROLE_MODE={run_spec.options.role_mode}")
    if exclude_csv:
        export_vars.append(f"EXCLUDE_LABELS={exclude_csv}")
    sbatch_args.append(f"--export=ALL,{','.join(export_vars)}")

    sbatch_args.append(str(script_path))

    result = subprocess.run(sbatch_args, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def cmd_baseline_extract(args: argparse.Namespace) -> int:
    run_spec = _baseline_run_spec(args)

    if run_spec.profile.launcher == "slurm" and not run_spec.options.execute_local:
        if not run_spec.options.submit:
            raise ValueError("SLURM profile selected without --submit. Add --submit or use a local profile.")
        job_id = _baseline_slurm_submit(run_spec, args.baseline_spec, "extract", device=args.device)
        print(json.dumps({"job_id": job_id, "action": "extract"}, indent=2))
        return 0

    from .baseline.extract import extract_features
    baseline_spec = load_baseline_spec(args.baseline_spec)
    stats = extract_features(run_spec, baseline_spec, device=args.device, overwrite=args.overwrite)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_baseline_train_fold(args: argparse.Namespace) -> int:
    """Train a single fold + single modality combo. Used by SLURM array jobs."""
    run_spec = _baseline_run_spec(args)
    baseline_spec = load_baseline_spec(args.baseline_spec)
    combo = args.modality_combo
    fold_id = args.folds[0]

    if not combo:
        raise ValueError("--modality-combo is required for train-fold")

    from .baseline.train import _train_fold_for_combo
    from .baseline.dataset_loader import load_fold_samples
    from .paths import resolver_for as _resolver_for

    resolver = _resolver_for(run_spec)
    variant_root = resolver.variant_root(run_spec.dataset_spec, run_spec.options)
    train_samples = load_fold_samples(variant_root / f"fold_{fold_id}_train.json")
    val_samples = load_fold_samples(variant_root / f"fold_{fold_id}_val.json")

    result = _train_fold_for_combo(run_spec, baseline_spec, fold_id, combo, train_samples, val_samples)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_baseline_train_cv(args: argparse.Namespace) -> int:
    run_spec = _baseline_run_spec(args)

    if run_spec.profile.launcher == "slurm" and not run_spec.options.execute_local:
        if not run_spec.options.submit:
            raise ValueError("SLURM profile selected without --submit. Add --submit or use a local profile.")
        job_id = _baseline_slurm_submit(run_spec, args.baseline_spec, "train-cv", device=args.device)
        print(json.dumps({"job_id": job_id, "action": "train-cv"}, indent=2))
        return 0

    from .baseline.train import train_and_predict_cv
    baseline_spec = load_baseline_spec(args.baseline_spec)
    combo = getattr(args, "modality_combo", None)
    results = train_and_predict_cv(run_spec, baseline_spec, modality_combo=combo)
    print(json.dumps(results, indent=2, default=str))
    return 0


def cmd_baseline_report_cv(args: argparse.Namespace) -> int:
    run_spec = _baseline_run_spec(args)

    if run_spec.profile.launcher == "slurm" and not run_spec.options.execute_local:
        if not run_spec.options.submit:
            raise ValueError("SLURM profile selected without --submit. Add --submit or use a local profile.")
        job_id = _baseline_slurm_submit(run_spec, args.baseline_spec, "report-cv", device=args.device)
        print(json.dumps({"job_id": job_id, "action": "report-cv"}, indent=2))
        return 0

    baseline_spec = load_baseline_spec(args.baseline_spec)
    from dataclasses import replace
    results = {}
    for combo in baseline_spec.modality_combos:
        eval_spec = replace(run_spec, model_spec=baseline_spec.to_model_spec(combo))
        summary = score_cv(eval_spec)
        summary_path = resolver_for(eval_spec).cv_summary_path(eval_spec.model_spec, eval_spec.dataset_spec, eval_spec.options)
        results[combo] = {"summary_path": str(summary_path), "folds": summary["folds_evaluated"]}
    print(json.dumps(results, indent=2))
    return 0


def cmd_baseline_pipeline(args: argparse.Namespace) -> int:
    run_spec = _baseline_run_spec(args)

    if run_spec.profile.launcher == "slurm" and not run_spec.options.execute_local:
        if not run_spec.options.submit:
            raise ValueError("SLURM profile selected without --submit. Add --submit or use a local profile.")
        # Chain: extract → train-cv → report-cv with dependencies
        extract_job = _baseline_slurm_submit(run_spec, args.baseline_spec, "extract", device=args.device)
        train_job = _baseline_slurm_submit(run_spec, args.baseline_spec, "train-cv", device=args.device, dependency=extract_job)
        report_job = _baseline_slurm_submit(run_spec, args.baseline_spec, "report-cv", device=args.device, dependency=train_job)
        print(json.dumps({"extract_job": extract_job, "train_job": train_job, "report_job": report_job}, indent=2))
        return 0

    cmd_baseline_extract(args)
    cmd_baseline_train_cv(args)
    cmd_baseline_report_cv(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swan-ft")
    top = parser.add_subparsers(dest="area", required=True)

    folds_parser = top.add_parser("folds")
    folds_sub = folds_parser.add_subparsers(dest="action", required=True)
    folds_create = folds_sub.add_parser("create")
    folds_create.add_argument("--profile", required=True, choices=list_named_configs(PROFILE_DIR))
    folds_create.add_argument("--dataset-spec", required=True, choices=list_named_configs(DATASET_SPEC_DIR))
    folds_create.add_argument("--n-folds", type=int, default=5)
    folds_create.add_argument("--test-fraction", type=float, default=0.2)
    folds_create.add_argument("--seed", type=int, default=42)
    folds_create.add_argument("--output", default=None)
    folds_create.set_defaults(func=cmd_folds_create)

    dataset_parser = top.add_parser("dataset")
    dataset_sub = dataset_parser.add_subparsers(dest="action", required=True)
    dataset_build = dataset_sub.add_parser("build")
    _add_common_run_args(dataset_build, require_model=False)
    dataset_build.set_defaults(func=cmd_dataset_build)
    dataset_inspect = dataset_sub.add_parser("inspect")
    _add_common_run_args(dataset_inspect, require_model=False)
    dataset_inspect.set_defaults(func=cmd_dataset_inspect)

    train_parser = top.add_parser("train")
    train_sub = train_parser.add_subparsers(dest="action", required=True)
    train_fold = train_sub.add_parser("fold")
    _add_common_run_args(train_fold)
    train_fold.set_defaults(func=cmd_train_fold)
    train_cv = train_sub.add_parser("cv")
    _add_common_run_args(train_cv)
    train_cv.set_defaults(func=cmd_train_cv)
    train_final = train_sub.add_parser("final")
    _add_common_run_args(train_final)
    train_final.set_defaults(func=cmd_train_final)

    predict_parser = top.add_parser("predict")
    predict_sub = predict_parser.add_subparsers(dest="action", required=True)
    predict_fold = predict_sub.add_parser("fold")
    _add_common_run_args(predict_fold)
    predict_fold.set_defaults(func=cmd_predict_fold)
    predict_cv = predict_sub.add_parser("cv")
    _add_common_run_args(predict_cv)
    predict_cv.set_defaults(func=cmd_predict_cv)
    predict_test = predict_sub.add_parser("test")
    _add_common_run_args(predict_test)
    predict_test.set_defaults(func=cmd_predict_test)

    report_parser = top.add_parser("report")
    report_sub = report_parser.add_subparsers(dest="action", required=True)
    report_fold = report_sub.add_parser("fold")
    _add_common_run_args(report_fold)
    report_fold.set_defaults(func=cmd_report_fold)
    report_cv = report_sub.add_parser("cv")
    _add_common_run_args(report_cv)
    report_cv.set_defaults(func=cmd_report_cv)
    report_test = report_sub.add_parser("test")
    _add_common_run_args(report_test)
    report_test.set_defaults(func=cmd_report_test)

    run_parser = top.add_parser("run")
    run_sub = run_parser.add_subparsers(dest="action", required=True)
    pipeline = run_sub.add_parser("pipeline")
    _add_common_run_args(pipeline)
    pipeline.set_defaults(func=cmd_run_pipeline)

    baseline_parser = top.add_parser("baseline")
    baseline_sub = baseline_parser.add_subparsers(dest="action", required=True)

    bl_extract = baseline_sub.add_parser("extract")
    _add_baseline_args(bl_extract)
    bl_extract.set_defaults(func=cmd_baseline_extract)

    bl_train_fold = baseline_sub.add_parser("train-fold")
    _add_baseline_args(bl_train_fold)
    bl_train_fold.set_defaults(func=cmd_baseline_train_fold)

    bl_train = baseline_sub.add_parser("train-cv")
    _add_baseline_args(bl_train)
    bl_train.set_defaults(func=cmd_baseline_train_cv)

    bl_report = baseline_sub.add_parser("report-cv")
    _add_baseline_args(bl_report)
    bl_report.set_defaults(func=cmd_baseline_report_cv)

    bl_pipeline = baseline_sub.add_parser("pipeline")
    _add_baseline_args(bl_pipeline)
    bl_pipeline.set_defaults(func=cmd_baseline_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

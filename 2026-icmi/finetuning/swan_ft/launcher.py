from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import RunSpec
from .paths import resolver_for


def _quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _base_cli_command(run_spec: RunSpec, area: str, action: str, extra: list[str]) -> list[str]:
    command = [
        run_spec.profile.python_bin,
        "-m",
        "swan_ft",
        area,
        action,
        "--profile",
        run_spec.profile.id,
        "--dataset-spec",
        run_spec.dataset_spec.id,
    ]
    if run_spec.model_spec is not None:
        command += ["--model", run_spec.model_spec.id]
    for fold_id in run_spec.options.folds:
        command += ["--folds", str(fold_id)]
    if run_spec.options.role_mode:
        command += ["--role-mode", run_spec.options.role_mode]
    for label in run_spec.options.exclude_labels:
        command += ["--exclude-label", label]
    if run_spec.options.overwrite:
        command.append("--overwrite")
    command += extra
    return command


@dataclass
class LocalLauncher:
    run_spec: RunSpec

    def run(self, area: str, action: str, extra: list[str] | None = None) -> int:
        command = _base_cli_command(self.run_spec, area, action, extra or [])
        command.append("--execute-local")
        env = os.environ.copy()
        if self.run_spec.profile.cache_root is not None:
            env["HF_HOME"] = str(self.run_spec.profile.cache_root)
        return subprocess.run(command, check=False, env=env).returncode


@dataclass
class SlurmLauncher:
    run_spec: RunSpec

    def _sbatch_args(self, job_name: str, dependency: str | None = None) -> list[str]:
        slurm = self.run_spec.profile.slurm
        resolver = resolver_for(self.run_spec)
        args = ["sbatch", "--parsable", f"--job-name={job_name}"]
        if slurm.partition:
            args.append(f"--partition={slurm.partition}")
        if slurm.account:
            args.append(f"--account={slurm.account}")
        if slurm.qos:
            args.append(f"--qos={slurm.qos}")
        if slurm.gres:
            args.append(f"--gres={slurm.gres}")
        if slurm.cpus_per_task:
            args.append(f"--cpus-per-task={slurm.cpus_per_task}")
        if slurm.mem:
            args.append(f"--mem={slurm.mem}")
        if slurm.time:
            args.append(f"--time={slurm.time}")
        if dependency:
            args.append(f"--dependency=afterok:{dependency}")
        output_path = resolver.logs_root() / f"{job_name}_%j.log"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args.append(f"--output={output_path}")
        args.extend(slurm.extra_sbatch_args)
        return args

    def _render_script(self, job_name: str, command: list[str]) -> Path:
        resolver = resolver_for(self.run_spec)
        path = resolver.slurm_script(job_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["#!/bin/bash", "set -euo pipefail"]
        if self.run_spec.profile.conda_activate:
            lines.append(f"source {shlex.quote(self.run_spec.profile.conda_activate)}")
            lines.append(f"conda activate {shlex.quote(self.run_spec.profile.env_name)}")
        if self.run_spec.profile.cache_root is not None:
            lines.append(f"export HF_HOME={shlex.quote(str(self.run_spec.profile.cache_root))}")
            lines.append("mkdir -p \"$HF_HOME\"")
        lines.append(_quote(command + ["--execute-local"]))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def submit(self, area: str, action: str, extra: list[str] | None = None, dependency: str | None = None) -> str:
        job_name = f"swan_{area}_{action}"
        command = _base_cli_command(self.run_spec, area, action, extra or [])
        script_path = self._render_script(job_name, command)
        result = subprocess.run(self._sbatch_args(job_name, dependency=dependency) + [str(script_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip()


def launcher_for(run_spec: RunSpec):
    if run_spec.options.execute_local or run_spec.profile.launcher == "local":
        return LocalLauncher(run_spec)
    if run_spec.profile.launcher == "slurm":
        return SlurmLauncher(run_spec)
    raise ValueError(f"Unsupported launcher: {run_spec.profile.launcher}")

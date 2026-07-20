#!/bin/bash

trim_whitespace() {
    local value="${1:-}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

swan_script_dir() {
    local source_path="${BASH_SOURCE[0]}"
    cd "$(dirname "$source_path")" && pwd
}

swan_ft_dir() {
    local script_dir
    script_dir="$(swan_script_dir)"
    cd "${script_dir}/.." && pwd
}

swan_repo_root() {
    local ft_dir
    ft_dir="$(swan_ft_dir)"
    cd "${ft_dir}/.." && pwd
}

swan_load_env() {
    local repo_root="${1:-$(swan_repo_root)}"
    local ft_dir
    ft_dir="$(swan_ft_dir)"
    local env_path
    for env_path in "${repo_root}/.env" "${ft_dir}/.env"; do
        if [ -f "$env_path" ]; then
            local temp_root sanitized_env
            temp_root="${TMPDIR:-${TEMP:-${TMP:-$(dirname "$env_path")}}}"
            mkdir -p "$temp_root"
            sanitized_env="$(mktemp "$temp_root/swan_env.XXXXXX")"
            tr -d '\r' < "$env_path" > "$sanitized_env"
            set -a
            source "$sanitized_env"
            set +a
            rm -f "$sanitized_env"
        fi
    done
}

swan_profile_token() {
    local profile="${1:?profile id required}"
    printf '%s' "$profile" | tr '[:lower:]-' '[:upper:]_'
}

swan_profile_value() {
    local profile="${1:?profile id required}"
    local suffix="${2:?env suffix required}"
    local token profile_var global_var
    token="$(swan_profile_token "$profile")"
    profile_var="SWAN_${token}_${suffix}"
    global_var="SWAN_${suffix}"
    if [ -n "${!profile_var:-}" ]; then
        printf '%s' "${!profile_var}"
        return 0
    fi
    if [ -n "${!global_var:-}" ]; then
        printf '%s' "${!global_var}"
    fi
}

swan_logs_root() {
    local profile="${1:?profile id required}"
    local ft_dir="${2:-$(swan_ft_dir)}"
    local value
    value="$(swan_profile_value "$profile" "LOGS_ROOT")"
    if [ -n "$value" ]; then
        printf '%s' "$value"
        return 0
    fi
    printf '%s' "${ft_dir}/logs/${profile}"
}

swan_conda_activate() {
    local profile="${1:?profile id required}"
    local value
    value="$(swan_profile_value "$profile" "CONDA_ACTIVATE")"
    if [ -n "$value" ]; then
        printf '%s' "$value"
        return 0
    fi
    printf '%s' "/mnt/data/miniforge3/etc/profile.d/conda.sh"
}

swan_python_bin() {
    local profile="${1:?profile id required}"
    local value
    value="$(swan_profile_value "$profile" "PYTHON_BIN")"
    if [ -n "$value" ]; then
        printf '%s' "$value"
        return 0
    fi
    printf '%s' "python"
}

swan_cache_root() {
    local profile="${1:?profile id required}"
    swan_profile_value "$profile" "CACHE_ROOT"
}

swan_temp_root() {
    local profile="${1:?profile id required}"
    local ft_dir="${2:-$(swan_ft_dir)}"
    local value
    value="$(swan_profile_value "$profile" "TEMP_ROOT")"
    if [ -n "$value" ]; then
        printf '%s' "$value"
        return 0
    fi
    printf '%s' "${ft_dir}/tmp/${profile}"
}

swan_split_csv() {
    local input="${1:-}"
    local -n out="$2"
    out=()
    local normalized="${input//,/ }"
    local item trimmed
    for item in $normalized; do
        trimmed="$(trim_whitespace "$item")"
        if [ -n "$trimmed" ]; then
            out+=("$trimmed")
        fi
    done
}


swan_expected_torchcodec_version() {
    python - <<'PY'
import re
try:
    import torch
except Exception:
    print("")
    raise SystemExit(0)
version = re.match(r"(\d+\.\d+)", getattr(torch, "__version__", ""))
version = version.group(1) if version else ""
mapping = {
    "2.4": "0.0.3",
    "2.5": "0.1",
    "2.6": "0.2",
    "2.7": "0.5",
    "2.8": "0.7",
    "2.9": "0.9",
    "2.10": "0.10",
    "2.11": "0.11",
}
print(mapping.get(version, ""))
PY
}

swan_install_compatible_torchcodec() {
    local expected_version
    expected_version="$(swan_expected_torchcodec_version)"
    if [ -n "$expected_version" ]; then
        echo "Installing torchcodec==${expected_version} to match the installed torch version..."
        python -m pip install --no-user --upgrade --force-reinstall "torchcodec==${expected_version}"
    else
        echo "Installing torchcodec without an explicit version pin because the torch version could not be detected..."
        python -m pip install --no-user --upgrade torchcodec
    fi
}

swan_add_filter_args() {
    local -n out="$1"
    local role_mode="${2:-}"
    local exclude_csv="${3:-}"
    if [ -n "$role_mode" ]; then
        out+=(--role-mode "$role_mode")
    fi
    if [ -n "$exclude_csv" ]; then
        local labels=()
        local label
        swan_split_csv "$exclude_csv" labels
        for label in "${labels[@]}"; do
            out+=(--exclude-label "$label")
        done
    fi
}

swan_add_fold_args() {
    local -n out="$1"
    local folds_csv="${2:-}"
    if [ -z "$folds_csv" ]; then
        return 0
    fi
    local folds=()
    swan_split_csv "$folds_csv" folds
    if [ "${#folds[@]}" -gt 0 ]; then
        out+=(--folds "${folds[@]}")
    fi
}


swan_log_runtime() {
    local profile="${1:?profile id required}"
    local ft_dir="${2:-$(swan_ft_dir)}"
    local temp_root cache_root logs_root data_root output_root session_root
    temp_root="$(swan_temp_root "$profile" "$ft_dir")"
    cache_root="$(swan_cache_root "$profile")"
    logs_root="$(swan_logs_root "$profile" "$ft_dir")"
    data_root="$(swan_profile_value "$profile" "DATA_ROOT")"
    output_root="$(swan_profile_value "$profile" "OUTPUT_ROOT")"
    session_root="$(swan_profile_value "$profile" "SESSION_ROOT")"

    echo "Runtime paths:"
    echo "  FT_DIR: ${ft_dir}"
    echo "  TMPDIR: ${TMPDIR:-unset}"
    echo "  Session root: ${session_root:-unset}"
    echo "  Data root: ${data_root:-unset}"
    echo "  Output root: ${output_root:-unset}"
    echo "  Logs root: ${logs_root:-unset}"
    echo "  Temp root: ${temp_root:-unset}"
    echo "  HF_HOME: ${HF_HOME:-${cache_root:-unset}}"
    echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-unset}"
    echo "  HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE:-unset}"
    echo "  TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE:-unset}"
    echo "  PYTHONNOUSERSITE: ${PYTHONNOUSERSITE:-unset}"
    echo "  PIP_CACHE_DIR: ${PIP_CACHE_DIR:-unset}"
}

swan_log_conda_envs() {
    echo "Available conda environments:"
    if ! conda env list; then
        echo "  <unable to list conda environments>"
    fi
}

swan_setup_runtime() {
    local profile="${1:?profile id required}"
    local ft_dir="${2:-$(swan_ft_dir)}"
    local conda_activate
    conda_activate="$(swan_conda_activate "$profile")"
    if [ ! -f "$conda_activate" ]; then
        echo "ERROR: Conda activation script not found: $conda_activate" >&2
        return 1
    fi

    source "$conda_activate"
    swan_log_conda_envs

    local env_name="${SWAN_CONDA_ENV_NAME:-swan2}"
    local base_env_name="${SWAN_BASE_CONDA_ENV_NAME:-swan}"
    local sync_requirements="${SWAN_SYNC_REQUIREMENTS:-0}"
    local temp_root lock_file pip_cache_dir requirements_hash stamp_path
    temp_root="$(swan_temp_root "$profile" "$ft_dir")"
    mkdir -p "$temp_root"
    export TMPDIR="$temp_root"
    export PYTHONNOUSERSITE=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_INPUT=1
    pip_cache_dir="$temp_root/pip-cache"
    mkdir -p "$pip_cache_dir"
    export PIP_CACHE_DIR="$pip_cache_dir"
    requirements_hash="$(sha256sum "$ft_dir/requirements.txt" | awk '{print $1}')"
    stamp_path="$temp_root/${env_name}.requirements.sha256"
    lock_file="$temp_root/swan_env_setup.lock"

    (
        flock -w 1800 200 || { echo "ERROR: Timed out waiting for env setup lock" >&2; exit 1; }
        if ! conda env list | grep -q "^${env_name} "; then
            if [ "$env_name" != "$base_env_name" ] && conda env list | grep -q "^${base_env_name} "; then
                echo "Creating ${env_name} by cloning ${base_env_name}..."
                conda create --clone "${base_env_name}" -n "${env_name}" -y
            else
                echo "Creating ${env_name} conda environment..."
                conda create -n "${env_name}" python=3.12 -y
                conda activate "${env_name}"
                python -m pip install -qq --no-user -U -r "${ft_dir}/requirements.txt"
            fi
            conda activate "${env_name}"
            printf '%s' "$requirements_hash" > "$stamp_path"
        else
            conda activate "${env_name}"
            if [ "$sync_requirements" = "1" ]; then
                echo "Synchronizing ${env_name} dependencies from requirements.txt..."
                python -m pip install -qq --no-user -U -r "${ft_dir}/requirements.txt"
                printf '%s' "$requirements_hash" > "$stamp_path"
            fi
        fi
    ) 200>"$lock_file"

    conda activate "${env_name}"

    local skip_torchcodec_check="${SWAN_SKIP_TORCHCODEC_CHECK:-0}"
    if [ "$skip_torchcodec_check" != "1" ] && ! python -c "import torchcodec" >/dev/null 2>&1; then
        (
            flock -w 1800 201 || { echo "ERROR: Timed out waiting for torchcodec repair lock" >&2; exit 1; }
            conda activate "${env_name}"
            if ! python -c "import torchcodec" >/dev/null 2>&1; then
                if [ "$env_name" != "$base_env_name" ] && conda env list | grep -q "^${base_env_name} "; then
                    echo "Recreating ${env_name} by cloning ${base_env_name} to restore missing dependencies..."
                    conda deactivate || true
                    conda activate base || true
                    conda env remove -n "${env_name}" -y || true
                    conda create --clone "${base_env_name}" -n "${env_name}" -y
                    conda activate "${env_name}"
                fi
            fi
            if ! python -c "import torchcodec" >/dev/null 2>&1; then
                swan_install_compatible_torchcodec
            fi
        ) 201>"$lock_file"
        conda activate "${env_name}"
    fi

    if [ "$skip_torchcodec_check" = "1" ]; then
        echo "Skipping torchcodec import check because SWAN_SKIP_TORCHCODEC_CHECK=1"
    elif ! python - <<'PY'
import traceback
try:
    import torchcodec  # noqa: F401
except Exception:
    traceback.print_exc()
    raise
PY
    then
        echo "ERROR: torchcodec is installed but failed to import in conda env '${env_name}'." >&2
        echo "Set SWAN_SKIP_TORCHCODEC_CHECK=1 only if you know this run path does not require torchcodec." >&2
        return 1
    fi

    if ! python -m pip check >/dev/null 2>&1; then
        echo "ERROR: pip dependency check failed in conda env '${env_name}'." >&2
        return 1
    fi

    local cache_root datasets_cache_root
    cache_root="$(swan_cache_root "$profile")"
    datasets_cache_root="$temp_root/hf_datasets_cache"
    export HF_DATASETS_CACHE="$datasets_cache_root"
    mkdir -p "$HF_DATASETS_CACHE"

    if [ -n "$cache_root" ]; then
        export HF_HOME="$cache_root"
        export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
        export TRANSFORMERS_CACHE="$HF_HOME/transformers"
        mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
    fi

    # Cluster downloads can be slow on first pull; use longer Hub timeouts by default.
    export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

    export HF_TOKEN="${HF_TOKEN:-<YOUR_HF_TOKEN>}"
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
}

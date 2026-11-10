#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

usage() {
    cat <<'EOF'
Submit from the cluster login node:
  bash benchmarks/kernels/run_muse_spark_soft_cap_slurm.sh submit \
    /shared/path/to/results GPU_TYPE [GPU_TYPE ...] -- [SBATCH_OPTIONS ...]

GPU_TYPE must match the cluster's GRES names (see sinfo -o '%P %G').
Options after -- apply to every job, for example --partition=gpu --account=lab.
For GPU types in different partitions, invoke submit separately for each group.
The checkout and its .venv must be accessible from the compute nodes.
Use a results directory on persistent shared storage, outside the checkout.
Keep the checkout unchanged until all submitted jobs finish.
EOF
}

submit() {
    if (( $# < 2 )); then
        usage >&2
        return 2
    fi
    local results_root=$1
    shift
    local -a gpu_types=()
    while (( $# )) && [[ $1 != -- ]]; do
        if [[ ! $1 =~ ^[[:alnum:]_.-]+$ ]]; then
            echo "Invalid GPU type: $1" >&2
            return 2
        fi
        gpu_types+=("$1")
        shift
    done
    if [[ ${1:-} == -- ]]; then
        shift
    fi
    if (( ${#gpu_types[@]} == 0 )); then
        usage >&2
        return 2
    fi

    local repo script commit submission_dir gpu gpu_dir job_id
    repo=$(git rev-parse --show-toplevel)
    script=$(realpath "${BASH_SOURCE[0]}")
    commit=$(git -C "$repo" rev-parse HEAD)
    if [[ ! -x $repo/.venv/bin/python ]]; then
        echo "Set up $repo/.venv before submitting jobs." >&2
        return 1
    fi
    if [[ -n $(git -C "$repo" status --porcelain) ]]; then
        echo "Commit your changes before submitting reproducible runs." >&2
        return 1
    fi
    results_root=$(realpath -m "$results_root")
    if [[ $results_root == "$repo" || $results_root == "$repo/"* ]]; then
        echo "Choose a results directory outside the Git checkout." >&2
        return 2
    fi
    mkdir -p "$results_root"
    submission_dir=$(mktemp -d "$results_root/$(date -u +%Y%m%dT%H%M%SZ)-${commit:0:12}-XXXXXX")
    printf 'gpu_type\tjob_id\n' > "$submission_dir/jobs.tsv"
    for gpu in "${gpu_types[@]}"; do
        gpu_dir=$submission_dir/$gpu
        mkdir -p "$gpu_dir"
        job_id=$(sbatch --parsable \
            --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
            "$@" \
            --gres="gpu:$gpu:1" \
            --job-name="soft-cap-$gpu" \
            --output="$gpu_dir/slurm-%j.out" \
            --error="$gpu_dir/slurm-%j.err" \
            --chdir="$repo" \
            "$script" run "$repo" "$gpu_dir" "$gpu" "$commit")
        printf '%s\t%s\n' "$gpu" "$job_id" >> "$submission_dir/jobs.tsv"
        printf 'Submitted %s: %s\n' "$gpu" "$job_id"
    done
    printf 'Results: %s\n' "$submission_dir"
}

run_logged() {
    local log=$1
    shift
    printf '%q ' "$@" >> "$run_dir/commands.txt"
    printf '\n' >> "$run_dir/commands.txt"
    "$@" 2>&1 | tee "$run_dir/$log"
}

run_job() {
    local repo=$1 gpu_dir=$2 gpu_type=$3 expected_commit=$4
    : "${SLURM_JOB_ID:?Run the submit command from the login node.}"
    run_dir=$gpu_dir/$SLURM_JOB_ID
    mkdir -p "$run_dir"
    trap 'printf "%s\n" "$?" > "$run_dir/exit-code.txt"' EXIT
    cd "$repo"
    if [[ $(git rev-parse HEAD) != "$expected_commit" ]] || \
        [[ -n $(git status --porcelain) ]]; then
        echo "The checkout changed after submission. Commit and resubmit." >&2
        exit 1
    fi
    module purge
    module load cuda
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    git status --short --branch > "$run_dir/git-status.txt"
    git log -1 --format=fuller > "$run_dir/commit.txt"
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/slurm-job.txt"
    nvidia-smi > "$run_dir/nvidia-smi.txt"
    printf 'Results: %s\n' "$run_dir"

    run_logged preflight.log .venv/bin/python - "$run_dir/metadata.json" "$gpu_type" <<'PY'
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from vllm.models.muse_spark.nvidia.ops.cute_dsl.mul_tanh_soft_cap import HAS_CUTEDSL

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one visible CUDA GPU in the allocation")
if not HAS_CUTEDSL:
    raise RuntimeError("CuTe DSL dependencies are missing; tests would be skipped")

versions = {}
for package in (
    "torch", "triton", "nvidia-cutlass-dsl", "flashinfer-python", "pytest", "vllm"
):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None

metadata = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "hostname": platform.node(),
    "python": sys.version,
    "python_executable": sys.executable,
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "requested_gpu_type": sys.argv[2],
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": torch.cuda.get_device_capability(0),
    "torch_cuda": torch.version.cuda,
    "versions": versions,
    "environment": {
        name: os.environ.get(name)
        for name in (
            "SLURM_JOB_ID", "SLURM_JOB_PARTITION",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"
        )
    },
    "timing": "Existing benchmark: warmed CUDA graphs, reused buffers, 5 repeats",
}
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
PY

    run_logged pytest.log .venv/bin/python -m pytest \
        tests/kernels/test_muse_spark_mul_tanh_soft_cap.py \
        -v -ra --junitxml="$run_dir/pytest.xml"
    run_logged benchmark.log .venv/bin/python \
        benchmarks/kernels/benchmark_muse_spark_mul_tanh_soft_cap.py \
        --rows 1 8 32 --width 202048 202049 --dtype all --repeats 5
    if [[ $(git rev-parse HEAD) != "$expected_commit" ]] || \
        [[ -n $(git status --porcelain) ]]; then
        echo "The checkout changed during the run; discard these results." >&2
        exit 1
    fi
}

case ${1:-} in
    submit) shift; submit "$@" ;;
    run) shift; run_job "$@" ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac

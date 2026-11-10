# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark or profile native CUDA and CuTe DSL MuseSpark soft-capping.

Run from the vLLM repository root. Correctness coverage lives in
tests/kernels/test_muse_spark_mul_tanh_soft_cap.py.
"""

import argparse
import statistics

import torch

import vllm._custom_ops  # noqa: F401
from vllm.models.muse_spark.nvidia.ops.cute_dsl.mul_tanh_soft_cap import (
    muse_spark_mul_tanh_soft_cap,
)

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
MODEL_MULTIPLIER = 0.17677669529663687
MODEL_TEMPERATURE = 20.0


def benchmark_case(
    rows: int,
    width: int,
    dtype_name: str,
    mode: str,
    repeats: int,
) -> None:
    dtype = DTYPES[dtype_name]
    x = torch.randn((rows, width), device="cuda", dtype=dtype)
    out_cuda = torch.empty_like(x, dtype=torch.float32)
    out_cute = torch.full_like(x, float("nan"), dtype=torch.float32)

    def run_cuda() -> None:
        torch.ops._C.muse_spark_mul_tanh_soft_cap(
            out_cuda, x, MODEL_MULTIPLIER, MODEL_TEMPERATURE
        )

    def run_cute() -> None:
        muse_spark_mul_tanh_soft_cap(out_cute, x, MODEL_MULTIPLIER, MODEL_TEMPERATURE)

    # Compile and warm both implementations before timing or profiling.
    for _ in range(10):
        run_cuda()
        run_cute()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out_cute, out_cuda)

    label = f"({rows}, {width}) {dtype_name}"
    if mode == "profile":
        print(f"Profiling {label}: native CUDA first, CuTe second", flush=True)
        torch.cuda.cudart().cudaProfilerStart()
        run_cuda()
        run_cute()
        torch.accelerator.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    from triton.testing import do_bench_cudagraph

    cuda_times = []
    cute_times = []
    # Alternate measurement order to reduce systematic order bias.
    for repeat in range(repeats):
        if repeat % 2 == 0:
            cuda_times.append(do_bench_cudagraph(run_cuda, rep=100) * 1000)
            cute_times.append(do_bench_cudagraph(run_cute, rep=100) * 1000)
        else:
            cute_times.append(do_bench_cudagraph(run_cute, rep=100) * 1000)
            cuda_times.append(do_bench_cudagraph(run_cuda, rep=100) * 1000)

    cuda_us = statistics.median(cuda_times)
    cute_us = statistics.median(cute_times)
    print(
        f"{label:26} "
        f"CUDA={cuda_us:.3f} us  CuTe={cute_us:.3f} us  "
        f"speedup={cuda_us / cute_us:.3f}x"
    )
    print(
        f"  ranges across {repeats} measurements: "
        f"CUDA={min(cuda_times):.3f}-{max(cuda_times):.3f} us, "
        f"CuTe={min(cute_times):.3f}-{max(cute_times):.3f} us"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--rows", type=int, nargs="+", default=None)
    parser.add_argument("--width", type=int, nargs="+", default=[202048])
    parser.add_argument("--dtype", choices=(*DTYPES, "all"), default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    rows = (
        args.rows
        if args.rows is not None
        else ([8] if args.mode == "profile" else [1, 8, 32])
    )
    dtype_name = args.dtype or ("fp32" if args.mode == "profile" else "all")
    dtype_names = list(DTYPES) if dtype_name == "all" else [dtype_name]

    if any(width < 1 for width in args.width) or any(row < 1 for row in rows):
        parser.error("rows and widths must be positive")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.mode == "profile" and (
        len(rows) != 1 or len(args.width) != 1 or len(dtype_names) != 1
    ):
        parser.error("profile mode requires one row count, one width, and one dtype")

    torch.manual_seed(0)
    for width in args.width:
        for row in rows:
            for name in dtype_names:
                benchmark_case(row, width, name, args.mode, args.repeats)


if __name__ == "__main__":
    main()

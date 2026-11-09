# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from typing import Any

from vllm._custom_ops import (
    muse_spark_mul_tanh_soft_cap as original_soft_cap
)

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32
    from cutlass.cute.runtime import make_ptr
    from cutlass.cute.math import tanh
    from cuda.bindings import driver as cuda
    HAS_CUTEDSL: bool = True
except ImportError:
    HAS_CUTEDSL: bool = False


_TORCH_TO_CUTLASS_DTYPE = (
    {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    if HAS_CUTEDSL
    else {}
)

MUSE_SOFT_CAP_VEC_SIZE = 4

if HAS_CUTEDSL:
    class MulTanhSoftCapKernel:
        @cute.jit
        def __call__(
            self,
            out_ptr: cute.Pointer,
            input_ptr: cute.Pointer,
            d: Int32,
            num_tokens: Int32,
            blocks_per_row: Int32,
            multiplier: Float32,
            soft_cap_temp: Float32,
            stream: cuda.CUstream
        ) -> None:
            d = cute.assume(d, divby=4)
            layout: cute.Layout = cute.make_layout(
                shape=(num_tokens, d),
                stride=(d, 1)
            )
            input_tensor: cute.Tensor = cute.make_tensor(input_ptr, layout)
            out_tensor: cute.Tensor = cute.make_tensor(out_ptr, layout)

            self.muse_spark_mul_tanh_soft_cap_kernel(
                out_tensor,
                input_tensor,
                d,
                multiplier,
                soft_cap_temp
            ).launch(
                grid=(num_tokens, blocks_per_row, 1),
                block=(128, 1, 1),
                stream=stream
            )

        @cute.kernel
        def muse_spark_mul_tanh_soft_cap_kernel(
            self,
            out: cute.Tensor,
            input: cute.Tensor,
            d: Int32,
            multiplier: Float32,
            soft_cap_temp: Float32
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            bdimx, _, _ = cute.arch.block_dim()
            _, gdimy, _ = cute.arch.grid_dim()

            tid: Int32 = bidy * bdimx + tidx
            stride: Int32 = gdimy * bdimx

            load_atom: cute.CopyAtom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                input.element_type,
                num_bits_per_copy=4 * input.element_type.width,
            )
            store_atom: cute.CopyAtom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=128
            )

            loaded: cute.Tensor = cute.make_rmem_tensor(
                cute.make_layout(4), input.element_type
            )
            results: cute.Tensor = cute.make_rmem_tensor(
                cute.make_layout(4), Float32
            )

            input_row: cute.Tensor = input[bidx, None]
            out_row: cute.Tensor = out[bidx, None]

            for group in range(tid, d // 4, stride):
                input4: cute.Tensor = cute.local_tile(input_row, (4,), (group,))
                out4: cute.Tensor = cute.local_tile(out_row, (4,), (group,))
                cute.copy(load_atom, input4, loaded)

                for i in cutlass.range_constexpr(4):
                    x: Float32 = loaded[i]
                    results[i] = soft_cap_temp * tanh(
                        x * multiplier / soft_cap_temp,
                        fastmath=False
                    )
                cute.copy(store_atom, results, out4)


    _compile_cache: dict[Any, Any] = {}

    def _compiled_kernel(dtype: Any, stream: cuda.CUstream) -> Any:
        key: tuple[Any, Any] = (dtype, torch.cuda.current_device())
        compiled: Any | None = _compile_cache.get(key)
        if compiled is None:
            compiled = cute.compile(
                MulTanhSoftCapKernel(),
                make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                # make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                # make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                Int32(1),
                Int32(1),
                Int32(1),
                Float32(1.0),
                Float32(1.0),
                stream,
            )
            _compile_cache[key] = compiled
        return compiled


def muse_spark_mul_tanh_soft_cap(
    out: torch.Tensor,
    input: torch.Tensor,
    multiplier: float,
    soft_cap_temp: float
) -> None:
    if not HAS_CUTEDSL:
        raise RuntimeError("This operation requires nvidia-cutlass-dsl")

    if not input.is_cuda or not out.is_cuda:
        raise RuntimeError("input and out must be CUDA tensors")

    if input.device != out.device:
        raise RuntimeError("input and out must be on the same CUDA device")

    if input.dim() == 0:
        raise RuntimeError("input must have at least one dimension")

    if input.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise RuntimeError(
            "muse_spark_mul_tanh_soft_cap: input must be float32, float16 or "
            "bfloat16"
        )

    if out.dtype != torch.float32:
        raise RuntimeError("muse_spark_mul_tanh_soft_cap: out must be float32")

    if not input.is_contiguous():
        raise RuntimeError(
            "muse_spark_mul_tanh_soft_cap: input must be contiguous"
        )

    if not out.is_contiguous():
        raise RuntimeError(
            "muse_spark_mul_tanh_soft_cap: out must be contiguous"
        )

    if input.numel() != out.numel():
        raise RuntimeError(
            "muse_spark_mul_tanh_soft_cap: out and input must have the same "
            "number of elements"
        )

    if soft_cap_temp == 0.0:
        raise RuntimeError(
            "muse_spark_mul_tanh_soft_cap: soft_cap_temp must be nonzero"
        )

    if input.numel() == 0:
        return

    d: int = input.shape[-1]
    num_tokens: int = input.numel() // d

    assert out.dtype == torch.float32
    assert input.data_ptr() % 16 == 0
    assert out.data_ptr() % 16 == 0
    assert d % 4 == 0

    kThreads: int = 128
    vecs_per_row: int = (d + MUSE_SOFT_CAP_VEC_SIZE - 1) // MUSE_SOFT_CAP_VEC_SIZE
    blocks_to_cover: int = (vecs_per_row + kThreads - 1) // kThreads
    kTargetBlocks: int = 32768
    blocks_per_row: int = min(
        max(kTargetBlocks // num_tokens, 1),
        max(blocks_to_cover, 1)
    )

    with torch.cuda.device(input.device):
        dtype: Any = _TORCH_TO_CUTLASS_DTYPE[input.dtype]
        stream: cuda.CUstream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled: Any = _compiled_kernel(dtype, stream)
        compiled(
            make_ptr(Float32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(dtype, input.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            # make_ptr(Float32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            # make_ptr(Float32, input.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
            Int32(d),
            Int32(num_tokens),
            Int32(blocks_per_row),
            Float32(multiplier),
            Float32(soft_cap_temp),
            stream
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("check", "bench", "profile"),
        default="check",
    )
    args = parser.parse_args()

    shapes = [
        (128,),          # Small, single row
        (3, 128),        # Multiple rows
        (2, 4096),       # Intermediate width
        (1, 202048),     # Model vocabulary width
        (8, 202048),
        (32, 202048),
    ]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]

    if args.mode == "profile":
        cases = [((8, 202048), torch.float32)]
    else:
        cases = [
            (shape, dtype)
            for shape in shapes
            for dtype in dtypes
        ]

    multiplier = 0.17677669529663687
    soft_cap_temp = 20.0

    if args.mode == "bench":
        from triton.testing import do_bench_cudagraph

    for shape, dtype in cases:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        out_cuda = torch.empty_like(x, dtype=torch.float32)
        out_cute = torch.empty_like(x, dtype=torch.float32)

        def run_cuda():
            torch.ops._C.muse_spark_mul_tanh_soft_cap(
                out_cuda, x, multiplier, soft_cap_temp
            )

        def run_cute():
            muse_spark_mul_tanh_soft_cap(
                out_cute, x, multiplier, soft_cap_temp
            )

        for _ in range(10):
            run_cuda()
            run_cute()

        torch.cuda.synchronize()

        expected = soft_cap_temp * torch.tanh(
            x.float() * multiplier / soft_cap_temp
        )
        torch.testing.assert_close(out_cute, out_cuda)
        torch.testing.assert_close(out_cute, expected)

        label = f"{str(shape):14} {str(dtype):16}"

        if args.mode == "check":
            print(f"{label} passed")

        elif args.mode == "bench":
            cuda_ms = do_bench_cudagraph(run_cuda, rep=100)
            cute_ms = do_bench_cudagraph(run_cute, rep=100)

            print(
                f"{label} "
                f"CUDA={cuda_ms * 1000:.3f} us  "
                f"CuTe={cute_ms * 1000:.3f} us  "
                f"speedup={cuda_ms / cute_ms:.3f}x"
            )

        elif args.mode == "profile":
            torch.cuda.cudart().cudaProfilerStart()
            run_cuda()
            run_cute()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

import vllm._custom_ops  # noqa: F401
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

HAS_CUTEDSL: bool = True

try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass import Float32, Int32
    from cutlass.cute.math import tanh
    from cutlass.cute.runtime import make_ptr
except ImportError:
    HAS_CUTEDSL = False


_TORCH_TO_CUTLASS_DTYPE = (
    {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    if HAS_CUTEDSL
    else {}
)


THREADS_PER_BLOCK = 128
ELEMENTS_PER_VECTOR = 4
TARGET_BLOCKS = 32768
VECTOR_ALIGNMENT_BYTES = 16


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
            vectorized: cutlass.Constexpr,
            stream: cuda.CUstream,
        ) -> None:
            if cutlass.const_expr(vectorized):
                d = cute.assume(d, divby=ELEMENTS_PER_VECTOR)

            layout: cute.Layout = cute.make_layout((num_tokens, d), stride=(d, 1))
            input_tensor: cute.Tensor = cute.make_tensor(input_ptr, layout)
            out_tensor: cute.Tensor = cute.make_tensor(out_ptr, layout)
            if cutlass.const_expr(vectorized):
                self.muse_spark_mul_tanh_soft_cap_kernel_vectorized(
                    out_tensor, input_tensor, d, multiplier, soft_cap_temp
                ).launch(
                    grid=(num_tokens, blocks_per_row, 1),
                    block=(THREADS_PER_BLOCK, 1, 1),
                    stream=stream,
                )
            else:
                self.muse_spark_mul_tanh_soft_cap_kernel_scalar(
                    out_tensor, input_tensor, d, multiplier, soft_cap_temp
                ).launch(
                    grid=(num_tokens, blocks_per_row, 1),
                    block=(THREADS_PER_BLOCK, 1, 1),
                    stream=stream,
                )

        @cute.kernel
        def muse_spark_mul_tanh_soft_cap_kernel_vectorized(
            self,
            out: cute.Tensor,
            input: cute.Tensor,
            d: Int32,
            multiplier: Float32,
            soft_cap_temp: Float32,
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
                num_bits_per_copy=ELEMENTS_PER_VECTOR * input.element_type.width,
            )
            store_atom: cute.CopyAtom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )

            loaded: cute.Tensor = cute.make_rmem_tensor(
                cute.make_layout(ELEMENTS_PER_VECTOR), input.element_type
            )
            results: cute.Tensor = cute.make_rmem_tensor(
                cute.make_layout(ELEMENTS_PER_VECTOR), Float32
            )

            input_row: cute.Tensor = input[bidx, None]
            out_row: cute.Tensor = out[bidx, None]

            for group in range(tid, d // ELEMENTS_PER_VECTOR, stride):
                input4: cute.Tensor = cute.local_tile(
                    input_row, (ELEMENTS_PER_VECTOR,), (group,)
                )
                out4: cute.Tensor = cute.local_tile(
                    out_row, (ELEMENTS_PER_VECTOR,), (group,)
                )

                cute.copy(load_atom, input4, loaded)
                for i in cutlass.range_constexpr(ELEMENTS_PER_VECTOR):
                    x: Float32 = loaded[i]
                    results[i] = soft_cap_temp * tanh(
                        x * multiplier / soft_cap_temp, fastmath=False
                    )
                cute.copy(store_atom, results, out4)

        @cute.kernel
        def muse_spark_mul_tanh_soft_cap_kernel_scalar(
            self,
            out: cute.Tensor,
            input: cute.Tensor,
            d: Int32,
            multiplier: Float32,
            soft_cap_temp: Float32,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            bdimx, _, _ = cute.arch.block_dim()
            _, gdimy, _ = cute.arch.grid_dim()

            tid: Int32 = bidy * bdimx + tidx
            stride: Int32 = gdimy * bdimx
            for i in range(tid, d, stride):
                x: Float32 = input[bidx, i]
                out[bidx, i] = soft_cap_temp * tanh(
                    x * multiplier / soft_cap_temp, fastmath=False
                )

    _compile_cache: dict[Any, Any] = {}

    def _compiled_kernel(dtype: Any, vectorized: bool, stream: cuda.CUstream) -> Any:
        key: tuple[Any, bool, Any] = (
            dtype,
            vectorized,
            torch.accelerator.current_device_index(),
        )
        out_ptr: cute.Pointer
        input_ptr: cute.Pointer
        compiled: Any | None = _compile_cache.get(key)
        if compiled is None:
            if vectorized:
                # Vectorized launch
                out_ptr = make_ptr(
                    Float32,
                    VECTOR_ALIGNMENT_BYTES,
                    cute.AddressSpace.gmem,
                    assumed_align=VECTOR_ALIGNMENT_BYTES,
                )
                input_ptr = make_ptr(
                    dtype,
                    VECTOR_ALIGNMENT_BYTES,
                    cute.AddressSpace.gmem,
                    assumed_align=VECTOR_ALIGNMENT_BYTES,
                )
            else:
                # Scalar launch
                out_ptr = make_ptr(
                    Float32, VECTOR_ALIGNMENT_BYTES, cute.AddressSpace.gmem
                )
                input_ptr = make_ptr(
                    dtype, VECTOR_ALIGNMENT_BYTES, cute.AddressSpace.gmem
                )

            compiled = cute.compile(
                MulTanhSoftCapKernel(),
                out_ptr,
                input_ptr,
                Int32(1),  # d
                Int32(1),  # num_tokens
                Int32(1),  # blocks_per_row
                Float32(1.0),  # multiplier
                Float32(1.0),  # soft_cap_temp
                vectorized,
                stream,
            )
            _compile_cache[key] = compiled
        return compiled


def _validate_inputs(
    out: torch.Tensor, input: torch.Tensor, soft_cap_temp: float
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
            "muse_spark_mul_tanh_soft_cap: input must be float32, float16 or bfloat16"
        )

    if out.dtype != torch.float32:
        raise RuntimeError("muse_spark_mul_tanh_soft_cap: out must be float32")

    if not input.is_contiguous():
        raise RuntimeError("muse_spark_mul_tanh_soft_cap: input must be contiguous")

    if not out.is_contiguous():
        raise RuntimeError("muse_spark_mul_tanh_soft_cap: out must be contiguous")

    if input.numel() != out.numel():
        raise RuntimeError(
            "muse_spark_mul_tanh_soft_cap: out and input must have the same "
            "number of elements"
        )

    if soft_cap_temp == 0.0:
        raise RuntimeError(
            "muse_spark_mul_tanh_soft_cap: soft_cap_temp must be nonzero"
        )

    assert out.dtype == torch.float32


def muse_spark_mul_tanh_soft_cap(
    out: torch.Tensor, input: torch.Tensor, multiplier: float, soft_cap_temp: float
) -> None:
    _validate_inputs(out, input, soft_cap_temp)

    if input.numel() == 0:
        return

    d: int = input.shape[-1]
    num_tokens: int = input.numel() // d
    vectorized: bool = (
        d % ELEMENTS_PER_VECTOR == 0
        and input.data_ptr() % VECTOR_ALIGNMENT_BYTES == 0
        and out.data_ptr() % VECTOR_ALIGNMENT_BYTES == 0
    )
    elements_per_thread: int = 4 if vectorized else 1
    work_per_row: int = (d + elements_per_thread - 1) // elements_per_thread
    blocks_to_cover: int = (work_per_row + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    blocks_per_row: int = min(
        max(TARGET_BLOCKS // num_tokens, 1), max(blocks_to_cover, 1)
    )

    with torch.accelerator.device_index(input.device.index):
        dtype: Any = _TORCH_TO_CUTLASS_DTYPE[input.dtype]
        out_ptr: cute.Pointer
        input_ptr: cute.Pointer
        if vectorized:
            # Vectorized launch
            out_ptr = make_ptr(
                Float32,
                out.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=VECTOR_ALIGNMENT_BYTES,
            )
            input_ptr = make_ptr(
                dtype,
                input.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=VECTOR_ALIGNMENT_BYTES,
            )
        else:
            # Scalar launch
            out_ptr = make_ptr(
                Float32,
                out.data_ptr(),
                cute.AddressSpace.gmem,
            )
            input_ptr = make_ptr(
                dtype,
                input.data_ptr(),
                cute.AddressSpace.gmem,
            )

        stream: cuda.CUstream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled: Any = _compiled_kernel(dtype, vectorized, stream)
        compiled(
            out_ptr,
            input_ptr,
            Int32(d),
            Int32(num_tokens),
            Int32(blocks_per_row),
            Float32(multiplier),
            Float32(soft_cap_temp),
            stream,
        )


def _soft_cap_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    multiplier: float,
    soft_cap_temp: float,
) -> None:
    return


if current_platform.is_cuda() and HAS_CUTEDSL:
    direct_register_custom_op(
        op_name="muse_spark_mul_tanh_soft_cap_cutedsl",
        op_func=muse_spark_mul_tanh_soft_cap,
        fake_impl=_soft_cap_fake,
        mutates_args=["out"],
    )

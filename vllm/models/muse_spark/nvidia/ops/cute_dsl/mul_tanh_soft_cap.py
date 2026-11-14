# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

import vllm._custom_ops  # noqa: F401
from vllm.models.muse_spark.common.layers import MulTanhSoftCap
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

HAS_CUTEDSL: bool = True

try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass import Float32, Int32, Int64
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
TARGET_BLOCKS = 32768
VECTOR_ALIGNMENT_BYTES = 16
ELEMENTS_PER_VECTOR = 4
assert (
    ELEMENTS_PER_VECTOR > 0 and (ELEMENTS_PER_VECTOR & (ELEMENTS_PER_VECTOR - 1)) == 0
), "ELEMENTS_PER_VECTOR must be a positive power of two"


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
            kernel_kind: cutlass.Constexpr,
            stream: cuda.CUstream,
        ) -> None:
            grid = (num_tokens, blocks_per_row, 1)
            block = (THREADS_PER_BLOCK, 1, 1)

            if cutlass.const_expr(kernel_kind == "full_vectorized"):
                d_aligned = cute.assume(d, divby=ELEMENTS_PER_VECTOR)
                self.kernel_full_vectorized(
                    out_ptr, input_ptr, d_aligned, multiplier, soft_cap_temp
                ).launch(grid=grid, block=block, stream=stream)
            elif cutlass.const_expr(kernel_kind == "full_scalar"):
                self.kernel_full_scalar(
                    out_ptr, input_ptr, d, multiplier, soft_cap_temp
                ).launch(grid=grid, block=block, stream=stream)
            else:  # "prefix_vector_tail"
                self.kernel_scalar_vectorized(
                    out_ptr, input_ptr, d, multiplier, soft_cap_temp
                ).launch(grid=grid, block=block, stream=stream)

        @cute.kernel
        def kernel_full_vectorized(
            self,
            out: cute.Pointer,
            input: cute.Pointer,
            d: Int32,
            multiplier: Float32,
            soft_cap_temp: Float32,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            bdimx, _, _ = cute.arch.block_dim()
            _, gdimy, _ = cute.arch.grid_dim()

            input_dtype = input.dtype
            input_vector_bits = ELEMENTS_PER_VECTOR * input_dtype.width
            row_offset = Int64(bidx) * Int64(d)
            num_vec = d >> 2  # Four elements per vector.

            vec_layout = cute.make_layout(
                (ELEMENTS_PER_VECTOR, num_vec),
                stride=(1, ELEMENTS_PER_VECTOR),
            )

            input_row_ptr = cute.make_ptr(
                input_dtype,
                (input + row_offset).toint(),
                cute.AddressSpace.gmem,
                assumed_align=input_vector_bits // 8,
            )
            out_row_ptr = cute.make_ptr(
                Float32,
                (out + row_offset).toint(),
                cute.AddressSpace.gmem,
                assumed_align=VECTOR_ALIGNMENT_BYTES,
            )
            input_vectors = cute.make_tensor(input_row_ptr, vec_layout)
            out_vectors = cute.make_tensor(out_row_ptr, vec_layout)

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                input_dtype,
                num_bits_per_copy=input_vector_bits,
            )
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=ELEMENTS_PER_VECTOR * Float32.width,
            )

            fragment_layout = cute.make_layout(ELEMENTS_PER_VECTOR)
            loaded = cute.make_rmem_tensor(fragment_layout, input_dtype)
            results = cute.make_rmem_tensor(fragment_layout, Float32)

            group = bidy * bdimx + tidx
            stride = gdimy * bdimx
            while group < num_vec:
                cute.copy(load_atom, input_vectors[None, group], loaded)

                for j in cutlass.range_constexpr(ELEMENTS_PER_VECTOR):
                    x = Float32(loaded[j])
                    results[j] = soft_cap_temp * tanh(
                        x * multiplier / soft_cap_temp,
                        fastmath=False,
                    )

                cute.copy(store_atom, results, out_vectors[None, group])
                group = group + stride

        @cute.kernel
        def kernel_full_scalar(
            self,
            out: cute.Pointer,
            input: cute.Pointer,
            d: Int32,
            multiplier: Float32,
            soft_cap_temp: Float32,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            bdimx, _, _ = cute.arch.block_dim()
            _, gdimy, _ = cute.arch.grid_dim()

            row_offset = Int64(bidx) * Int64(d)
            row_layout = cute.make_layout(d)
            input_row = cute.make_tensor(input + row_offset, row_layout)
            out_row = cute.make_tensor(out + row_offset, row_layout)

            i = bidy * bdimx + tidx
            stride = gdimy * bdimx
            while i < d:
                x = Float32(input_row[i])
                out_row[i] = soft_cap_temp * tanh(
                    x * multiplier / soft_cap_temp, fastmath=False
                )
                i = i + stride

        @cute.kernel
        def kernel_scalar_vectorized(
            self,
            out: cute.Pointer,
            input: cute.Pointer,
            d: Int32,
            multiplier: Float32,
            soft_cap_temp: Float32,
        ) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            bdimx, _, _ = cute.arch.block_dim()
            _, gdimy, _ = cute.arch.grid_dim()

            tid = bidy * bdimx + tidx
            stride = gdimy * bdimx

            row_offset = Int64(bidx) * Int64(d)
            row_layout = cute.make_layout(d)
            input_row = cute.make_tensor(input + row_offset, row_layout)
            out_row = cute.make_tensor(out + row_offset, row_layout)

            input_row_addr = input_row.iterator.toint()
            out_row_addr = out_row.iterator.toint()

            input_element_bytes = input.dtype.width // 8
            out_element_bytes = out.dtype.width // 8
            in_vec_bytes = ELEMENTS_PER_VECTOR * input_element_bytes
            out_vec_bytes = ELEMENTS_PER_VECTOR * out_element_bytes

            # Find how many scalar elements precede the first aligned input vector.
            alignment_mask = in_vec_bytes - 1
            misalignment_bytes = Int32(input_row_addr & alignment_mask)
            skipped_bytes = (in_vec_bytes - misalignment_bytes) & alignment_mask
            skipped = min(skipped_bytes // input_element_bytes, d)

            # The same element offset must also align the output.
            out_addr_after_skip = out_row_addr + skipped * out_element_bytes

            # Process the scalar prefix, vectorized middle, and scalar tail.
            if tid < skipped:
                x = Float32(input_row[tid])
                out_row[tid] = soft_cap_temp * tanh(
                    x * multiplier / soft_cap_temp, fastmath=False
                )

            # Vectorized middle starts after the prefix.
            num_vec = Int32((d - skipped) >> 2)  # Four elements per vector.
            if num_vec > 0:
                in_vec_ptr = cute.make_ptr(
                    input.dtype,
                    input_row_addr + skipped * input_element_bytes,
                    cute.AddressSpace.gmem,
                    assumed_align=in_vec_bytes,
                )
                out_vec_ptr = cute.make_ptr(
                    out.dtype,
                    out_addr_after_skip,
                    cute.AddressSpace.gmem,
                    assumed_align=out_vec_bytes,
                )

                load_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    input.dtype,
                    num_bits_per_copy=in_vec_bytes * 8,
                )
                store_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    out.dtype,
                    num_bits_per_copy=out_vec_bytes * 8,
                )

                fragment_layout = cute.make_layout(ELEMENTS_PER_VECTOR)
                loaded = cute.make_rmem_tensor(fragment_layout, input.dtype)
                results = cute.make_rmem_tensor(fragment_layout, out.dtype)

                vectorized_elements = num_vec * ELEMENTS_PER_VECTOR
                element_stride = stride * ELEMENTS_PER_VECTOR
                element_offset = tid * ELEMENTS_PER_VECTOR
                while element_offset < vectorized_elements:
                    aligned_offset = cute.assume(
                        element_offset, divby=ELEMENTS_PER_VECTOR
                    )

                    input_pack = cute.make_tensor(
                        in_vec_ptr + aligned_offset, fragment_layout
                    )
                    cute.copy(load_atom, input_pack, loaded)

                    for j in cutlass.range_constexpr(ELEMENTS_PER_VECTOR):
                        x = Float32(loaded[j])
                        results[j] = soft_cap_temp * tanh(
                            x * multiplier / soft_cap_temp, fastmath=False
                        )

                    output_pack = cute.make_tensor(
                        out_vec_ptr + aligned_offset, fragment_layout
                    )
                    cute.copy(store_atom, results, output_pack)
                    element_offset = element_offset + element_stride

                # group = tid
                # while group < num_vec:
                #     # Typed pointer arithmetic uses elements. Both pointers advance by
                #     # four elements per group, regardless of their dtype.
                #     element_offset = cute.assume(
                #         group * ELEMENTS_PER_VECTOR,
                #         divby=ELEMENTS_PER_VECTOR,
                #     )
                #     input_pack = cute.make_tensor(
                #         in_vec_ptr + element_offset, fragment_layout
                #     )

                #     cute.copy(load_atom, input_pack, loaded)
                #     for j in cutlass.range_constexpr(ELEMENTS_PER_VECTOR):
                #         x = Float32(loaded[j])
                #         results[j] = soft_cap_temp * tanh(
                #             x * multiplier / soft_cap_temp,
                #             fastmath=False,
                #         )
                #     output_pack = cute.make_tensor(
                #         out_vec_ptr + element_offset, fragment_layout
                #     )
                #     cute.copy(store_atom, results, output_pack)
                #     group = group + stride

            # Scalar tail starts after the last complete vector.
            tail_start = skipped + (num_vec << 2)
            i = tail_start + tid
            if i < d:
                x = Float32(input_row[i])
                out_row[i] = soft_cap_temp * tanh(
                    x * multiplier / soft_cap_temp, fastmath=False
                )

            # if can_vectorize:
            #     num_vec = Int32((d - skipped) >> 2)  # elements per vector == 4
            #     tail_start = Int32(skipped + (num_vec << 2))

            #     i: int = tid
            #     if i < skipped:
            #         x = Float32(input_row[i])
            #         out_row[i] = soft_cap_temp * tanh(
            #             x * multiplier / soft_cap_temp, fastmath=False
            #         )

            #     if num_vec > 0:
            #         in_vec_ptr = cute.make_ptr(
            #             input.dtype,
            #             input_row_addr + skipped * input_element_bytes,
            #             cute.AddressSpace.gmem,
            #             assumed_align=in_vec_bytes,
            #         )
            #         out_vec_ptr = cute.make_ptr(
            #             out.dtype,
            #             out_addr_after_skip,
            #             cute.AddressSpace.gmem,
            #             assumed_align=out_vec_bytes,
            #         )

            #         vec_layout = cute.make_layout(
            #             (ELEMENTS_PER_VECTOR, num_vec), stride=(1, ELEMENTS_PER_VECTOR
            # )
            #         )
            #         input_vectors = cute.make_tensor(in_vec_ptr, vec_layout)
            #         out_vectors = cute.make_tensor(out_vec_ptr, vec_layout)

            #         load_atom: cute.CopyAtom = cute.make_copy_atom(
            #             cute.nvgpu.CopyUniversalOp(),
            #             input.dtype,
            #             num_bits_per_copy=in_vec_bytes * 8,
            #         )
            #         store_atom: cute.CopyAtom = cute.make_copy_atom(
            #             cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            #         )

            #         loaded: cute.Tensor = cute.make_rmem_tensor(
            #             cute.make_layout(ELEMENTS_PER_VECTOR), input.dtype
            #         )
            #         results: cute.Tensor = cute.make_rmem_tensor(
            #             cute.make_layout(ELEMENTS_PER_VECTOR), Float32
            #         )

            #         group: int = tid
            #         if group < num_vec:
            #             cute.copy(load_atom, input_vectors[None, group], loaded)
            #             for k in cutlass.range_constexpr(ELEMENTS_PER_VECTOR):
            #                 x = Float32(loaded[k])
            #                 results[k] = soft_cap_temp * tanh(
            #                     x * multiplier / soft_cap_temp, fastmath=False
            #                 )
            #             cute.copy(store_atom, results, out_vectors[None, group])

            #     i = tail_start + tid
            #     if i < d:
            #         x = Float32(input_row[i])
            #         out_row[i] = soft_cap_temp * tanh(
            #             x * multiplier / soft_cap_temp, fastmath=False
            #         )

            # else:
            #     i = tid
            #     while i < d:
            #         x = Float32(input_row[i])
            #         out_row[i] = soft_cap_temp * tanh(
            #             x * multiplier / soft_cap_temp, fastmath=False
            #         )
            #         i = i + stride

    _compile_cache: dict[Any, Any] = {}

    def _make_ptrs(dtype, out_addr, input_addr, *, vectorized):
        input_alignment = dtype.width // 8
        output_alignment = Float32.width // 8

        if vectorized:
            input_alignment *= ELEMENTS_PER_VECTOR
            output_alignment *= ELEMENTS_PER_VECTOR

        return (
            make_ptr(
                Float32,
                out_addr,
                cute.AddressSpace.gmem,
                assumed_align=output_alignment,
            ),
            make_ptr(
                dtype,
                input_addr,
                cute.AddressSpace.gmem,
                assumed_align=input_alignment,
            ),
        )

    def _compiled_kernel(dtype, kernel_kind, stream):
        key = (dtype, kernel_kind, torch.accelerator.current_device_index())
        compiled = _compile_cache.get(key)
        if compiled is None:
            out_ptr, input_ptr = _make_ptrs(
                dtype,
                VECTOR_ALIGNMENT_BYTES,
                VECTOR_ALIGNMENT_BYTES,
                vectorized=(kernel_kind == "full_vectorized"),
            )
            compiled = cute.compile(
                MulTanhSoftCapKernel(),
                out_ptr,
                input_ptr,
                Int32(ELEMENTS_PER_VECTOR),  # Example d satisfies aligned dispatch.
                Int32(1),  # num_tokens
                Int32(1),  # blocks_per_row
                Float32(1.0),
                Float32(1.0),
                kernel_kind,
                stream,
                options="--ptxas-options -maxrregcount=32",
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


def muse_spark_mul_tanh_soft_cap(
    out: torch.Tensor,
    input: torch.Tensor,
    multiplier: float,
    soft_cap_temp: float,
) -> None:
    """Write ``soft_cap_temp * tanh(input * multiplier / soft_cap_temp)`` to out.

    Compute in FP32 on the input device's current CUDA stream. Use a fully
    vectorized kernel for aligned rows, a prefix/vector/tail kernel when
    both pointers can be aligned by the same prefix, and a scalar kernel
    otherwise. Empty inputs require no kernel launch.
    """
    _validate_inputs(out, input, soft_cap_temp)

    if input.numel() == 0:
        return

    d = input.shape[-1]
    num_tokens = input.numel() // d
    vec_size = ELEMENTS_PER_VECTOR

    input_addr = input.data_ptr()
    out_addr = out.data_ptr()
    input_phase = (input_addr // input.element_size()) % vec_size
    out_phase = (out_addr // out.element_size()) % vec_size

    if input_phase != out_phase:
        kernel_kind = "full_scalar"
    elif input_phase == 0 and d % vec_size == 0:
        kernel_kind = "full_vectorized"
    else:
        kernel_kind = "prefix_vector_tail"

    work_per_row = (d + vec_size - 1) // vec_size
    blocks_to_cover = (work_per_row + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    blocks_per_row = min(
        max(TARGET_BLOCKS // num_tokens, 1),
        max(blocks_to_cover, 1),
    )

    input_dtype = _TORCH_TO_CUTLASS_DTYPE[input.dtype]
    with torch.accelerator.device_index(input.device.index):
        out_ptr, input_ptr = _make_ptrs(
            input_dtype,
            out_addr,
            input_addr,
            vectorized=(kernel_kind == "full_vectorized"),
        )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _compiled_kernel(input_dtype, kernel_kind, stream)

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


def supports_mul_tanh_soft_cap(dtype: torch.dtype) -> bool:
    return (
        current_platform.is_cuda() and HAS_CUTEDSL and dtype in _TORCH_TO_CUTLASS_DTYPE
    )


def mul_tanh_soft_cap(
    x: torch.Tensor,
    multiplier: float,
    soft_cap_temp: float,
) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float32)
    torch.ops.vllm.muse_spark_mul_tanh_soft_cap_cutedsl(
        out, x, multiplier, soft_cap_temp
    )
    return out


class CuTeMulTanhSoftCap(MulTanhSoftCap):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.is_contiguous() and supports_mul_tanh_soft_cap(x.dtype):
            return mul_tanh_soft_cap(x, self.multiplier, self.soft_cap_temp)

        return super().forward(x)

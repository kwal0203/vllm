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
        # @cute.jit
        # def __call__(
        #     self,
        #     out_ptr: cute.Pointer,
        #     input_ptr: cute.Pointer,
        #     d: Int32,
        #     num_tokens: Int32,
        #     blocks_per_row: Int32,
        #     multiplier: Float32,
        #     soft_cap_temp: Float32,
        #     vectorized: cutlass.Constexpr,
        #     stream: cuda.CUstream,
        # ) -> None:
        #     if cutlass.const_expr(vectorized):
        #         d = cute.assume(d, divby=ELEMENTS_PER_VECTOR)

        #     self.kernel_scalar(
        #         out_ptr,
        #         input_ptr,
        #         d,
        #         multiplier,
        #         soft_cap_temp,
        #     ).launch(
        #         grid=(num_tokens, blocks_per_row, 1),
        #         block=(THREADS_PER_BLOCK, 1, 1),
        #         stream=stream,
        #     )

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
            grid = (num_tokens, blocks_per_row, 1)
            block = (THREADS_PER_BLOCK, 1, 1)

            if cutlass.const_expr(vectorized):
                self.kernel_vectorized(
                    out_ptr, input_ptr, d, multiplier, soft_cap_temp
                ).launch(grid=grid, block=block, stream=stream)
            else:
                self.kernel_scalar(
                    out_ptr, input_ptr, d, multiplier, soft_cap_temp
                ).launch(grid=grid, block=block, stream=stream)

        @cute.kernel
        def kernel_vectorized(
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
        def kernel_scalar(
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

            tid: Int32 = bidy * bdimx + tidx
            # const int tid = blockIdx.y * blockDim.x + threadIdx.x;

            stride: Int32 = gdimy * bdimx
            # const int stride = gridDim.y * blockDim.x;

            row_offset: Int64 = Int64(bidx) * Int64(d)
            # const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * d;

            row_layout: cute.Layout = cute.make_layout(d)
            input_row: cute.Tensor = cute.make_tensor(input + row_offset, row_layout)
            # const scalar_t* __restrict__ in_row = input + row_offset;
            out_row: cute.Tensor = cute.make_tensor(out + row_offset, row_layout)
            # float* __restrict__ out_row = out + row_offset;

            input_row_addr = input_row.iterator.toint()
            # uintptr_t addr = reinterpret_cast<uintptr_t>(in);
            out_row_addr = out_row.iterator.toint()
            # uintptr_t out_addr = reinterpret_cast<uintptr_t>(out);

            input_element_bytes = input.dtype.width >> 3
            out_element_bytes = out.dtype.width >> 3

            in_vec_bytes = ELEMENTS_PER_VECTOR * input_element_bytes
            # constexpr int WIDTH = VEC_SIZE * sizeof(InT);       // eg: 16 B
            out_vec_bytes = ELEMENTS_PER_VECTOR * out_element_bytes
            # constexpr int OUT_WIDTH = VEC_SIZE * sizeof(OutT);  // eg: 16 B

            can_vec = (
                ((input_row_addr & (in_vec_bytes - 1)) == 0)
                & ((out_row_addr & (out_vec_bytes - 1)) == 0)
                & ((d & (ELEMENTS_PER_VECTOR - 1)) == 0)
            )
            # bool can_vec =
            # ((addr & (WIDTH - 1)) == 0) &&
            # ((out_addr & (OUT_WIDTH - 1)) == 0) &&
            # ((len & (VEC_SIZE - 1)) == 0);

            if can_vec:
                num_vec: Int32 = Int32(d >> 2)  # elements per vector == 4
                # int num_vec = len / VEC_SIZE;

                in_vec_ptr: cute.Pointer = cute.make_ptr(
                    input.dtype,
                    input_row_addr,
                    cute.AddressSpace.gmem,
                    assumed_align=in_vec_bytes,
                )
                # using vin_t = vec_n_t<InT, VEC_SIZE>;

                out_vec_ptr: cute.Pointer = cute.make_ptr(
                    out.dtype,
                    out_row_addr,
                    cute.AddressSpace.gmem,
                    assumed_align=out_vec_bytes,
                )
                # using vout_t = vec_n_t<OutT, VEC_SIZE>;

                vec_layout = cute.make_layout(
                    (ELEMENTS_PER_VECTOR, num_vec), stride=(1, ELEMENTS_PER_VECTOR)
                )
                input_vectors = cute.make_tensor(in_vec_ptr, vec_layout)
                # auto* v_in = reinterpret_cast<const vin_t*>(in);
                out_vectors = cute.make_tensor(out_vec_ptr, vec_layout)
                # auto* v_out = reinterpret_cast<vout_t*>(out);

                load_atom: cute.CopyAtom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    input.dtype,
                    num_bits_per_copy=in_vec_bytes * 8,
                )
                store_atom: cute.CopyAtom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    out.dtype,
                    num_bits_per_copy=out_vec_bytes * 8,
                )

                loaded: cute.Tensor = cute.make_rmem_tensor(
                    cute.make_layout(ELEMENTS_PER_VECTOR), input.dtype
                )
                results: cute.Tensor = cute.make_rmem_tensor(
                    cute.make_layout(ELEMENTS_PER_VECTOR), Float32
                )

                group: int = tid
                while group < num_vec:
                    cute.copy(load_atom, input_vectors[None, group], loaded)
                    for k in cutlass.range_constexpr(ELEMENTS_PER_VECTOR):
                        x = Float32(loaded[k])
                        results[k] = soft_cap_temp * tanh(
                            x * multiplier / soft_cap_temp, fastmath=False
                        )
                    cute.copy(store_atom, results, out_vectors[None, group])
                    group = group + stride
                # for (int i = tid; i < num_vec; i += stride) {
                # vout_t tmp;
                # // Make a local copy of the entire pack
                # vin_t src = v_in[i];  // <- encourages a single vector ld
                # vec_op(tmp, src);
                # v_out[i] = tmp;  // <- encourages a single vector st
                # return;
            else:
                alignment_mask = in_vec_bytes - 1
                misalignment_bytes = Int32(input_row_addr & alignment_mask)
                # int misalignment_offset = addr & (WIDTH - 1);       // addr % 64

                skipped_bytes = (in_vec_bytes - misalignment_bytes) & alignment_mask
                # int alignment_bytes = WIDTH - misalignment_offset;  // 64 -
                # (addr % 64)

                skipped = min(skipped_bytes // input_element_bytes, d)
                # int prefix_elems = alignment_bytes & (WIDTH - 1);   // handle 64
                # prefix_elems /= sizeof(InT);
                # prefix_elems = min(prefix_elems, len);  // 0 ≤ prefix < 16

                out_addr_after_skip = out_row_addr + skipped * out_element_bytes
                out_misaligned = Int32(
                    out_addr_after_skip & (out_vec_bytes - 1)
                ) != Int32(0)

                if out_misaligned:
                    # Process the entire row with scalar operations.
                    i = tid
                    while i < d:
                        x = Float32(input_row[i])
                        out_row[i] = soft_cap_temp * tanh(
                            x * multiplier / soft_cap_temp, fastmath=False
                        )
                        i = i + stride
                    # if (((out_addr + prefix_elems * sizeof(OutT)) & (OUT_WIDTH - 1))
                    # != 0) {
                    #     for (int i = tid; i < len; i += stride) {
                    #     scalar_op(out[i], in[i]);
                    #     }
                    #     return;
                    # }
                else:
                    # Process the scalar prefix, vectorized middle, and scalar tail.
                    num_vec = Int32((d - skipped) >> 2)  # elements per vector == 4
                    tail_start = Int32(skipped + (num_vec << 2))

                    i = tid
                    if i < skipped:
                        x = Float32(input_row[i])
                        out_row[i] = soft_cap_temp * tanh(
                            x * multiplier / soft_cap_temp, fastmath=False
                        )

                    # // 1. prefill the when it is unsafe to vectorize
                    # for (int i = tid; i < prefix_elems; i += stride) {
                    #     scalar_op(out[i], in[i]);
                    # }

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

                        vec_layout = cute.make_layout(
                            (ELEMENTS_PER_VECTOR, num_vec),
                            stride=(1, ELEMENTS_PER_VECTOR),
                        )
                        input_vectors = cute.make_tensor(in_vec_ptr, vec_layout)
                        out_vectors = cute.make_tensor(out_vec_ptr, vec_layout)

                        load_atom = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            input.dtype,
                            num_bits_per_copy=in_vec_bytes * 8,
                        )
                        store_atom = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                        )

                        loaded = cute.make_rmem_tensor(
                            cute.make_layout(ELEMENTS_PER_VECTOR), input.dtype
                        )
                        results = cute.make_rmem_tensor(
                            cute.make_layout(ELEMENTS_PER_VECTOR), Float32
                        )

                        group = tid
                        while group < num_vec:
                            cute.copy(load_atom, input_vectors[None, group], loaded)
                            for k in cutlass.range_constexpr(ELEMENTS_PER_VECTOR):
                                x = Float32(loaded[k])
                                results[k] = soft_cap_temp * tanh(
                                    x * multiplier / soft_cap_temp, fastmath=False
                                )
                            cute.copy(store_atom, results, out_vectors[None, group])
                            group = group + stride

                    # in += prefix_elems;
                    # out += prefix_elems;
                    # len -= prefix_elems;

                    # int num_vec = len / VEC_SIZE;
                    # using vin_t = vec_n_t<InT, VEC_SIZE>;
                    # using vout_t = vec_n_t<OutT, VEC_SIZE>;
                    # auto* v_in = reinterpret_cast<const vin_t*>(in);
                    # auto* v_out = reinterpret_cast<vout_t*>(out);

                    # // 2. vectorize the main part
                    # for (int i = tid; i < num_vec; i += stride) {
                    #     vout_t tmp;
                    #     // Make a local copy of the entire pack
                    #     vin_t src = v_in[i];  // <- encourages a single vector ld
                    #     vec_op(tmp, src);
                    #     v_out[i] = tmp;  // <- encourages a single vector st
                    # }

                    i = tail_start + tid
                    if i < d:
                        x = Float32(input_row[i])
                        out_row[i] = soft_cap_temp * tanh(
                            x * multiplier / soft_cap_temp, fastmath=False
                        )
                    # // 3. handle the tail
                    # int tail_start = num_vec * VEC_SIZE;
                    # for (int i = tid + tail_start; i < len; i += stride) {
                    #     scalar_op(out[i], in[i]);
                    # }

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

    def _make_ptrs(
        dtype: Any,
        out_addr: cute.Pointer,
        input_addr: cute.Pointer,
        *,
        vectorized: bool,
    ) -> tuple[Any, Any]:
        align_kwargs: dict[str, Any] = (
            {"assumed_align": VECTOR_ALIGNMENT_BYTES} if vectorized else {}
        )
        return (
            make_ptr(Float32, out_addr, cute.AddressSpace.gmem, **align_kwargs),
            make_ptr(dtype, input_addr, cute.AddressSpace.gmem, **align_kwargs),
        )

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
            out_ptr, input_ptr = _make_ptrs(
                dtype,
                VECTOR_ALIGNMENT_BYTES,
                VECTOR_ALIGNMENT_BYTES,
                vectorized=vectorized,
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
    out: torch.Tensor, input: torch.Tensor, multiplier: float, soft_cap_temp: float
) -> None:
    """Write ``soft_cap_temp * tanh(input * multiplier / soft_cap_temp)`` to out.

    The computation runs in FP32 on the input device's current CUDA stream.
    Use four-element vectors when the last dimension is divisible by four and
    both base addresses are 16-byte aligned. Other valid inputs use the scalar
    kernel. Empty inputs require no kernel launch.

    Args:
        out: Contiguous FP32 CUDA buffer with the same number of elements as
            ``input``. Written in place.
        input: Contiguous FP16, BF16, or FP32 CUDA tensor with at least one
            dimension, on the same device as ``out``.
        multiplier: Scale applied to the input before soft-capping.
        soft_cap_temp: Nonzero soft-cap temperature.

    Raises:
        RuntimeError: If CuTe DSL is unavailable or an input requirement fails.

    """
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
    work_per_row: int = (d + ELEMENTS_PER_VECTOR - 1) // ELEMENTS_PER_VECTOR
    blocks_to_cover: int = (work_per_row + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    blocks_per_row: int = min(
        max(TARGET_BLOCKS // num_tokens, 1), max(blocks_to_cover, 1)
    )

    with torch.accelerator.device_index(input.device.index):
        out_ptr, input_ptr = _make_ptrs(
            _TORCH_TO_CUTLASS_DTYPE[input.dtype],
            out.data_ptr(),
            input.data_ptr(),
            vectorized=vectorized,
        )

        stream: cuda.CUstream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled: Any = _compiled_kernel(
            _TORCH_TO_CUTLASS_DTYPE[input.dtype], vectorized, stream
        )

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

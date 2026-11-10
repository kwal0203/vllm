# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness coverage for the CuTe DSL MuseSpark soft-cap kernel.

Compare against both the existing native CUDA operation and a PyTorch
reference. Performance benchmarks belong outside this test file.
"""

import math

import pytest
import torch

import vllm._custom_ops  # noqa: F401
from vllm.models.muse_spark.nvidia.ops.cute_dsl.mul_tanh_soft_cap import (
    HAS_CUTEDSL,
    muse_spark_mul_tanh_soft_cap,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_CUTEDSL,
    reason="requires CUDA and nvidia-cutlass-dsl",
)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DTYPE_IDS = ["fp16", "bf16", "fp32"]
MODEL_MULTIPLIER = 0.17677669529663687
MODEL_TEMPERATURE = 20.0

SHAPES = [
    (128,),
    (3, 128),
    (3, 129),
    (2, 4096),
    (1, 202048),
    (8, 202048),
    (32, 202048),
    (0,),
    (0, 129),
    (3, 0),
    (1,),
    (3,),
    (4,),
    (5,),
    (2, 3, 129),
    (2, 4097),
]


def _assert_matches_references(
    x: torch.Tensor,
    out: torch.Tensor,
    multiplier: float = MODEL_MULTIPLIER,
    temperature: float = MODEL_TEMPERATURE,
) -> None:
    """Run the candidate once and compare both references in FP32."""
    out_cuda = torch.empty_like(x, dtype=torch.float32)
    torch.ops._C.muse_spark_mul_tanh_soft_cap(out_cuda, x, multiplier, temperature)
    muse_spark_mul_tanh_soft_cap(out, x, multiplier, temperature)
    torch.accelerator.synchronize()

    # The native launcher narrows the Python scalars to FP32 before launch.
    multiplier_fp32 = torch.tensor(multiplier, device=x.device, dtype=torch.float32)
    temperature_fp32 = torch.tensor(temperature, device=x.device, dtype=torch.float32)
    expected = temperature_fp32 * torch.tanh(
        x.float() * multiplier_fp32 / temperature_fp32
    )

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert out.device == x.device
    torch.testing.assert_close(out, out_cuda, msg="CuTe vs native CUDA")
    torch.testing.assert_close(out, expected, msg="CuTe vs PyTorch")


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", SHAPES, ids=[str(shape) for shape in SHAPES])
def test_soft_cap_shapes(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Cover empty, small, odd-width, higher-rank, and model-sized inputs."""
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    # Finite references make any unwritten NaN output fail the comparison.
    out = torch.full_like(x, float("nan"), dtype=torch.float32)
    _assert_matches_references(x, out)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    ("input_offset", "output_offset"),
    [(1, 0), (0, 1), (1, 1)],
    ids=["offset-input", "offset-output", "offset-both"],
)
def test_soft_cap_offset_buffers(
    dtype: torch.dtype,
    input_offset: int,
    output_offset: int,
) -> None:
    """Offset addresses by one element while preserving contiguity."""
    torch.manual_seed(0)
    shape = (3, 128)
    numel = math.prod(shape)

    input_storage = torch.randn(numel + input_offset, device="cuda", dtype=dtype)
    x = input_storage[input_offset:].view(shape)
    output_storage = torch.full(
        (numel + output_offset,),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )
    out = output_storage[output_offset:].view(shape)

    assert x.is_contiguous()
    assert out.is_contiguous()
    if input_offset:
        assert x.data_ptr() % 16 != 0
    if output_offset:
        assert out.data_ptr() % 16 != 0

    _assert_matches_references(x, out)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", [(3, 128), (3, 129)], ids=["vectorized", "scalar"])
@pytest.mark.parametrize("values", ["zeros", "tiny", "normal", "large"])
@pytest.mark.parametrize(
    "multiplier",
    [0.0, 1.0, MODEL_MULTIPLIER, -1.0],
    ids=["zero", "one", "model", "negative"],
)
@pytest.mark.parametrize("temperature", [0.25, 1.0, MODEL_TEMPERATURE])
def test_soft_cap_numerical_cases(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    values: str,
    multiplier: float,
    temperature: float,
) -> None:
    """Exercise magnitudes and scalar parameters on both kernel paths."""
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if values == "zeros":
        x.zero_()
    elif values == "tiny":
        x.mul_(1e-4)
    elif values == "large":
        x.mul_(1000.0)

    out = torch.full_like(x, float("nan"), dtype=torch.float32)
    _assert_matches_references(x, out, multiplier, temperature)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("width", [128, 129], ids=["vectorized", "scalar"])
def test_soft_cap_custom_op_compile(dtype: torch.dtype, width: int) -> None:
    torch.manual_seed(0)

    def run_op(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x, dtype=torch.float32)
        torch.ops.vllm.muse_spark_mul_tanh_soft_cap_cutedsl(
            out, x, MODEL_MULTIPLIER, MODEL_TEMPERATURE
        )
        return out

    x = torch.randn((3, width), device="cuda", dtype=dtype)

    # Test registration and warm the CuTe kernel before compilation.
    eager_out = run_op(x)
    expected = MODEL_TEMPERATURE * torch.tanh(
        x.float() * MODEL_MULTIPLIER / MODEL_TEMPERATURE
    )
    torch.testing.assert_close(eager_out, expected)

    compiled_op = torch.compile(run_op, fullgraph=True)

    # Reuse the compiled function with different input values.
    for test_input in (x, -x):
        out = compiled_op(test_input)
        expected = MODEL_TEMPERATURE * torch.tanh(
            test_input.float() * MODEL_MULTIPLIER / MODEL_TEMPERATURE
        )
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("path", ["cute", "fallback", "noncontiguous", "cpu"])
def test_soft_cap_layer_dispatch(
    dtype: torch.dtype,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.muse_spark.nvidia.ops.cute_dsl import (
        mul_tanh_soft_cap as soft_cap_module,
    )

    torch.manual_seed(0)
    device = "cpu" if path == "cpu" else "cuda"
    x = torch.randn((3, 128), device=device, dtype=dtype)
    if path == "noncontiguous":
        x = x.t()
        assert not x.is_contiguous()

    original_wrapper = soft_cap_module.mul_tanh_soft_cap
    calls = []

    def tracked_wrapper(x, multiplier, soft_cap_temp):
        calls.append(True)
        return original_wrapper(x, multiplier, soft_cap_temp)

    monkeypatch.setattr(soft_cap_module, "mul_tanh_soft_cap", tracked_wrapper)
    if path == "fallback":
        monkeypatch.setattr(
            soft_cap_module,
            "supports_mul_tanh_soft_cap",
            lambda dtype: False,
        )

    layer = soft_cap_module.CuTeMulTanhSoftCap(MODEL_MULTIPLIER, MODEL_TEMPERATURE)
    out = layer(x)

    expected = MODEL_TEMPERATURE * torch.tanh(
        x.float() * MODEL_MULTIPLIER / MODEL_TEMPERATURE
    )
    torch.testing.assert_close(out, expected)
    assert out.dtype == torch.float32
    assert bool(calls) == (path == "cute")


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("width", [128, 129])
def test_soft_cap_layer_compile(dtype: torch.dtype, width: int) -> None:
    from vllm.models.muse_spark.nvidia.ops.cute_dsl.mul_tanh_soft_cap import (
        CuTeMulTanhSoftCap,
    )

    torch.manual_seed(0)
    layer = CuTeMulTanhSoftCap(MODEL_MULTIPLIER, MODEL_TEMPERATURE)
    x = torch.randn((3, width), device="cuda", dtype=dtype)

    layer(x)  # Warm the CuTe kernel before compiled execution.
    compiled_layer = torch.compile(layer, fullgraph=True)

    for test_input in (x, -x):
        expected = MODEL_TEMPERATURE * torch.tanh(
            test_input.float() * MODEL_MULTIPLIER / MODEL_TEMPERATURE
        )
        torch.testing.assert_close(compiled_layer(test_input), expected)

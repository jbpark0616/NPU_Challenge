# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import ttnn
from models.bos_model.resnet50_jbpark.tests.resnet50_test_infra import create_test_infra
from models.common.utility_functions import enable_memory_reports


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        4,
    ],
    ids=[
        "batch_1",
        "batch_2",
        "batch_4",
    ],
)
@pytest.mark.parametrize(
    "act_dtype, weight_dtype, math_fidelity",
    ((ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [False],
    ids=[
        "pretrained_weight_false",
    ],
)
def test_resnet_50(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
    model_location_generator,
):
    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        model_location_generator=model_location_generator,
    )
    enable_memory_reports()
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # First run configures convs JIT
    test_infra.run()
    # # Optimized run
    # test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # test_infra.run()
    # # # More optimized run with caching
    # test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # test_infra.run()
    # Skip PCC validation for stage-split runs (STAGE >= 0) — synthesized inputs
    # produce garbage outputs by design (see ttnn_functional_resnet50.__call__).
    if int(os.environ.get("RESNET_STAGE", "-1")) < 0:
        test_infra.validate()

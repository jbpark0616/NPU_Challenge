#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly CLEAN_DEVICE_CONFIRMATION="${1:-}"
readonly BUILD_JOBS="${CONV_CORE_REPLACEMENT_BUILD_JOBS:-8}"
readonly TEST_DIR="tests/tt_metal/tt_metal/data_movement/conv_core_replacement"
readonly TEST_FILE="${TEST_DIR}/test_bos_resnet50_baseline.py"
readonly GUARDED_RUNNER="${TEST_DIR}/run_bos_resnet50_full_model_guarded.sh"

if [[ "${CLEAN_DEVICE_CONFIRMATION}" != "--clean-device-confirmed" ]]; then
    echo "Refusing to launch the HEIGHT_SHARDED regression sequence." >&2
    echo "After a clean device reset, pass --clean-device-confirmed." >&2
    exit 2
fi

readonly -a DEVICE_TESTS=(
    environment_baseline
    layer1_height_segmented_column_conv2_isolated
    layer1_height_segmented_row_conv2_isolated
    layer1_height_four_rectangle_conv2_isolated
    layer1_height_four_rectangle_conv2_to_conv3_exits_remapping
    layer1_height_boundary_translation_conv2_isolated
    layer1_height_segmented_column_conv2_full_model
    layer1_height_segmented_row_conv2_full_model
    layer1_height_boundary_translation_conv2_full_model
)

echo "Building and running the host-only Conv core-remapping safety suite."
cmake --build build --target unit_tests_conv_core_replacement -j "${BUILD_JOBS}"
./build/test/ttnn/unit_tests_conv_core_replacement

echo "Checking Python and guarded-runner syntax before device submission."
python -m py_compile "${TEST_FILE}"
bash -n "${GUARDED_RUNNER}"

echo "Starting HEIGHT_SHARDED device regression (${#DEVICE_TESTS[@]} independent guarded processes)."
for test_filter in "${DEVICE_TESTS[@]}"; do
    echo
    echo "HEIGHT REGRESSION: ${test_filter}"
    # set -e stops the sequence on every nonzero exit, including watchdog,
    # assertion failure, host validation failure, or fixture teardown failure.
    bash "${GUARDED_RUNNER}" "${test_filter}" --clean-device-confirmed
done

echo
echo "HEIGHT_SHARDED REGRESSION PASSED: ${#DEVICE_TESTS[@]} device tests completed."

#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

readonly TEST_FILE="tests/tt_metal/tt_metal/data_movement/conv_core_replacement/test_bos_resnet50_baseline.py"
readonly TIMEOUT_SECONDS="${CONV_CORE_REPLACEMENT_TIMEOUT_SECONDS:-120}"
readonly TEST_FILTER="${1:-environment_baseline}"
readonly CLEAN_DEVICE_CONFIRMATION="${2:-}"

case "${TEST_FILTER}" in
    environment_baseline|layer1_height_segmented_column_conv2_isolated|layer1_height_boundary_translation_conv2_isolated|layer1_height_segmented_row_conv2_isolated|layer1_height_four_rectangle_conv2_isolated|layer1_height_four_rectangle_conv2_to_conv3_exits_remapping|layer1_height_segmented_column_conv2_full_model|layer1_height_boundary_translation_conv2_full_model|layer1_height_segmented_row_conv2_full_model|layer3_cyclic_column_rest_full_model|layer4_translated_conv1_isolated|layer4_translated_conv1_reshard_if_not_optimal_isolated|layer4_segmented_column_conv1_isolated|layer4_segmented_row_conv1_isolated|layer4_four_rectangle_conv1_isolated|layer4_translated_conv2_isolated|layer4_segmented_column_conv2_isolated|layer4_segmented_row_conv2_isolated|layer4_row_and_column_exclusions_conv2_isolated|layer4_segmented_conv2_to_conv3_exits_remapping|layer4_segmented_row_conv2_to_conv3_exits_remapping|layer4_four_rectangle_conv2_to_conv3_exits_remapping|layer4_translated_downsample_exits_remapping|layer4_contiguous_core_set_16_translation|layer4_segmented_column_conv2_full_model|layer4_four_rectangle_conv2_full_model)
        ;;
    *)
        echo "Refusing unknown high-risk device-test filter: ${TEST_FILTER}" >&2
        echo "The known-deadlocking milp_phase_full_model filter is quarantined." >&2
        echo "Use run_bos_resnet50_validated_latency_guarded.sh for latency captures." >&2
        exit 2
        ;;
esac

if [[ "${CLEAN_DEVICE_CONFIRMATION}" != "--clean-device-confirmed" ]]; then
    echo "Refusing to launch a high-risk device test." >&2
    echo "After a clean device reset, pass --clean-device-confirmed as the second argument." >&2
    exit 2
fi

echo "WARNING: a host watchdog cannot recover a deadlocked device program."
echo "Launching ${TEST_FILTER} with a ${TIMEOUT_SECONDS}s external watchdog."

set +e
timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
    env RUN_CONV_CORE_REPLACEMENT_DEVICE_TESTS=1 \
        RUN_CONV_CORE_REPLACEMENT_HIGH_RISK_TESTS=1 \
    pytest -v -s "${TEST_FILE}" -k "${TEST_FILTER}"
status=$?
set -e

if [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
    echo "DEVICE-TEST WATCHDOG FIRED (exit=${status})." >&2
    echo "Do not submit another device workload; the board may require a power-cycle." >&2
fi

exit "${status}"

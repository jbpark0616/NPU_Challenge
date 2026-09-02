#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Profile only the two full-model paths that have already completed on board.
# Run each mode separately so a failed run can never launch the other mode.

set -uo pipefail

readonly TEST_FILE="tests/tt_metal/tt_metal/data_movement/conv_core_replacement/test_bos_resnet50_baseline.py"
readonly MODE="${1:-}"
readonly CLEAN_DEVICE_CONFIRMATION="${2:-}"
# Tracy can spend several minutes serializing a successful full-model trace
# after pytest and the device have already closed normally.
readonly TIMEOUT_SECONDS="${CONV_CORE_REPLACEMENT_TIMEOUT_SECONDS:-360}"
readonly OUTPUT_ROOT="${CONV_CORE_VALIDATED_LATENCY_OUTPUT_DIR:-generated/profiler/conv_core_validated_latency}"
readonly DEVICE_LOG="generated/profiler/.logs/profile_log_device.csv"

case "${MODE}" in
    stock)
        readonly TEST_FILTER="environment_baseline"
        ;;
    remapped)
        # This exact phase-0 mapping is the full-model configuration that
        # previously completed with PCC=0.999603608935384.
        readonly TEST_FILTER="layer3_cyclic_column_rest_full_model"
        ;;
    *)
        echo "Usage: $0 {stock|remapped} --clean-device-confirmed" >&2
        echo "This runner intentionally refuses MILP manifests and arbitrary phases." >&2
        exit 2
        ;;
esac

if [[ "${CLEAN_DEVICE_CONFIRMATION}" != "--clean-device-confirmed" ]]; then
    echo "Refusing to open the device without --clean-device-confirmed." >&2
    exit 2
fi

readonly MODE_DIR="${OUTPUT_ROOT}/${MODE}"
readonly MARKER="${MODE_DIR}/profile_start.marker"
mkdir -p "${MODE_DIR}"
touch "${MARKER}"

echo "Profiling validated path: mode=${MODE} test=${TEST_FILTER}"
echo "Output: ${MODE_DIR}"
echo "The per-Conv synchronization used during board validation remains enabled."

set +e
timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
    env RUN_CONV_CORE_REPLACEMENT_DEVICE_TESTS=1 \
        RUN_CONV_CORE_REPLACEMENT_HIGH_RISK_TESTS=1 \
        CONV_CORE_ROTATION_INFERENCE_INDEX=0 \
    python -m tracy -r -p -v -m pytest \
        "${TEST_FILE}" -k "${TEST_FILTER}"
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
    echo "VALIDATED LATENCY PROFILE FAILED mode=${MODE} exit=${status}." >&2
    if [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
        echo "WATCHDOG FIRED. Do not submit another workload; reset the board." >&2
    fi
    exit "${status}"
fi

if [[ ! -f "${DEVICE_LOG}" || ! "${DEVICE_LOG}" -nt "${MARKER}" ]]; then
    echo "Fresh device profile was not produced: ${DEVICE_LOG}" >&2
    exit 3
fi

ops_csv="$(
    find generated/profiler -type f -name 'ops_perf_results_*.csv' -newer "${MARKER}" \
        -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
)"
if [[ -z "${ops_csv}" || ! -f "${ops_csv}" ]]; then
    echo "Fresh ops_perf_results CSV was not produced." >&2
    exit 3
fi

cp "${DEVICE_LOG}" "${MODE_DIR}/profile_log_device.csv"
cp "${ops_csv}" "${MODE_DIR}/ops_perf_results.csv"
test_sha256="$(sha256sum "${TEST_FILE}" | cut -d' ' -f1)"
printf '%s\n' \
    "mode=${MODE}" \
    "test_filter=${TEST_FILTER}" \
    "rotation_inference_index=0" \
    "per_conv_sync=enabled" \
    "test_file_sha256=${test_sha256}" \
    "ops_source=${ops_csv}" \
    > "${MODE_DIR}/profile_metadata.txt"

echo "Captured validated ${MODE} profile in ${MODE_DIR}."
echo "After both modes pass, run extract_validated_latency_comparison.py."

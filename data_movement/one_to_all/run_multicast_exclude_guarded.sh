#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

readonly TIMEOUT_SECONDS="${NOC_EXCLUDE_TIMEOUT_SECONDS:-30}"
readonly CLEAN_DEVICE_CONFIRMATION="${1:-}"
readonly NOC_SELECTION="${2:-noc0}"
readonly TEST_BINARY="./build/test/tt_metal/unit_tests_data_movement"

case "${NOC_SELECTION}" in
    noc0)
        readonly TEST_FILTER="GenericMeshDeviceFixture.TensixDataMovementMulticastExcludePostedQuadrantNoc0"
        ;;
    noc1)
        readonly TEST_FILTER="GenericMeshDeviceFixture.TensixDataMovementMulticastExcludePostedQuadrantNoc1"
        ;;
    *)
        echo "Unknown NoC selection: ${NOC_SELECTION}; expected noc0 or noc1" >&2
        exit 2
        ;;
esac

if [[ "${CLEAN_DEVICE_CONFIRMATION}" != "--clean-device-confirmed" ]]; then
    echo "Refusing to launch the NOC exclude discovery test." >&2
    echo "Power-cycle/reset the device, then pass --clean-device-confirmed." >&2
    exit 2
fi

if [[ ! -x "${TEST_BINARY}" ]]; then
    echo "Missing test binary: ${TEST_BINARY}" >&2
    exit 2
fi

readonly TEST_NAME="${TEST_FILTER#*.}"
if ! "${TEST_BINARY}" --gtest_list_tests | grep -Fq "${TEST_NAME}"; then
    echo "Refusing to launch: test is not registered in ${TEST_BINARY}: ${TEST_FILTER}" >&2
    exit 2
fi

echo "WARNING: this test writes an undocumented Blackhole NOC command-buffer register."
echo "A host watchdog cannot recover a wedged device. Do not launch another workload after a timeout."

set +e
timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
    env RUN_NOC_EXCLUDE_MICROTESTS=1 \
    "${TEST_BINARY}" --gtest_filter="${TEST_FILTER}"
status=$?
set -e

if [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
    echo "NOC EXCLUDE WATCHDOG FIRED (exit=${status})." >&2
    echo "Do not submit another device workload; power-cycle/reset the board first." >&2
elif [[ ${status} -ne 0 ]]; then
    echo "NOC exclude discovery failed (exit=${status})." >&2
    echo "Treat the device state as suspect and reset it before another workload." >&2
fi

exit "${status}"

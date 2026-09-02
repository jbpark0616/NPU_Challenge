#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

echo "REFUSED: the H=20 MILP manifest is quarantined after a board deadlock at layer3.module2.conv3." >&2
echo "Use run_bos_resnet50_validated_latency_guarded.sh with stock or remapped instead." >&2
exit 2

readonly TEST_FILE="tests/tt_metal/tt_metal/data_movement/conv_core_replacement/test_bos_resnet50_baseline.py"
readonly DEFAULT_MANIFEST="tests/tt_metal/tt_metal/data_movement/conv_core_replacement/policy/data/experiments/single_axis_split/full_model_schedule_h20.json"
readonly MANIFEST="${CONV_CORE_MILP_MANIFEST:-${DEFAULT_MANIFEST}}"
readonly HORIZON="${CONV_CORE_MILP_HORIZON:-20}"
readonly TIMEOUT_SECONDS="${CONV_CORE_REPLACEMENT_TIMEOUT_SECONDS:-180}"
readonly CLEAN_DEVICE_CONFIRMATION="${1:-}"
readonly FULL_PERIOD_CONFIRMATION="${2:-}"
readonly DEVICE_LOG="generated/profiler/.logs/profile_log_device.csv"
readonly RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly OUTPUT_ROOT="${CONV_CORE_MILP_PROFILE_OUTPUT_DIR:-generated/profiler/conv_core_milp_period_${RUN_STAMP}}"

if [[ "${CLEAN_DEVICE_CONFIRMATION}" != "--clean-device-confirmed" ]]; then
    echo "Refusing to launch high-risk MILP full-model profiling." >&2
    echo "After a clean device reset, pass --clean-device-confirmed." >&2
    exit 2
fi
if [[ "${FULL_PERIOD_CONFIRMATION}" != "--run-full-period" ]]; then
    echo "Refusing to launch all ${HORIZON} phases without --run-full-period." >&2
    echo "Validate phase 0 first with run_bos_resnet50_full_model_guarded.sh." >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "MILP manifest not found: ${MANIFEST}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cp "${MANIFEST}" "${OUTPUT_ROOT}/full_model_schedule_h20.json"
candidate_csv="$(dirname "${MANIFEST}")/full_model_domain_candidates.csv"
if [[ ! -f "${candidate_csv}" ]]; then
    echo "Candidate table not found beside manifest: ${candidate_csv}" >&2
    exit 2
fi
cp "${candidate_csv}" "${OUTPUT_ROOT}/full_model_domain_candidates.csv"
manifest_sha256="$(sha256sum "${MANIFEST}" | cut -d' ' -f1)"
candidate_sha256="$(sha256sum "${candidate_csv}" | cut -d' ' -f1)"
printf '%s\n' "manifest=${MANIFEST}" "horizon=${HORIZON}" "started_utc=${RUN_STAMP}" \
    "manifest_sha256=${manifest_sha256}" "candidate_sha256=${candidate_sha256}" \
    > "${OUTPUT_ROOT}/period_run_metadata.txt"

echo "WARNING: a host watchdog cannot recover a deadlocked device program."
echo "Profiling ${HORIZON} independent full-model phases into ${OUTPUT_ROOT}."

for ((phase = 0; phase < HORIZON; phase++)); do
    phase_name="$(printf 'phase_%02d' "${phase}")"
    phase_dir="${OUTPUT_ROOT}/${phase_name}"
    marker="${phase_dir}/profile_start.marker"
    mkdir -p "${phase_dir}"
    touch "${marker}"
    echo "MILP PROFILE phase=${phase}/${HORIZON} output=${phase_dir}"

    set +e
    timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
        env RUN_CONV_CORE_REPLACEMENT_DEVICE_TESTS=1 \
            RUN_CONV_CORE_REPLACEMENT_HIGH_RISK_TESTS=1 \
            CONV_CORE_MILP_PHASE="${phase}" \
            CONV_CORE_MILP_MANIFEST="${MANIFEST}" \
            CONV_CORE_MILP_SYNC_EACH_CONV=0 \
        python -m tracy -r -p -v -m pytest \
            "${TEST_FILE}" -k milp_phase_full_model
    status=$?
    set -e

    if [[ ${status} -ne 0 ]]; then
        echo "MILP PROFILE FAILED phase=${phase} exit=${status}." >&2
        if [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
            echo "WATCHDOG FIRED. Do not submit another workload; reset the board." >&2
        fi
        exit "${status}"
    fi
    if [[ ! -f "${DEVICE_LOG}" || ! "${DEVICE_LOG}" -nt "${marker}" ]]; then
        echo "Fresh device profile was not produced for phase ${phase}: ${DEVICE_LOG}" >&2
        exit 3
    fi

    ops_csv="$(
        find generated/profiler -type f -name 'ops_perf_results_*.csv' -newer "${marker}" \
            -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
    )"
    if [[ -z "${ops_csv}" || ! -f "${ops_csv}" ]]; then
        echo "Fresh ops_perf_results CSV was not produced for phase ${phase}." >&2
        exit 3
    fi

    cp "${DEVICE_LOG}" "${phase_dir}/profile_log_device.csv"
    cp "${ops_csv}" "${phase_dir}/$(basename "${ops_csv}")"
    printf '%s\n' "phase=${phase}" "ops_source=${ops_csv}" \
        > "${phase_dir}/phase_metadata.txt"
    echo "MILP PROFILE phase=${phase} captured"
done

echo "MILP full-period profiling completed: ${OUTPUT_ROOT}"
echo "Run extract_milp_period_trisc_exposure.py on this directory."

#!/bin/bash
# Run tt-npe profiling across ResNet50 stages 0..4 with per-stage log/report isolation.
#
# Safe for shared containers: does NOT wipe /tt-metal/generated/profiler/.logs/,
# only moves files created during our own run (via mtime marker). Report dirs
# are matched by name-diff against a pre-run snapshot to avoid picking up
# concurrent runs by teammates.
#
# Prereq inside container:
#   source /tt-metal/env_set.sh
#   source /tt-npe/ENV_SETUP
#
# Usage:
#   bash /tt-metal/models/bos_model/resnet50_jbpark/run_stage_split.sh
#   bash /tt-metal/models/bos_model/resnet50_jbpark/run_stage_split.sh 3    # single stage
#
# Outputs:
#   /tt-metal/generated/profiler/stage_split_<ts>/stage_<N>/logs        (raw noc traces created during our run)
#   /tt-metal/generated/profiler/stage_split_<ts>/stage_<N>/report      (report dir created during our run)

set -e

cd /tt-metal

BASE_LOGS=generated/profiler/.logs
BASE_REPORTS=generated/profiler/reports
OUT_DIR=generated/profiler/stage_split_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"

# Stages to run: all 0..4 by default; or a single stage if arg provided
if [ -n "$1" ]; then
    STAGES="$1"
else
    STAGES="0 1 2 3 4"
fi

# Ensure base dirs exist so subsequent tooling doesn't fail
mkdir -p "$BASE_LOGS" "$BASE_REPORTS"

for STAGE in $STAGES; do
    echo "================================================================"
    echo "==== Running RESNET_STAGE=$STAGE"
    echo "================================================================"

    # --- SAFETY: mark start-of-run time & snapshot existing report dirs ---
    MARKER=$(mktemp)
    touch "$MARKER"
    # Snapshot existing report dir names (so we skip those when collecting)
    REPORT_SNAPSHOT=$(mktemp)
    (cd "$BASE_REPORTS" && ls -1 2>/dev/null | sort) > "$REPORT_SNAPSHOT" || true

    # --- Run profiled pytest ---
    RESNET_STAGE=$STAGE TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 \
        tools/tracy/profile_this.py \
        --collect-noc-traces \
        -c "pytest models/bos_model/resnet50_jbpark/tests/test_ttnn_functional_resnet50.py -k batch_1"

    STAGE_OUT="$OUT_DIR/stage_$STAGE"
    mkdir -p "$STAGE_OUT/logs"

    # --- Collect only OUR .logs/ files (created after MARKER) ---
    NEW_LOG_COUNT=$(find "$BASE_LOGS" -maxdepth 1 -type f -newer "$MARKER" 2>/dev/null | wc -l)
    if [ "$NEW_LOG_COUNT" -gt 0 ]; then
        find "$BASE_LOGS" -maxdepth 1 -type f -newer "$MARKER" -exec mv {} "$STAGE_OUT/logs/" \;
        echo "==== Moved $NEW_LOG_COUNT new .logs files → $STAGE_OUT/logs/"
    else
        echo "WARN: no new .logs files created for STAGE=$STAGE"
    fi

    # --- Collect only NEW report dirs (name-diff against snapshot) ---
    REPORT_NOW=$(mktemp)
    (cd "$BASE_REPORTS" && ls -1 2>/dev/null | sort) > "$REPORT_NOW" || true
    NEW_REPORTS=$(comm -13 "$REPORT_SNAPSHOT" "$REPORT_NOW")

    if [ -n "$NEW_REPORTS" ]; then
        for rep in $NEW_REPORTS; do
            if [ -d "$BASE_REPORTS/$rep" ]; then
                mv "$BASE_REPORTS/$rep" "$STAGE_OUT/report_$rep"
                echo "==== Moved report dir → $STAGE_OUT/report_$rep"
            fi
        done
    else
        echo "WARN: no new report dir found for STAGE=$STAGE (op ceiling exceeded? Tracy stitching may have failed)"
    fi

    rm -f "$MARKER" "$REPORT_SNAPSHOT" "$REPORT_NOW"
    echo "==== STAGE=$STAGE outputs → $STAGE_OUT"
done

echo ""
echo "================================================================"
echo "All stages done."
echo "Output root: $OUT_DIR"
echo ""
echo "To analyze raw traces per stage:"
echo "  for s in $STAGES; do"
echo "    npe_analyze_noc_trace_dir.py $OUT_DIR/stage_\$s/logs -e -m 200 \\"
echo "      > $OUT_DIR/stage_\$s/summary.txt 2>&1"
echo "  done"
echo "================================================================"

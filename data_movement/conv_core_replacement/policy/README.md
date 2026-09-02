# Core-remapping policy workspace

This directory contains the board-independent calibration and scheduling work
for the ResNet50 core-remapping policy. Device implementation and correctness
tests remain in the parent directory; raw profiling reports remain immutable at
their original capture location.

## Layout

```text
policy/
├── README.md
├── RESULTS.md
├── SCHEDULER_DESIGN.md
├── build_full_model_placement_domains.py
├── extract_full_model_trisc_exposure.py
├── generate_full_model_candidates.py
├── optimize_full_model_schedule.py
├── requirements.txt
├── validate_policy_outputs.py
└── data/
    └── full_model_baseline/
        ├── README.md
        ├── input/                # immutable CSVs from one full-model run
        └── output/               # generated calibration tables
```

## Input contract

Calibration accepts one uninterrupted full-model profiling report containing:

```text
profile_log_device.csv
ops_perf_results_*.csv
```

## Validated full-model latency comparison

Do not use the H=20 MILP manifest for board latency measurement. Its phase 0
deadlocked at `layer3.module2.conv3`, and later operation-placement pairs were
never reached. The latency comparison is intentionally restricted to the two
full-model paths that already completed on board: the stock environment
baseline and the phase-0 Layer3 cyclic-column placement.

After a clean board reset, capture the two paths separately. Do not launch the
second command if the first command times out or fails:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_validated_latency_guarded.sh \
  stock --clean-device-confirmed

bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_validated_latency_guarded.sh \
  remapped --clean-device-confirmed
```

Both commands retain the per-Conv synchronization used by the successful
validation runs. Compare profiler device durations rather than pytest wall
time:

```bash
python tests/tt_metal/tt_metal/data_movement/conv_core_replacement/policy/extract_validated_latency_comparison.py \
  --profile-root generated/profiler/conv_core_validated_latency \
  --output-dir generated/profiler/conv_core_validated_latency/comparison
```

The extractor reads `profile_log_device.csv` directly, so a Tracy timeout
during `ops_perf_results` post-processing does not invalidate a completed
device run. Use the summed per-operation TRISC critical span as the latency
proxy and the summed TRISC engine cycles as its compute-exposure companion.
These are device-kernel metrics, not an end-to-end production latency claim.

Both files must come from the same run. `stage_split_*` inputs are rejected:
those captures were split for NPE/DRAM-register collection and do not preserve
the production full-model graph or layer boundaries.

The full-model CSV pair currently used for development is stored together at:

```text
data_movement/conv_core_replacement/policy/data/full_model_baseline/input/
```

The larger Tracy/NPE capture archive remains under
`study_note/aging_simulation/resnet50_batch1_report/`; policy code does not
depend on those files.

## Local environment

Create a small Python environment inside this policy workspace. No TT-Metal
build, device runtime, or container is required.

From Windows PowerShell at the repository root:

```powershell
python -m venv data_movement\conv_core_replacement\policy\.venv
data_movement\conv_core_replacement\policy\.venv\Scripts\python.exe -m pip install -r data_movement\conv_core_replacement\policy\requirements.txt
```

If `python` is not installed locally, install a normal Windows Python runtime
first. Do not copy profiling data into the TT-Metal device container merely to
run this analysis.

## Run the extractor

From Windows PowerShell at the repository root:

```powershell
data_movement\conv_core_replacement\policy\.venv\Scripts\python.exe `
  data_movement\conv_core_replacement\policy\extract_full_model_trisc_exposure.py `
  --report-dir data_movement\conv_core_replacement\policy\data\full_model_baseline\input `
  --output-dir data_movement\conv_core_replacement\policy\data\full_model_baseline\output
```

This is offline CSV processing. It does not open a device, run pytest, require
a CMake build, or require a container.

Generated files:

```text
full_model_trisc_per_op_per_worker.csv
full_model_trisc_op_summary.csv
full_model_trisc_model_compute_costs.csv
full_model_compute_inventory.csv
```

Build the model-specific placement domains from those calibration tables:

```powershell
data_movement\conv_core_replacement\policy\.venv\Scripts\python.exe `
  data_movement\conv_core_replacement\policy\build_full_model_placement_domains.py `
  --calibration-dir data_movement\conv_core_replacement\policy\data\full_model_baseline\output
```

Enumerate the enabled placement slice and construct predicted candidate costs:

```powershell
data_movement\conv_core_replacement\policy\.venv\Scripts\python.exe `
  data_movement\conv_core_replacement\policy\generate_full_model_candidates.py `
  --calibration-dir data_movement\conv_core_replacement\policy\data\full_model_baseline\output
```

Generate the initial predicted `H=20` schedule:

```powershell
data_movement\conv_core_replacement\policy\.venv\Scripts\python.exe `
  data_movement\conv_core_replacement\policy\optimize_full_model_schedule.py `
  --calibration-dir data_movement\conv_core_replacement\policy\data\full_model_baseline\output `
  --horizon 20 `
  --solver-time-limit 120
```

The optimizer writes stock, the existing Layer3 cyclic baseline, and the
predicted full-model comparison together with five lexicographic solver stages:
maximum load, absolute deviation, extra axis segments, used placement programs,
and cyclic phase switches. The highspy interface passes each incumbent to the
next stage as a MIP warm start. See `RESULTS.md` for the current result, global
solver gaps, and validation boundary. The generated manifest is intentionally
marked non-deployable until candidate cycle permutation and pending chain
classes have been measured.

Validate all generated cross-file references and load reconstruction:

```powershell
data_movement\conv_core_replacement\policy\.venv\Scripts\python.exe `
  data_movement\conv_core_replacement\policy\validate_policy_outputs.py `
  --calibration-dir data_movement\conv_core_replacement\policy\data\full_model_baseline\output `
  --experiment-dir data_movement\conv_core_replacement\policy\data\experiments\single_axis_split `
  --horizon 20
```

Only `TRISC-KERNEL` intervals are used. `TRISC-FW` is an enclosing profiler
zone and must not be added to its nested kernel interval.

## Guarded full-model board validation

The generated `H=20` manifest is a predicted policy until the real full model
has executed it. Do not start with all phases. After rebuilding `ttnn` and a
clean device reset, validate one phase first from `/tt-metal`:

```bash
CONV_CORE_MILP_PHASE=0 \
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_full_model_guarded.sh \
  milp_phase_full_model --clean-device-confirmed
```

The deploy-validation manifest intentionally omits
`layer2.module1.downsample`. That operation is a HEIGHT_SHARDED 1x1 Conv2d,
which is not one of the executable explicit-remapping paths, so it remains at
its stock placement in every phase. The remaining 16 domains retain their
existing H=20 choices; this removal was not followed by MILP re-optimization,
so board profiling—not the earlier 26.5% prediction—is the result of record.

This correctness run synchronizes after every Conv so the final completed Conv
is visible if a device program stalls. It verifies all 17 selected domains,
the `Conv2 -> Conv3` remapping lifetime, HEIGHT_SHARDED exit-to-stock behavior,
and final-model PCC. A watchdog exit (`124` or `137`) means no further device
work may be submitted before a board reset.

Only after the validation run returns normally may the complete period be
profiled:

```bash
bash tests/tt_metal/tt_metal/data_movement/conv_core_replacement/run_bos_resnet50_milp_profile_period_guarded.sh \
  --clean-device-confirmed --run-full-period
```

The period runner launches 20 independent full-model processes, stops at the
first failure, disables the per-Conv diagnostic synchronization, and archives
one `profile_log_device.csv` plus one `ops_perf_results_*.csv` per phase. It
also copies the exact manifest/candidate table and their SHA-256 hashes into
the capture directory.

After copying that capture directory to the Windows policy workspace, extract
the measured physical-worker exposure:

```powershell
data_movement\conv_core_replacement\policy\.venv\Scripts\python.exe `
  data_movement\conv_core_replacement\policy\extract_milp_period_trisc_exposure.py `
  --period-dir <copied-period-directory> `
  --output-dir data_movement\conv_core_replacement\policy\data\full_model_measured\single_axis_split
```

The primary result is measured cumulative `TRISC-KERNEL` exposure over all 20
phases. Host timing from this instrumented pytest path is diagnostic rather
than a production latency claim.

## Boundary with the existing aging analysis

`study_note/aging_simulation/compute_axis.py` remains the baseline analysis
entry point. It now uses the same `TRISC-KERNEL` definition, but policy code
must not import analysis scripts or depend on stage-split artifacts.

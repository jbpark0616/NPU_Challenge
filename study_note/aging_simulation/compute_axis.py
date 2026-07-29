"""Extract per-core compute (TRISC) activity from profile_log_device.csv.

Output: alpha_compute per core (normalized to worst=1) to feed into aging_sim.py
as a third aging axis.

Method:
  1. Load profile_log_device.csv (per-core per-RISC zone timing).
  2. Filter to TRISC_0/1/2 (compute engines; BRISC/NCRISC are data movement,
     already captured in NIU axis).
  3. Match ZONE_START/ZONE_END by (core_x, core_y, RISC, timer_id, run_host_id).
  4. Sum active durations per core across all 3 TRISCs.
  5. Normalize to alpha_compute in (0, 1].
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path('C:/NPU_Challenge')
DEVICE_CSV = REPO / 'study_note/aging_simulation/resnet50_batch1_report/profile_log_device.csv'
OUT_CSV    = REPO / 'study_note/aging_simulation/plots/per_core_compute.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Match aging_sim.py compute filter (yaml-verified)
COMPUTE_Y = {0, 1, 2, 4}
COMPUTE_X = set(range(5))
TRISC_TYPES = {'TRISC_0', 'TRISC_1', 'TRISC_2'}


def load_device_log():
    df = pd.read_csv(DEVICE_CSV, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_active_cycles(df):
    """For each (core_x, core_y, RISC), pair ZONE_START <-> ZONE_END by
       (timer_id, run_host_id) and sum durations."""
    df = df[df['RISC processor type'].isin(TRISC_TYPES)].copy()
    df = df[(df['core_x'].isin(COMPUTE_X)) & (df['core_y'].isin(COMPUTE_Y))]

    starts = df[df['type'] == 'ZONE_START'][
        ['core_x', 'core_y', 'RISC processor type', 'timer_id', 'run host ID', 'time[cycles since reset]']
    ].rename(columns={'time[cycles since reset]': 't_start'})
    ends = df[df['type'] == 'ZONE_END'][
        ['core_x', 'core_y', 'RISC processor type', 'timer_id', 'run host ID', 'time[cycles since reset]']
    ].rename(columns={'time[cycles since reset]': 't_end'})

    key = ['core_x', 'core_y', 'RISC processor type', 'timer_id', 'run host ID']
    paired = starts.merge(ends, on=key, how='inner')
    paired['duration'] = paired['t_end'] - paired['t_start']
    # Drop non-positive (bad pairing) and outliers
    paired = paired[paired['duration'] > 0]

    per_core = paired.groupby(['core_x', 'core_y'])['duration'].sum().reset_index()
    per_core.columns = ['sx', 'sy', 'trisc_active_cycles']
    return per_core, paired['duration'].sum()


def add_missing_cores(per_core):
    """Ensure all 20 compute cores appear (some may not have run any TRISC op)."""
    full = pd.DataFrame(
        [(x, y) for y in sorted(COMPUTE_Y) for x in sorted(COMPUTE_X)],
        columns=['sx', 'sy']
    )
    return full.merge(per_core, on=['sx', 'sy'], how='left').fillna(0)


def normalize_alpha(cycles):
    m = cycles.max()
    if m == 0:
        return cycles
    return (cycles / m).clip(lower=1e-6)


def main():
    df = load_device_log()
    per_core, total_cycles = compute_active_cycles(df)
    per_core = add_missing_cores(per_core)
    per_core['alpha_compute'] = normalize_alpha(per_core['trisc_active_cycles'])

    per_core = per_core.sort_values(['sy', 'sx']).reset_index(drop=True)
    print(f'Total TRISC-active cycles (all cores): {total_cycles:,.0f}')
    print(f'Compute cores loaded: {len(per_core)}')
    print()
    print('per-core TRISC active cycles:')
    print(per_core.to_string(index=False))

    a = per_core['alpha_compute']
    print()
    print(f'alpha_compute: min={a.min():.4f}  max={a.max():.4f}  ratio={a.max()/a.min():.2f}x')

    per_core.to_csv(OUT_CSV, index=False)
    print(f'\nWritten: {OUT_CSV}')

if __name__ == '__main__':
    main()

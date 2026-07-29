"""
Baseline aging simulation for Eagle-N NPU (Blackhole class).

Chen et al. (ISOCC 2024) NBTI/HCI equations, but ONLY defensible inputs are
kept explicit. All process/material/environment constants are lumped into
opaque K_NBTI, K_HCI coefficients (which we do NOT quantify).

Consequences:
  - Absolute aging values are NOT reported (units = arbitrary).
  - Only RATIOS across cores / RATIOS across scenarios (baseline vs rotation)
    are defensible — the K coefficients cancel in ratios.

Chen Eq (1)(2) original:
    dVth_NBTI = A_NBTI * exp(-Ea/kT) * V^4 * t^(1/6) * u^(1/6)
    dVth_HCI  = A_HCI  * u * f * exp(-Eb/kT) * t^(1/6)

Collapsed form used here:
    K_NBTI := A_NBTI * exp(-Ea/kT) * V^4        (unknown material constant, lumped)
    K_HCI  := A_HCI  * exp(-Eb/kT)               (unknown material constant, lumped)

    dVth_NBTI = K_NBTI * t^(1/6) * u^(1/6)
    dVth_HCI  = K_HCI  * u * f * t^(1/6)

We report the K_NBTI=K_HCI=1 result as "aging index [a.u.]" and only
interpret ratios.

Defensible inputs:
  - u : activity factor (measured per-core NoC bytes / cycles, normalized 0..1)
  - f : clock frequency 650 MHz (measured from profile_log_device.csv)
  - t : simulation time (parameter, seconds)

Future work (moved out of scope for undergrad meeting):
  - Calibrate K_NBTI, K_HCI to silicon: requires TSMC 6nm process data or
    reference to Facelift MICRO'08 / Oboril DSN'12 numeric constants.
  - Include temperature sensitivity sweep (currently lumped into K).
  - Include voltage variation (currently lumped into K_NBTI via V^4).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication typography (Brandalero DAC'20 / Chen ISOCC'24 figure style)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Chen/Brandalero-style palette: viridis (perceptually uniform, colorblind-safe)
CMAP_HEAT = 'viridis'   # for aging intensity heatmaps
CMAP_HEAT_R = 'viridis_r'
COLOR_WORST = '#d62728'   # red (worst core)
COLOR_MEAN  = '#333333'   # neutral dark
COLOR_BEST  = '#2ca02c'   # green (best core)
COLOR_NBTI  = '#1f77b4'   # blue for NBTI series
COLOR_HCI   = '#d62728'   # red for HCI series

# ---------------------------------------------------------------------------
# Defensible inputs only
# ---------------------------------------------------------------------------
F_CLK    = 650e6                      # Hz, measured (profile_log_device.csv header)
YEAR_SEC = 365 * 24 * 3600

# Unknown material coefficients — set to 1 so we report unitless "aging index".
# Only ratios are meaningful; absolute values are arbitrary.
K_NBTI = 1.0
K_HCI  = 1.0

# ---------------------------------------------------------------------------
# Chen equations, collapsed form (K absorbs all process/env unknowns)
# ---------------------------------------------------------------------------
def vth_nbti(u, t_sec):
    """NBTI aging index [a.u.]. Multiply by real K_NBTI to get Volts."""
    return K_NBTI * (t_sec ** (1/6)) * (u ** (1/6))

def vth_hci(u, t_sec, f=F_CLK):
    """HCI aging index [a.u.]. Multiply by real K_HCI to get Volts."""
    return K_HCI * u * f * (t_sec ** (1/6))

def vth_total(u, t_sec):
    return vth_nbti(u, t_sec) + vth_hci(u, t_sec)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
REPO = Path('C:/NPU_Challenge')
NOC_CSV     = REPO / 'bos_model_copy/resnet50_jbpark/analysis/outputs/per_core.csv'
COMPUTE_CSV = REPO / 'study_note/aging_simulation/plots/per_core_compute.csv'

# Eagle-N compute cores per BOS blackhole_140_arch.yaml:
#   functional_workers = y in {0,1,2,4} x x in {0..5} = 24 workers
#   y=3 is DRAM row (memory, no compute)
#   x=5 is dispatch/reserved (injected=0, ops_as_src=0 in trace)
# ttnn logical grid (0..4, 0..3) = 20 compute cores = x in {0..4}, y in {0,1,2,4}.
COMPUTE_Y = {0, 1, 2, 4}
COMPUTE_X = set(range(5))

def load_compute_cores():
    """Return DataFrame filtered to 20 compute cores (per BOS yaml).
       Includes NIU, SRAM (from NoC trace) + Compute (from TRISC device log)."""
    df = pd.read_csv(NOC_CSV)
    df = df[df['sy'].isin(COMPUTE_Y) & df['sx'].isin(COMPUTE_X)].copy()
    df['total_noc'] = df['injected_bytes'] + df['consumed_bytes']

    if COMPUTE_CSV.exists():
        cmp_df = pd.read_csv(COMPUTE_CSV)
        df = df.merge(cmp_df[['sx', 'sy', 'trisc_active_cycles']], on=['sx', 'sy'], how='left')
    else:
        df['trisc_active_cycles'] = 0
        print(f'WARN: {COMPUTE_CSV} not found. Run compute_axis.py first.')

    return df.reset_index(drop=True)

def normalize_alpha(series):
    """Map raw activity to alpha in (0, 1]. Uses max normalization."""
    max_v = series.max()
    return (series / max_v).clip(lower=1e-6)  # avoid u=0 in u^(1/6)

# ---------------------------------------------------------------------------
# Baseline simulation
# ---------------------------------------------------------------------------
def simulate_baseline(years=(1, 2, 3, 5, 7, 10)):
    df = load_compute_cores()

    df['alpha_niu']     = normalize_alpha(df['total_noc'])
    df['alpha_sram']    = normalize_alpha(df['consumed_bytes'])
    df['alpha_compute'] = normalize_alpha(df['trisc_active_cycles'])

    result = {'df': df, 'years': list(years), 'axes': ['niu', 'sram', 'compute']}
    for axis in result['axes']:
        u = df[f'alpha_{axis}'].values
        # per-core Vth shift over time [num_cores x num_years]
        vth_matrix = np.zeros((len(u), len(years)))
        for j, y in enumerate(years):
            t_sec = y * YEAR_SEC
            vth_matrix[:, j] = vth_nbti(u, t_sec) + vth_hci(u, t_sec)
        # Normalize each time-column so max core = 1.0 (unit removed, ratio kept)
        # This makes the plot readable and drives home "relative only".
        result[f'vth_{axis}'] = vth_matrix
        result[f'vth_{axis}_rel'] = vth_matrix / vth_matrix.max(axis=0, keepdims=True)
    return result

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
OUT_DIR = REPO / 'study_note/aging_simulation/plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _core_grid_matrix(df, value_col):
    """Reshape per-core values to (row=y, col=x) matrix for heatmap."""
    y_list = sorted(COMPUTE_Y)                       # [1, 2, 4]
    x_list = sorted(COMPUTE_X)                       # [0..4]
    mat = np.full((len(y_list), len(x_list)), np.nan)
    for _, row in df.iterrows():
        yi = y_list.index(int(row['sy']))
        xi = x_list.index(int(row['sx']))
        mat[yi, xi] = row[value_col]
    return mat, y_list, x_list

def plot_alpha_heatmaps(df):
    axes_info = [
        ('niu',     'NIU  (injected + consumed)'),
        ('sram',    'SRAM  (consumed only)'),
        ('compute', 'Compute  (TRISC active cycles)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, (axis_name, title) in zip(axes, axes_info):
        col = f'alpha_{axis_name}'
        mat, y_list, x_list = _core_grid_matrix(df, col)
        im = ax.imshow(mat, cmap=CMAP_HEAT, vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(len(x_list))); ax.set_xticklabels(x_list)
        ax.set_yticks(range(len(y_list))); ax.set_yticklabels(y_list)
        ax.set_xlabel('core_x'); ax.set_ylabel('core_y')
        r = df[col].max() / df[col].min()
        ax.set_title(f'{title}\nspread = {r:.1f}×', pad=6, fontsize=10)
        for (i, j), v in np.ndenumerate(mat):
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='white' if v < 0.6 else '#111', fontsize=8.5, fontweight='medium')
        cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
        cb.set_label('activity factor  α', rotation=90, labelpad=10)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_edgecolor('#cccccc')
    plt.suptitle('Baseline per-core activity — 3 axes (ResNet50 batch=1, 20 compute cores)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / 'baseline_alpha_heatmap.png'
    plt.savefig(out)
    plt.close()
    return out

def plot_vth_heatmap(result, axis_name, year_idx):
    """NBTI vs HCI heatmaps side by side, cell text = normalized (worst=1)."""
    df = result['df']
    year = result['years'][year_idx]
    t_sec = year * YEAR_SEC
    u = df[f'alpha_{axis_name}'].values

    nbti_vals = (t_sec ** (1/6)) * (u ** (1/6))
    hci_vals  = u * F_CLK * (t_sec ** (1/6))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (mech, vals, K_symbol) in zip(axes, [
        ('NBTI', nbti_vals, 'K_NBTI'),
        ('HCI',  hci_vals,  'K_HCI'),
    ]):
        dfc = df.copy(); dfc['aging'] = vals
        mat, y_list, x_list = _core_grid_matrix(dfc, 'aging')
        vmax = float(np.nanmax(mat))
        mat_norm = mat / vmax
        im = ax.imshow(mat_norm, cmap=CMAP_HEAT, vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(len(x_list))); ax.set_xticklabels(x_list)
        ax.set_yticks(range(len(y_list))); ax.set_yticklabels(y_list)
        ax.set_xlabel('core_x'); ax.set_ylabel('core_y')
        u_max, u_min = u.max(), u.min()
        if mech == 'NBTI':
            spread = (u_max / u_min) ** (1/6)
        else:
            spread = u_max / u_min
        ax.set_title(f'{mech}  [ × {K_symbol} ]   worst/best = {spread:.2f}×', pad=8)
        for (i, j), v in np.ndenumerate(mat_norm):
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color='white' if v < 0.6 else '#111', fontsize=9, fontweight='medium')
        cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
        cb.set_label(f'normalized aging  (worst = 1)', rotation=90, labelpad=10)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_edgecolor('#cccccc')

    fig.suptitle(f'Baseline aging @ {year} years — {axis_name.upper()} axis',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / f'baseline_vth_{axis_name}_year{year}.png'
    plt.savefig(out)
    plt.close()
    return out

def plot_worst_vs_best_over_time(result):
    """Option B: symbolic K, y-axis in units of K_NBTI or K_HCI.
       4 panels: (NIU, SRAM) x (NBTI, HCI). Each shows worst/mean/best
       aging index over time — grows as t^(1/6). Absolute magnitude
       requires calibration of K_NBTI or K_HCI (future work).
    """
    df = result['df']
    years = np.array(result['years'], dtype=float)
    t_sec = years * YEAR_SEC
    axes_list = ('niu', 'sram', 'compute')
    fig, axes = plt.subplots(len(axes_list), 2, figsize=(12, 4 * len(axes_list)))

    for row, axis_name in enumerate(axes_list):
        u = df[f'alpha_{axis_name}'].values

        # NBTI: aging / K_NBTI = t^(1/6) * u^(1/6)
        nbti_per_core = np.outer(u ** (1/6), t_sec ** (1/6))     # (cores, years)
        # HCI: aging / K_HCI = u * f * t^(1/6)
        hci_per_core  = np.outer(u * F_CLK, t_sec ** (1/6))

        for col, (mech, per_core, K_symbol) in enumerate([
            ('NBTI', nbti_per_core, 'K_NBTI'),
            ('HCI',  hci_per_core,  'K_HCI'),
        ]):
            ax = axes[row, col]
            worst = per_core.max(axis=0)
            mean  = per_core.mean(axis=0)
            best  = per_core.min(axis=0)

            ax.plot(years, worst, color=COLOR_WORST, marker='o', markersize=5,
                    linewidth=1.8, label='worst core')
            ax.plot(years, mean,  color=COLOR_MEAN,  linestyle='--', linewidth=1.4,
                    label='mean')
            ax.plot(years, best,  color=COLOR_BEST,  marker='o', markersize=5,
                    linewidth=1.8, label='best core')
            ax.fill_between(years, best, worst, alpha=0.10, color=COLOR_WORST)

            ax.set_xlabel('Age [years]')
            ax.set_ylabel(f'aging index  [ × {K_symbol} ]')
            ax.set_title(f'{axis_name.upper()}  —  {mech}    (grows as t^(1/6))', pad=6)
            ax.grid(alpha=0.4)
            ax.legend(fontsize=8, loc='lower right')

    plt.suptitle('Aging over time  —  symbolic-K approach\n'
                 'Shape (t^(1/6)) and relative core positions are exact.\n'
                 'Absolute magnitude scales with unknown K_NBTI, K_HCI  (6nm calibration = future work)',
                 fontsize=11, y=1.00)
    plt.tight_layout()

def plot_worst_vs_best_over_time_split(result):
    """Split version: one PNG per axis (NIU / SRAM / Compute).
       Each PNG has 2 panels (NBTI + HCI) — smaller, PPT-friendly.
    """
    df = result['df']
    years = np.array(result['years'], dtype=float)
    t_sec = years * YEAR_SEC
    outs = []

    for axis_name in ('niu', 'sram', 'compute'):
        u = df[f'alpha_{axis_name}'].values
        nbti_per_core = np.outer(u ** (1/6), t_sec ** (1/6))
        hci_per_core  = np.outer(u * F_CLK, t_sec ** (1/6))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (mech, per_core, K_symbol) in zip(axes, [
            ('NBTI', nbti_per_core, 'K_NBTI'),
            ('HCI',  hci_per_core,  'K_HCI'),
        ]):
            worst = per_core.max(axis=0)
            mean  = per_core.mean(axis=0)
            best  = per_core.min(axis=0)

            ax.plot(years, worst, color=COLOR_WORST, marker='o', markersize=6,
                    linewidth=2.0, label='worst core')
            ax.plot(years, mean,  color=COLOR_MEAN,  linestyle='--', linewidth=1.5,
                    label='mean')
            ax.plot(years, best,  color=COLOR_BEST,  marker='o', markersize=6,
                    linewidth=2.0, label='best core')
            ax.fill_between(years, best, worst, alpha=0.10, color=COLOR_WORST)

            ax.set_xlabel('Age [years]')
            ax.set_ylabel(f'aging index  [ × {K_symbol} ]')
            ax.set_title(f'{mech}    (grows as t^(1/6))', pad=6, fontsize=11)
            ax.grid(alpha=0.4)
            ax.legend(fontsize=9, loc='lower right')

        plt.suptitle(f'{axis_name.upper()} axis — aging over time (symbolic K)',
                     fontsize=12.5, y=1.02)
        plt.tight_layout()
        out = OUT_DIR / f'baseline_vth_over_time_{axis_name}.png'
        plt.savefig(out)
        plt.close()
        outs.append(out)
    return outs
    out = OUT_DIR / 'baseline_vth_over_time.png'
    plt.savefig(out)
    plt.close()
    return out

def plot_distribution(result, year=5):
    year_idx = result['years'].index(year)
    axes_list = ('niu', 'sram', 'compute')
    fig, axes = plt.subplots(1, len(axes_list), figsize=(5 * len(axes_list), 4))
    for ax, axis_name in zip(axes, axes_list):
        vth = result[f'vth_{axis_name}_rel'][:, year_idx]  # 0..1
        ax.hist(vth, bins=10, color=plt.get_cmap(CMAP_HEAT)(0.55),
                edgecolor='#333', alpha=0.85)
        ax.axvline(vth.mean(), color=COLOR_MEAN, linestyle='--', linewidth=1.5,
                   label=f'mean {vth.mean():.2f}')
        ax.axvline(vth.max(),  color=COLOR_WORST, linestyle='--', linewidth=1.5,
                   label=f'max  {vth.max():.2f}')
        ax.axvline(vth.min(),  color=COLOR_BEST, linestyle='--', linewidth=1.5,
                   label=f'min  {vth.min():.2f}')
        ax.set_xlabel('normalized aging  (worst = 1)')
        ax.set_ylabel('# cores')
        ax.set_title(f'{axis_name.upper()} axis  @  {year} years', pad=6)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)
    plt.suptitle(f'Per-core aging distribution @ {year} years — baseline',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / f'baseline_vth_dist_year{year}.png'
    plt.savefig(out)
    plt.close()
    return out

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def summarize(result):
    df = result['df']
    lines = ['=== Baseline aging simulation summary (relative-only) ===',
             f'Model:      Chen et al. ISOCC 2024 NBTI+HCI equations',
             f'Platform:   Eagle-N (Blackhole class), 20 compute cores',
             f'Workload:   ResNet50 batch=1, stage-split aggregated NoC data',
             f'Defensible: alpha (measured), f={F_CLK/1e6:.0f}MHz (measured), t (parameter)',
             f'Unknown  :  K_NBTI, K_HCI  -> collapsed. Only RATIOS reported.',
             '',
             'NBTI vs HCI are reported SEPARATELY because their sum depends on the',
             'unknown ratio K_NBTI/K_HCI. Each mechanism ratio individually is K-free.',
             '',
             f'{"axis":6s} {"mech":>5s} {"year":>4s}  {"worst/best":>12s}  {"mean/best":>12s}']
    for axis in ('niu', 'sram', 'compute'):
        u = df[f'alpha_{axis}'].values
        for y in result['years']:
            t = y * YEAR_SEC
            # NBTI-only ratio: u^(1/6) so ratio = (u_max/u_min)^(1/6)
            nbti = (t ** (1/6)) * (u ** (1/6))
            # HCI-only ratio: u linear, so ratio = u_max/u_min
            hci  = u * F_CLK * (t ** (1/6))
            for mech, v in (('NBTI', nbti), ('HCI', hci)):
                lines.append(f'{axis:6s} {mech:>5s} {y:4d}  {v.max()/v.min():12.2f}  {v.mean()/v.min():12.2f}')
        lines.append('')
    lines.append('Interpretation:')
    lines.append('  NBTI ratio is much softer than HCI ratio because NBTI ~ u^(1/6) while HCI ~ u.')
    lines.append('  Ratios are constant over time — the t^(1/6) factor is shared and cancels.')
    lines.append('  Absolute Vth values NOT reported — requires 6nm calibration (future work).')
    return '\n'.join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    result = simulate_baseline()
    df = result['df']

    print(f'\nLoaded {len(df)} compute cores.')
    print(f'  alpha_niu     (measured): min={df["alpha_niu"].min():.3f}  max={df["alpha_niu"].max():.3f}  ratio={df["alpha_niu"].max()/df["alpha_niu"].min():.2f}x')
    print(f'  alpha_sram    (measured): min={df["alpha_sram"].min():.3f}  max={df["alpha_sram"].max():.3f}  ratio={df["alpha_sram"].max()/df["alpha_sram"].min():.2f}x')
    print(f'  alpha_compute (measured): min={df["alpha_compute"].min():.3f}  max={df["alpha_compute"].max():.3f}  ratio={df["alpha_compute"].max()/df["alpha_compute"].min():.2f}x')
    print(f'  clock f       (measured): {F_CLK/1e6:.0f} MHz')
    print(f'  K_NBTI, K_HCI        : set to 1.0 (material constants collapsed, only ratios meaningful)')

    y5 = result['years'].index(5)
    plots = [plot_alpha_heatmaps(df),
             plot_vth_heatmap(result, 'niu',     y5),
             plot_vth_heatmap(result, 'sram',    y5),
             plot_vth_heatmap(result, 'compute', y5),
             plot_worst_vs_best_over_time(result),
             *plot_worst_vs_best_over_time_split(result),   # 3 PNGs (per axis)
             plot_distribution(result, year=5),
             plot_distribution(result, year=10)]

    print('\nPlots written:')
    for p in plots:
        print(f'  {p}')

    report = summarize(result)
    print('\n' + report)

    report_path = OUT_DIR / 'baseline_summary.txt'
    report_path.write_text(report, encoding='utf-8')
    print(f'\nReport: {report_path}')

if __name__ == '__main__':
    main()

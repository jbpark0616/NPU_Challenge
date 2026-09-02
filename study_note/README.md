# Architecture and Aging Research Notes

This directory contains the exploratory work that led to the programmable
core-remapping project.

## Main topics

- `Tenstorrent_Architecture_note.md` — device and software architecture notes.
- `stage_split_profiling_workflow.md` — profiling workflow and capture boundary.
- `aging_rotation_research_note.md` — development of the aging-aware rotation
  problem and its assumptions.
- `aging_simulation/` — compute/NoC exposure analysis, aging simulations, and
  generated plots.
- `spike_k_measurement/` — supporting measurement experiments.
- `figures/` — reusable architecture and research figures.

These documents are research records rather than a single polished report.
The current implementation and validated claims are summarized in
[`../data_movement/conv_core_replacement/`](../data_movement/conv_core_replacement/README.md).

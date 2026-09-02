# Full-model baseline dataset

This dataset is the single uninterrupted BOS ResNet50 batch-1 profiling run
used to calibrate the remapping policy.

```text
input/
├── profile_log_device.csv
└── ops_perf_results_2026_07_26_14_16_54.csv

output/
└── generated policy calibration CSVs
```

The two input CSVs are an inseparable pair from the same profiling run. Do not
replace only one of them, and do not substitute any `stage_split_*` capture.
The source capture's Tracy file and NPE visualizations remain archived under
`study_note/aging_simulation/resnet50_batch1_report/`.

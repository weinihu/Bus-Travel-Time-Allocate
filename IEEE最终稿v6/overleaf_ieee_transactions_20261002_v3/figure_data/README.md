# Figure source data

`followup_mae_differences.pdf` reports layout-level MAE differences from direct prediction for the validation-selected 960-update comparison. Negative differences indicate lower MAE. `ablation_panels.pdf` reports paired differences for the separate fixed-$\\rho=0.45$ controls and Singapore prefix study. Points show observed layouts or paired runs; error bars are sample standard deviations, not confidence intervals.

Astana conditions contain nine pairs from three fixed label layouts and three paired initializations. Singapore conditions contain three paired initializations. These repetitions are not independent datasets. Absolute errors remain in manuscript tables. `ablation_paired_differences.csv` provides the Astana differences directly.

Rebuild figures from the audited frozen observations without training:

```bash
python figure_data/build_followup_results_20261002.py --repo-root /path/to/BusSegPredicionV2
```

`visual_check.json` records checks for dimensions, fonts and clipping. PDF/SVG are vector outputs; PNG and grayscale PNG are review previews. The manuscript captions describe experiments and statistics; page-layout requirements do not belong in those captions.

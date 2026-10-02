#!/usr/bin/env python3
"""Render the saved median and upper-quantile interval cases with source data."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent / 'paper小论文/overleaf_ieee_transactions_20260930_final'


def main():
    from scripts.build_ieee_revision_evidence_20260930 import read_verified_prediction
    saved = ROOT / 'results/thesis_final_protocol_20260926/f0.1_m20260919'
    _, proposed, _ = read_verified_prediction(saved / 'ours/s42')
    _, direct, _ = read_verified_prediction(saved / 'direct_time/s42', proposed)
    evidence = json.loads((PACKAGE / 'evidence/teacher_revision.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 6.4,
        'axes.labelsize': 6.2, 'axes.titlesize': 6.5, 'legend.fontsize': 6.2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .65})
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.15), sharex='col',
        gridspec_kw={'height_ratios': [1.8, 1.]}, layout='constrained')
    rows = []
    for col, case in enumerate(evidence['cases']):
        row, n = case['row'], case['length']
        x = np.arange(1, n + 1)
        truth, pred, baseline = (a[row, :n] for a in
            (proposed['targets'], proposed['preds'], direct['preds']))
        for values, label, color, style in (
            (truth, 'Observed', '#222222', '-'),
            (pred, 'Ours', '#0072B2', '--'),
            (baseline, 'Direct prediction', '#D55E00', ':')):
            axes[0, col].plot(x, values, label=label, color=color,
                             linestyle=style, linewidth=1.)
        axes[0, col].set_title(f"({'a' if col == 0 else 'b'}) " +
            ('Median trip error' if col == 0 else '90th-percentile trip error'))
        axes[0, col].set_ylim(bottom=0)
        axes[1, col].axhline(0, color='0.5', linewidth=.6)
        axes[1, col].plot(x, pred - truth, color='#0072B2', linestyle='--', linewidth=1.)
        axes[1, col].plot(x, baseline - truth, color='#D55E00', linestyle=':', linewidth=1.)
        axes[1, col].set_xlabel('Ordered inter-stop interval')
        for ax in axes[:, col]:
            ax.set_xlim(1, n)
            ax.grid(axis='y', color='0.88', linewidth=.5)
            ax.set_axisbelow(True)
        for i in range(n):
            rows.append({'case': 'median' if col == 0 else 'percentile90',
                'interval': i + 1, 'observed_seconds': float(truth[i]),
                'proposed_seconds': float(pred[i]), 'direct_seconds': float(baseline[i]),
                'proposed_error_seconds': float(pred[i] - truth[i]),
                'direct_error_seconds': float(baseline[i] - truth[i])})
    axes[0, 0].set_ylabel('Running time (s)')
    axes[1, 0].set_ylabel('Signed error (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               handlelength=1.2, columnspacing=.8, handletextpad=.25)
    fig.get_layout_engine().set(rect=(0., 0., 1., .91))
    figures, data = PACKAGE / 'figures', PACKAGE / 'figure_data'
    figures.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    fig.savefig(figures / 'local_cases.pdf')
    fig.savefig(figures / 'local_cases.png', dpi=600)
    plt.close(fig)
    with (data / 'local_cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(__file__, data / Path(__file__).name)


if __name__ == '__main__':
    main()

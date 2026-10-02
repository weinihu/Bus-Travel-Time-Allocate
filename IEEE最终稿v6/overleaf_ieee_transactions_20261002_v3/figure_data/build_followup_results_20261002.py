#!/usr/bin/env python3
"""Build paper tables and a paired MAE figure from audited development results."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
METRICS = ("mae", "rmse", "wape", "prefix50", "top10_mae", "tail_mae")
METHODS = ("uniform", "identity_mean", "nnls_ridge_tuned", "tree",
           "direct_time", "ours", "blend")
LABELS = {
    "uniform": ("Uniform", "均匀分配"),
    "identity_mean": ("Identity means", "同段均值"),
    "nnls_ridge_tuned": ("NNLS-Ridge", "NNLS-Ridge"),
    "tree": ("ExtraTrees", "ExtraTrees"),
    "direct_time": ("Direct prediction", "Direct prediction"),
    "ours": ("Ours", "Ours"),
    "blend": (r"Ours + Direct ($\alpha=0.5$)", r"Ours + Direct ($\alpha=0.5$)"),
}
TABLE_METRIC_LABELS = {
    "en": ("MAE", "RMSE", "WAPE (\\%)", "Prefix50", "Top-10\\% MAE", "Long MAE"),
    "zh": ("MAE", "RMSE", "WAPE (\\%)", "Prefix50", "高贡献10\\% MAE", "长段MAE"),
}
METHOD_STYLE = {
    "identity_mean": ("#009E73", "s"),
    "nnls_ridge_tuned": ("#E69F00", "^"),
    "tree": ("#56B4E9", "D"),
    "ours": ("#D55E00", "o"),
    "blend": ("#CC79A7", "P"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def assert_development_only(*reports):
    for report in reports:
        protocol = report.get("protocol", {})
        if not isinstance(protocol, dict):
            protocol = {}
        if protocol.get("test_labels_opened", False):
            raise ValueError("a result report says test labels were opened")
        if protocol.get("fixed_or_sealed_test_truth_read", False):
            raise ValueError("a result report says fixed/sealed test truth was read")
        if protocol.get("development_labels_used_for_selection", False):
            raise ValueError("development labels were used for selection")
        if report.get("test_labels_opened", False) or report.get("fixed_or_sealed_test_truth_read", False):
            raise ValueError("a result report says fixed/sealed test truth was read")


def load_evidence(repo: Path):
    result_dir = repo / "results/paper_followup_20261002_rho_tree_v3"
    performance = load(repo / "results/paper_performance_20261002/baseline_recomputed.json")
    budget_audit = load(repo / "results/submission_revision_20261001/budget_audit.json")
    budget_gate = load(repo / "results/submission_revision_20261001/budget_gate.json")
    rho = load(result_dir / "budget_rho_confirmation.json")
    blend = load(result_dir / "blend_confirmation.json")
    tree = load(result_dir / "tree_confirmation.json")
    rho_selection = load(result_dir / "budget_rho_selection.json")
    blend_selection = load(result_dir / "blend_selection.json")
    tree_selection = load(result_dir / "tree_selection.json")

    assert_development_only(performance, budget_audit, rho, blend, tree)
    if budget_gate.get("status") != "closed":
        raise ValueError("the 5% baseline audit gate is not closed")
    if rho["selection"]["selection_sha256"] != rho_selection["selection_sha256"]:
        raise ValueError("rho confirmation does not reference the frozen selection")
    if blend.get("screen_sha256") != blend_selection["screen_sha256"]:
        raise ValueError("blend confirmation does not reference the frozen selection")
    if tree["screen_sha256"] != tree_selection["screen_sha256"]:
        raise ValueError("tree confirmation does not reference the frozen selection")

    records = {str(f): {m: [] for m in METHODS} for f in (0.01, 0.05, 0.1)}
    # The non-neural references are frozen by label mask; 5% comes from its closed audit.
    for row in performance["rows"]:
        if row["method"] == "identity_mean":
            records[str(row["fraction"])][row["method"]].append(row)
    records["0.05"]["identity_mean"] = budget_audit["summary"]["identity_mean"]["per_run"]
    records["0.05"]["nnls_ridge_tuned"] = budget_audit["summary"]["nnls_ridge_tuned"]["per_run"]
    sys_path = str(repo.resolve())
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from scripts.build_ieee_revision_evidence_20260930 import allocation_metrics
    nnls_sources = {}
    for fraction in ("0.01", "0.1"):
        for mask in (20260919, 20260920, 20260921):
            folder = repo / "results/thesis_final_protocol_20260926" / f"f{float(fraction):g}_m{mask}" / "nnls_ridge_tuned"
            result_path = folder / "result.json"
            prediction_path = folder / "predictions.npz"
            result = load(result_path)
            assert_development_only(result)
            with np.load(prediction_path, allow_pickle=False) as saved:
                metric = allocation_metrics(pred=saved["preds"], target=saved["targets"],
                    padding=saved["padding"], true_total=saved["total"])
            for name in ("mae", "rmse", "wape", "prefix50", "tail_mae"):
                if abs(metric[name] - result["metrics"][name]) > 2e-5:
                    raise ValueError(f"NNLS prediction replay differs for {fraction}/{mask}/{name}")
            records[fraction]["nnls_ridge_tuned"].append({**metric, "mask_seed": mask})
            nnls_sources[str(result_path.relative_to(repo))] = sha(result_path)
            nnls_sources[str(prediction_path.relative_to(repo))] = sha(prediction_path)

    uniform_row = dict(budget_audit["summary"]["uniform"]["per_run"][0],
                       mask_seed=20260919, seed=None)
    for fraction in ("0.01", "0.05", "0.1"):
        records[fraction]["uniform"] = [uniform_row]

    for row in rho["rows"]:
        records[str(row["fraction"])]["ours"].append({
            **row["ours"], "mask_seed": row["mask_seed"], "seed": row["seed"],
            "rho": row["rho"],
        })
        records[str(row["fraction"])]["direct_time"].append({
            **row["direct_time"], "mask_seed": row["mask_seed"], "seed": row["seed"],
        })
    for row in blend["rows"]:
        records[str(row["fraction"])]["blend"].append({
            **row["blend"], "mask_seed": row["mask_seed"], "seed": row["seed"],
            "alpha_ours": row["alpha_ours"],
        })
    for row in tree["rows"]:
        records[str(row["fraction"])]["tree"].append({
            **row["tree"], "mask_seed": row["mask_seed"],
            "min_samples_leaf": row["min_samples_leaf"],
            "paired_direct_mae": row["direct_time"]["mae"],
        })

    expected = {"uniform": 1, "identity_mean": 3, "nnls_ridge_tuned": 3,
                "tree": 3, "direct_time": 9, "ours": 9, "blend": 9}
    for budget, methods in records.items():
        for method, count in expected.items():
            if len(methods[method]) != count:
                raise ValueError(f"{budget} {method}: expected {count}, found {len(methods[method])}")
    selection_rho = rho_selection["selected_rho_by_fraction"]
    for fraction, rows in ((str(k), v) for k, v in records.items()):
        expected_rho = selection_rho[fraction]
        if any("rho" in row and abs(row["rho"] - expected_rho) > 1e-12 for row in rows["ours"]):
            raise ValueError(f"Ours rho mismatch at {fraction}")
        if any(abs(row["alpha_ours"] - .5) > 1e-12 for row in rows["blend"]):
            raise ValueError(f"blend weight mismatch at {fraction}")
    return records, {
        "budget_rho_confirmation.json": sha(result_dir / "budget_rho_confirmation.json"),
        "budget_rho_selection.json": sha(result_dir / "budget_rho_selection.json"),
        "blend_confirmation.json": sha(result_dir / "blend_confirmation.json"),
        "blend_selection.json": sha(result_dir / "blend_selection.json"),
        "tree_confirmation.json": sha(result_dir / "tree_confirmation.json"),
        "tree_selection.json": sha(result_dir / "tree_selection.json"),
        "baseline_recomputed.json": sha(repo / "results/paper_performance_20261002/baseline_recomputed.json"),
        "budget_audit.json": sha(repo / "results/submission_revision_20261001/budget_audit.json"),
        "budget_gate.json": sha(repo / "results/submission_revision_20261001/budget_gate.json"),
        **nnls_sources,
    }


def mean_sd(rows, metric):
    values = [float(r[metric]) for r in rows]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def tex_cell(mean, sd, bold=False, decimals=2, deterministic=False):
    value = f"{mean:.{decimals}f}"
    if not deterministic:
        value += rf"\pm{sd:.{decimals}f}"
    if bold:
        value = rf"\mathbf{{{value}}}"
    return f"${value}$"


def write_tables(records):
    for language in ("en", "zh"):
        rows = []
        for fraction in ("0.01", "0.05", "0.1"):
            summaries = {}
            for method in METHODS:
                summaries[method] = {}
                for metric in METRICS:
                    scale = 100.0 if metric == "wape" else 1.0
                    mean, sd = mean_sd(records[fraction][method], metric)
                    summaries[method][metric] = (mean * scale, sd * scale)
            minima = {metric: min(summaries[m][metric][0] for m in METHODS) for metric in METRICS}
            percent = int(float(fraction) * 100)
            if rows:
                rows.append(r"\midrule")
            for method in METHODS:
                values = []
                for metric in METRICS:
                    mean, sd = summaries[method][metric]
                    deterministic = method == "uniform"
                    values.append(tex_cell(mean, sd, mean == minima[metric], deterministic=deterministic))
                rows.append(f"{percent}\\% & {LABELS[method][language == 'zh']} & "
                            + " & ".join(values) + r"\\")
        if language == "en":
            caption = ("Astana development-set errors on 2,182 trips and 87,715 intervals. "
                       "Values are mean $\\pm$ sample SD except the deterministic uniform reference. "
                       "Neural methods and the fixed-weight combination use nine runs; identity and NNLS "
                       "use three label layouts; ExtraTrees uses three layouts.")
            header = "Budget & Method & MAE & RMSE & WAPE (\\%) & Prefix50 & Top-10\\% MAE & Long MAE"
        else:
            caption = ("Astana开发集误差：2,182趟、87,715段。除确定性均匀分配外，数值为均值$\\pm$样本标准差。" 
                       "神经方法及固定权重组合各九次运行；同段均值、NNLS和ExtraTrees各覆盖三个标签布局。")
            header = "预算 & 方法 & MAE & RMSE & WAPE (\\%) & Prefix50 & 高贡献10\\% MAE & 长段MAE"
        table = ("\\begin{table*}[t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n"
                 f"\\caption{{{caption}}}\\label{{tab:limited}}\n"
                 "\\begin{tabular}{llrrrrrr}\\toprule\n" + header + r"\\\midrule" + "\n"
                 + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\end{table*}\n")
        (PAPER / "tables" / language / "limited.tex").write_text(table)

        paired_rows = []
        for fraction in ("0.01", "0.05", "0.1"):
            ours = {(r["mask_seed"], r["seed"]): r["mae"] for r in records[fraction]["ours"]}
            direct = {(r["mask_seed"], r["seed"]): r["mae"] for r in records[fraction]["direct_time"]}
            differences = [ours[key] - direct[key] for key in sorted(ours)]
            if set(ours) != set(direct) or len(differences) != 9:
                raise ValueError(f"unmatched Ours/direct runs at {fraction}")
            percent = int(float(fraction) * 100)
            paired_rows.append(f"{percent}\\% & {statistics.mean(differences):+.3f} & "
                               f"{statistics.stdev(differences):.3f} & "
                               f"{sum(value < 0 for value in differences)}/9" + r"\\")
        if language == "en":
            paired_caption = ("Paired Ours-minus-direct MAE across the same three label layouts and three initializations. "
                              "Negative differences favor Ours; SD is across nine paired runs.")
            paired_header = "Budget & Mean difference (s) & SD (s) & Ours wins"
        else:
            paired_caption = ("同一三组标签布局与三次初始化下Ours减直接预测的配对MAE差。负值表示Ours误差较低；" 
                              "标准差按九组配对运行计算。")
            paired_header = "预算 & 平均差值（秒） & 标准差（秒） & Ours胜出数"
        paired_table = ("\\begin{table}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
                        f"\\caption{{{paired_caption}}}\\label{{tab:paired}}\n"
                        "\\begin{tabular}{lrrr}\\toprule\n" + paired_header + r"\\\midrule" + "\n"
                        + "\n".join(paired_rows) + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")
        (PAPER / "tables" / language / "paired.tex").write_text(paired_table)

        detail_rows = []
        for fraction in ("0.01", "0.05", "0.1"):
            percent = int(float(fraction) * 100)
            for method in ("ours", "direct_time"):
                cells = []
                for metric in METRICS:
                    scale = 100.0 if metric == "wape" else 1.0
                    mean, sd = mean_sd(records[fraction][method], metric)
                    cells.append(tex_cell(mean * scale, sd * scale, decimals=2))
                detail_rows.append(f"{percent}\\% & {LABELS[method][language == 'zh']} & "
                                   + " & ".join(cells) + r"\\")
            if fraction != "0.1":
                detail_rows.append(r"\midrule")
        if language == "en":
            detail_caption = ("Budget-specific Ours and direct-prediction confirmation with 960 updates and "
                              "validation-selected $\\rho$. Values are mean $\\pm$ sample SD over nine paired runs.")
            detail_header = "Budget & Method & MAE & RMSE & WAPE (\\%) & Prefix50 & Top-10\\% MAE & Long MAE"
        else:
            detail_caption = ("960步预算特定确认结果，$\\rho$由验证集选择。数值为九组配对运行的均值$\\pm$样本标准差。")
            detail_header = "预算 & 方法 & MAE & RMSE & WAPE (\\%) & Prefix50 & 高贡献10\\% MAE & 长段MAE"
        detail_table = ("\\begin{table*}[t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n"
                        f"\\caption{{{detail_caption}}}\\label{{tab:performance-confirmation}}\n"
                        "\\begin{tabular}{llrrrrrr}\\toprule\n" + detail_header + r"\\\midrule" + "\n"
                        + "\n".join(detail_rows) + "\n\\bottomrule\\end{tabular}\n\\end{table*}\n")
        (PAPER / "tables" / language / "performance_confirmation.tex").write_text(detail_table)


def make_plot(records, source_hashes):
    # For layout-level comparisons, average the three paired neural seeds within each mask.
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.65), sharey=True)
    fig.subplots_adjust(left=.22, right=.99, top=.88, bottom=.23, wspace=.12)
    candidates = ("identity_mean", "nnls_ridge_tuned", "tree", "ours", "blend")
    y_positions = np.arange(len(candidates))
    by_budget = {}
    for fraction, methods in records.items():
        direct_by_mask = {
            mask: statistics.mean(r["mae"] for r in methods["direct_time"] if r["mask_seed"] == mask)
            for mask in sorted({r["mask_seed"] for r in methods["direct_time"]})
        }
        points_by_method = {}
        for method in candidates:
            values = []
            for mask in direct_by_mask:
                rows = [r for r in methods[method] if r["mask_seed"] == mask]
                candidate = statistics.mean(r["mae"] for r in rows)
                if method == "tree":
                    direct = next(r["paired_direct_mae"] for r in rows)
                else:
                    direct = direct_by_mask[mask]
                values.append(candidate - direct)
            points_by_method[method] = values
        by_budget[fraction] = points_by_method
    maximum = max(abs(value) for data in by_budget.values() for vals in data.values() for value in vals)
    limit = max(1.0, np.ceil(maximum * 1.12 * 2) / 2)
    for ax, fraction in zip(axes, ("0.01", "0.05", "0.1")):
        ax.axvline(0, color=".15", linewidth=.75, zorder=1)
        ax.grid(axis="x", color=".88", linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        for index, method in enumerate(candidates):
            color, marker = METHOD_STYLE[method]
            values = np.asarray(by_budget[fraction][method], dtype=float)
            jitter = np.linspace(-.09, .09, len(values))
            ax.scatter(values, y_positions[index] + jitter, s=18, color=color,
                       marker=marker, edgecolor="white", linewidth=.4, zorder=3)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.
            ax.errorbar(mean, y_positions[index], xerr=sd, fmt="D", markersize=3.4,
                        color=color, ecolor=color, elinewidth=.9, capsize=2, zorder=4)
        ax.set_title(f"{int(float(fraction) * 100)}% labels", fontsize=8, pad=5)
        ax.set_xlim(-limit, limit)
        ax.set_xticks((-limit, 0., limit))
        ax.tick_params(axis="x", labelsize=7, length=2, pad=2)
        ax.tick_params(axis="y", labelsize=7, length=0, pad=3)
        ax.set_ylim(len(candidates) - .45, -.55)
    axes[0].set_yticks(y_positions, [LABELS[m][0] for m in candidates])
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(.65)
    fig.supxlabel("MAE difference from Direct (s); negative is lower", fontsize=7.5, y=.06)
    fig.savefig(PAPER / "figures" / "followup_mae_differences.pdf", bbox_inches="tight")
    fig.savefig(PAPER / "figures" / "followup_mae_differences.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for fraction, methods in by_budget.items():
        for method, values in methods.items():
            for index, value in enumerate(values, start=1):
                rows.append({"budget": int(float(fraction) * 100), "method": method,
                             "layout_repeat": index, "mae_difference_from_direct_s": value})
    data_path = PAPER / "figure_data" / "followup_mae_differences.csv"
    import csv
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (PAPER / "figure_data" / "followup_mae_differences.json").write_text(
        json.dumps({"source_hashes": source_hashes,
                    "protocol": "development labels only; no final/sealed test labels",
                    "points": rows}, indent=2) + "\n")


def make_local_cases(repo: Path, source_hashes):
    ours_dir = repo / "results/paper_followup_20261002_rho_tree_v3/rho_confirm/f0.1_m20260919/rho0.25/s42"
    direct_dir = repo / "results/paper_performance_20261002/confirm/cosine_001_960/f0.1_m20260919/direct_time/s42"
    ours_path, direct_path = ours_dir / "development_predictions.npz", direct_dir / "predictions.npz"
    with np.load(ours_path, allow_pickle=False) as saved:
        ours = {key: saved[key].copy() for key in ("predictions", "targets", "padding", "total")}
    with np.load(direct_path, allow_pickle=False) as saved:
        direct = {key: saved["preds" if key == "predictions" else key].copy()
                  for key in ("predictions", "targets", "padding", "total")}
    for key in ("targets", "padding", "total"):
        np.testing.assert_array_equal(ours[key], direct[key])
    valid = ~ours["padding"]
    absolute = np.abs(ours["predictions"] - ours["targets"])
    counts = valid.sum(axis=1)
    trip_mae = np.divide((absolute * valid).sum(axis=1), counts,
                         out=np.zeros(len(counts), dtype=np.float64), where=counts > 0)
    cases = []
    for label, quantile in (("median", .5), ("90th percentile", .9)):
        target = np.quantile(trip_mae, quantile)
        row = int(np.argmin(np.abs(trip_mae - target)))
        length = int(counts[row])
        truth = ours["targets"][row, :length]
        prediction = ours["predictions"][row, :length]
        baseline = direct["predictions"][row, :length]
        cases.append({"label": label, "row": row, "length": length,
            "total_seconds": float(ours["total"][row]),
            "ours_trip_mae": float(trip_mae[row]),
            "ours_mae": float(np.abs(prediction - truth).mean()),
            "direct_mae": float(np.abs(baseline - truth).mean()),
            "max_total_residual": float(abs(prediction.sum() - ours["total"][row]))})

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 6.4,
        "axes.labelsize": 6.2, "axes.titlesize": 6.5, "legend.fontsize": 6.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .65})
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.15), sharex="col",
        gridspec_kw={"height_ratios": [1.8, 1.]}, layout="constrained")
    rows = []
    for col, case in enumerate(cases):
        row, length = case["row"], case["length"]
        x = np.arange(1, length + 1)
        truth = ours["targets"][row, :length]
        prediction = ours["predictions"][row, :length]
        baseline = direct["predictions"][row, :length]
        for values, label, color, style in (
            (truth, "Observed", "#222222", "-"),
            (prediction, "Ours", "#0072B2", "--"),
            (baseline, "Direct prediction", "#D55E00", ":")):
            axes[0, col].plot(x, values, label=label, color=color,
                              linestyle=style, linewidth=1.)
        axes[0, col].set_title(("(a) " if col == 0 else "(b) ") + case["label"] + " trip error")
        axes[0, col].set_ylim(bottom=0)
        axes[1, col].axhline(0, color="0.5", linewidth=.6)
        axes[1, col].plot(x, prediction - truth, color="#0072B2", linestyle="--", linewidth=1.)
        axes[1, col].plot(x, baseline - truth, color="#D55E00", linestyle=":", linewidth=1.)
        axes[1, col].set_xlabel("Ordered inter-stop interval")
        for ax in axes[:, col]:
            ax.set_xlim(1, length)
            ax.grid(axis="y", color="0.88", linewidth=.5)
            ax.set_axisbelow(True)
        for index in range(length):
            rows.append({"case": case["label"], "development_row": row,
                "interval": index + 1, "observed_seconds": float(truth[index]),
                "ours_seconds": float(prediction[index]), "direct_seconds": float(baseline[index]),
                "ours_error_seconds": float(prediction[index] - truth[index]),
                "direct_error_seconds": float(baseline[index] - truth[index])})
    axes[0, 0].set_ylabel("Running time (s)")
    axes[1, 0].set_ylabel("Signed error (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               handlelength=1.2, columnspacing=.8, handletextpad=.25)
    fig.get_layout_engine().set(rect=(0., 0., 1., .91))
    fig.savefig(PAPER / "figures/local_cases.pdf")
    fig.savefig(PAPER / "figures/local_cases.png", dpi=600)
    plt.close(fig)
    import csv
    with (PAPER / "figure_data/local_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (PAPER / "figure_data/local_cases_summary.json").write_text(json.dumps({
        "source_hashes": {"ours_predictions": sha(ours_path), "direct_predictions": sha(direct_path),
                          **source_hashes},
        "selection": "nearest development-trip MAE to the 50th and 90th percentiles",
        "cases": cases,
    }, indent=2) + "\n")
    by_label = {case["label"]: case for case in cases}
    replacements = {
        "CASE1_LENGTH": str(by_label["median"]["length"]),
        "CASE2_LENGTH": str(by_label["90th percentile"]["length"]),
        "CASE1_OURS": f"{by_label['median']['ours_mae']:.3f}",
        "CASE1_DIRECT": f"{by_label['median']['direct_mae']:.3f}",
        "CASE2_OURS": f"{by_label['90th percentile']['ours_mae']:.3f}",
        "CASE2_DIRECT": f"{by_label['90th percentile']['direct_mae']:.3f}",
    }
    for relative in ("sections/en/experiments.tex", "sections/zh/experiments.tex"):
        path = PAPER / relative
        text = path.read_text()
        for token, value in replacements.items():
            text = text.replace(token, value)
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    records, source_hashes = load_evidence(args.repo_root.resolve())
    write_tables(records)
    make_plot(records, source_hashes)
    make_local_cases(args.repo_root.resolve(), source_hashes)
    summaries = {}
    for fraction, methods in records.items():
        summaries[fraction] = {}
        for method, rows in methods.items():
            summaries[fraction][method] = {metric: {"mean": mean_sd(rows, metric)[0],
                "sd": mean_sd(rows, metric)[1]} for metric in METRICS}
    (PAPER / "evidence" / "paper_followup_20261002.json").write_text(
        json.dumps({"protocol": "Astana development set; validation-selected configuration; test sealed",
                    "source_hashes": source_hashes, "summary": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()

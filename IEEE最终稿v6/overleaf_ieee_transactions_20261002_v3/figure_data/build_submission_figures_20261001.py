#!/usr/bin/env python3
"""Build manuscript tables and a reproducible, publication-size analysis figure.

In the Overleaf package, run the copied program with --from-data to redraw the
figure without training data, checkpoints, PyTorch, or repository imports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent / 'paper小论文/overleaf_ieee_transactions_20260930_final'
MASKS = (20260919, 20260920, 20260921)
SEEDS = (42, 43, 44)
METRICS = ('mae', 'rmse', 'wape', 'prefix50', 'top10_mae')
NAMES = {'ours': ('Ours', 'Ours'),
         'direct_time': ('Direct prediction', '直接预测'),
         'identity_mean': ('Identity means', '同段均值'),
         'nnls_ridge_tuned': ('NNLS-Ridge', 'NNLS-Ridge'),
         'uniform': ('Uniform', '均匀分配')}
STYLES = {'identity_mean': ('#009E73', '^', ':'),
          'nnls_ridge_tuned': ('#E69F00', 'v', '-.'),
          'direct_time': ('#0072B2', 's', '--'),
          'ours': ('#D55E00', 'o', '-'),
          'uniform': ('#777777', 'D', '-')}
HATCHES = {'uniform': '/', 'identity_mean': '\\', 'nnls_ridge_tuned': 'x',
           'direct_time': '.', 'ours': '+'}
VARIANTS = ('rho010', 'rho025', 'centered045_reference', 'uncentered045')
SINGLE_COLUMN = 3.5
COMPARISON_SIZE = (SINGLE_COLUMN, 4.4)
ABLATION_SIZE = (SINGLE_COLUMN, 3.9)
ABLATION_HATCHES = ('//', r'\\', 'xx', '..')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n')


def verified_gate(root, stage):
    folder = root / 'results/submission_revision_20261001'
    gate = load(folder / f'{stage}_gate.json')
    audit_path = folder / f'{stage}_audit.json'
    if gate['status'] != 'closed' or gate['audit_sha256'] != sha(audit_path):
        raise ValueError(f'{stage} audit is not closed or has changed')
    for relative, digest in gate['result_hashes'].items():
        if sha(root / relative) != digest:
            raise ValueError(f'{stage} result changed: {relative}')
        result = load(root / relative)
        for name, artifact_digest in result.get('artifact_hashes', {}).items():
            if sha((root / relative).parent / name) != artifact_digest:
                raise ValueError(f'{stage} artifact changed: {relative} / {name}')
    report = load(audit_path)
    if report['fixed_or_sealed_test_truth_read'] or report['evaluation_labels_used_for_selection']:
        raise ValueError('experiment information boundary violated')
    return report


def verified_extension_gate(root):
    folder = root / 'results/submission_ablation_extension_20261002'
    gate = load(folder / 'gate.json')
    audit_path = folder / 'audit.json'
    if gate['status'] != 'closed' or gate['stage'] != 'multi_layout_ablation_extension':
        raise ValueError('multi-layout extension audit is not closed')
    if gate['audit_sha256'] != sha(audit_path):
        raise ValueError('multi-layout extension audit has changed')
    report = load(audit_path)
    if (report['audited_new_runs'] != 36 or report['fixed_or_sealed_test_truth_read']
            or report['evaluation_labels_used_for_selection']):
        raise ValueError('multi-layout extension failed its evidence boundary')
    for relative, digest in gate['result_hashes'].items():
        if sha(root / relative) != digest:
            raise ValueError(f'multi-layout extension result changed: {relative}')
    return report


def summarize_observations(rows, conditions, condition_key):
    summaries = {}
    for condition in conditions:
        values = [row for row in rows if row[condition_key] == condition]
        if len(values) != 9:
            raise ValueError(f'{condition} must have nine observations across three layouts')
        metrics = {}
        for metric in METRICS + ('tail_mae',):
            scores = np.asarray([row[metric] for row in values], dtype=float)
            metrics[metric] = {'mean': float(scores.mean()),
                'sd': float(scores.std(ddof=1)), 'n': len(scores)}
        summaries[condition] = {'runs': len(values), **metrics, 'per_run': values}
    return summaries


def collect(root, package, existing_only=False):
    sys.path.insert(0, str(root))
    from scripts import build_ieee_revision_evidence_20260930 as verified
    frozen = load(package / 'evidence/results.json')
    teacher = load(package / 'evidence/teacher_revision.json')
    budget = None if existing_only else verified_gate(root, 'budget')
    mechanism = None if existing_only else verified_gate(root, 'mechanism')
    extension = None if existing_only else verified_extension_gate(root)
    limited = {key: value for key, value in frozen['limited'].items()}
    if budget:
        limited['0.05'] = budget['summary']
    source_files = {str(package / 'evidence' / name): sha(package / 'evidence' / name)
                    for name in ('results.json', 'teacher_revision.json')}
    for relative, digest in teacher['source_files'].items():
        if sha(root / relative) != digest:
            raise ValueError(f'teacher source changed: {relative}')
        source_files[str(root / relative)] = digest
    observations = []
    reference = None
    for fraction in (.01, .1):
        for method in ('identity_mean', 'nnls_ridge_tuned', 'direct_time', 'ours'):
            for repeat, mask in enumerate(MASKS, 1):
                for init, seed in enumerate(SEEDS if method in ('ours', 'direct_time') else (None,), 1):
                    if method == 'identity_mean':
                        path = package / 'evidence' / f'identity_mean_f{fraction:g}_m{mask}.npz'
                        with np.load(path, allow_pickle=False) as archive:
                            arrays = {key: archive[key].copy() for key in ('preds', 'targets', 'padding', 'total')}
                        metric = verified.allocation_metrics(**{'pred': arrays['preds'], 'target': arrays['targets'],
                            'padding': arrays['padding'], 'true_total': arrays['total']})
                    else:
                        path = root / 'results/thesis_final_protocol_20260926' / f'f{fraction:g}_m{mask}' / method
                        if seed is not None:
                            path /= f's{seed}'
                        result, arrays, metric = verified.read_verified_prediction(path, reference)
                        if result['protocol']['test_labels_opened'] or result['protocol']['development_labels_used_for_selection']:
                            raise ValueError('frozen run information boundary violated')
                        path /= 'result.json'
                    reference = arrays if reference is None else reference
                    for key in ('targets', 'padding', 'total'):
                        np.testing.assert_array_equal(arrays[key], reference[key])
                    source_files[str(path)] = sha(path)
                    observations.append({'budget_percent': int(fraction * 100), 'method': method,
                        'mask_repeat': repeat, 'initialization_repeat': init, 'mask_seed': mask,
                        'seed': seed, 'source': str(path), **metric})
            values = [r['mae'] for r in observations if r['budget_percent'] == int(fraction * 100) and r['method'] == method]
            if abs(float(np.mean(values)) - limited[str(fraction)][method]['mae']['mean']) > 1e-9:
                raise ValueError('frozen summary differs from verified observations')
    if budget:
        for method in ('identity_mean', 'nnls_ridge_tuned', 'direct_time', 'ours'):
            for row in budget['summary'][method]['per_run']:
                observations.append({'budget_percent': 5, 'method': method,
                    'mask_repeat': MASKS.index(row['mask_seed']) + 1,
                    'initialization_repeat': SEEDS.index(row['seed']) + 1 if row['seed'] else 1,
                    'source': str(root / row['path'] / 'result.json'), **row})
    uniform_path = package / 'evidence/uniform_f0.01_m20260919.npz'
    with np.load(uniform_path, allow_pickle=False) as archive:
        uniform_arrays = {key: archive[key].copy() for key in ('preds', 'targets', 'padding', 'total')}
    for key in ('targets', 'padding', 'total'):
        np.testing.assert_array_equal(uniform_arrays[key], reference[key])
    uniform_metrics = verified.allocation_metrics(pred=uniform_arrays['preds'],
        target=uniform_arrays['targets'], padding=uniform_arrays['padding'],
        true_total=uniform_arrays['total'])
    source_files[str(uniform_path)] = sha(uniform_path)
    for percent, summary in ((1, limited['0.01']), (10, limited['0.1'])):
        for metric in (*METRICS, 'tail_mae'):
            value = uniform_metrics[metric]
            if abs(value - summary['uniform'][metric]['mean']) > 1e-9:
                raise ValueError('uniform prediction differs from frozen summary')
    if budget:
        for metric in (*METRICS, 'tail_mae'):
            value = uniform_metrics[metric]
            if abs(value - budget['summary']['uniform'][metric]['mean']) > 1e-9:
                raise ValueError('uniform prediction differs from 5% budget summary')
    for percent in (1, 5, 10):
        observations.append({'budget_percent': percent, 'method': 'uniform',
            'mask_repeat': 1, 'initialization_repeat': 1, 'mask_seed': MASKS[0],
            'seed': None, 'source': str(uniform_path), **uniform_metrics})
    paired = []
    for percent in (1, 5, 10):
        if percent == 5 and not budget:
            continue
        keyed = {(r['method'], r['mask_repeat'], r['initialization_repeat']): r
                 for r in observations if r['budget_percent'] == percent}
        for mask in range(1, 4):
            for init in range(1, 4):
                ours, direct = (keyed[(m, mask, init)] for m in ('ours', 'direct_time'))
                paired.append({'budget_percent': percent, 'mask_repeat': mask,
                    'initialization_repeat': init, 'ours_mae_seconds': ours['mae'],
                    'direct_mae_seconds': direct['mae'], 'difference_seconds': ours['mae'] - direct['mae'],
                    'ours_source': ours['source'], 'direct_source': direct['source']})
    layout_path = root / 'paper/tables/thesis_followup_20260922/block_missing_summary.csv'
    dynamics_path = root / 'paper/tables/thesis_evidence_20260920/dynamic_summary.csv'
    for path in (layout_path, dynamics_path):
        relative = str(path.relative_to(root))
        if sha(path) != frozen['source_files'][relative]:
            raise ValueError(f'frozen CSV drift: {relative}')
        source_files[str(path)] = sha(path)
    layout = list(csv.DictReader(layout_path.open()))
    dynamics = list(csv.DictReader(dynamics_path.open()))
    identity_counts = []
    for category in ('random', 'prefix_block', 'moving_blackout'):
        rows = [r for r in layout if r['setting'] == category]
        counts = {int(r['fit_visible_identities']) for r in rows}
        if len(counts) != 1:
            raise ValueError('layout counts disagree across methods')
        identity_counts.append({'layout': category, 'visible_identities': counts.pop(),
                                'training_labels': int(rows[0]['visible_labels']), 'source': str(layout_path)})
    coverage = []
    ablation = []
    for kind, target in (('coverage', coverage), ('ablation', ablation)):
        for group, value in teacher[kind].items():
            methods = value if kind == 'coverage' else {group: value}
            for method, summary in methods.items():
                for row in summary['per_seed']:
                    target.append({'group': group, 'method': method,
                        'initialization_repeat': SEEDS.index(row['seed']) + 1, **row})
    correction = []
    extension_summary = None
    if extension:
        ablation = [dict(row, panel='calendar_correction',
            initialization_repeat=SEEDS.index(row['seed']) + 1,
            mask_repeat=MASKS.index(row['mask_seed']) + 1)
            for row in extension['ablation_comparison_runs']]
        correction = [dict(row, panel='correction_parameterization',
            variant=row['method'], initialization_repeat=SEEDS.index(row['seed']) + 1,
            mask_repeat=MASKS.index(row['mask_seed']) + 1)
            for row in extension['parameterization_comparison_runs']]
        extension_summary = {'ablation': summarize_observations(ablation,
                ('ours', 'no_deviation', 'no_time', 'shuffle_time'), 'method'),
            'correction_parameterization': summarize_observations(correction,
                ('centered045_reference', 'rho010', 'rho025', 'uncentered045'), 'method')}
    elif mechanism:
        for variant in VARIANTS:
            for row in mechanism['summary'][variant]['per_run']:
                correction.append({'variant': variant, 'rho': {'rho010': .1, 'rho025': .25,
                    'centered045_reference': .45, 'uncentered045': .45}[variant],
                    'weighted_centered': variant != 'uncentered045',
                    'initialization_repeat': SEEDS.index(row['seed']) + 1, **row})
    for report in (budget, mechanism):
        if report:
            source = root / 'results/submission_revision_20261001' / f"{report['stage']}_audit.json"
            source_files[str(source)] = sha(source)
            for row in report['audits']:
                path = root / row['path'] / 'result.json'
                source_files[str(path)] = sha(path)
    if extension:
        for relative in ('results/submission_ablation_extension_20261002/audit.json',
                         'results/submission_ablation_extension_20261002/gate.json',
                         'results/submission_ablation_extension_20261002/run_manifest.json'):
            source_files[str(root / relative)] = sha(root / relative)
        for row in extension['new_runs']:
            path = root / row['path'] / 'result.json'
            source_files[str(path)] = sha(path)
    source_files[str(Path(__file__).resolve())] = sha(__file__)
    return {'schema': 'submission_figure_data_v1', 'units': {'errors': 'seconds', 'wape': 'fraction'},
        'protocol': {'fixed_or_sealed_test_truth_read': False, 'primary_rho': .45,
            'repeats': 'three label masks by three initializations for neural budget runs; three initializations for controls',
            'paired_panel': 'individual paired run differences; no significance test or confidence interval',
            'paired_table': 'existing conditional trip bootstrap; does not include retraining uncertainty'},
        'panel_protocols': {'a_b': '1/5/10% training and validation labels; three masks by three initializations for neural methods',
            'c': 'full-validation layout control; equal 23720-label training budgets',
            'd': 'identity-count-matched layouts; limited validation labels; three paired initializations',
            'e_f': 'one fixed 10% training/validation mask; three paired initializations; primary rho=.45 retained'},
        'limited': limited, 'frozen': frozen, 'teacher': teacher, 'budget': budget,
        'mechanism': mechanism, 'extension': extension, 'extension_summary': extension_summary,
        'observations': observations, 'paired_observations': paired, 'identity_counts': identity_counts,
        'coverage_observations': coverage, 'correction_observations': correction,
        'ablation_observations': ablation, 'layout': layout, 'dynamics': dynamics,
        'source_files': source_files}


def cell(value, digits=3, bold=False, sd=None):
    text = f'{value:.{digits}f}'
    if sd is not None:
        text += rf'\pm{sd:.{digits}f}'
    return '$' + (rf'\mathbf{{{text}}}' if bold else text) + '$'


def write_tables(data, package):
    audit = []
    for lang in ('en', 'zh'):
        ix = lang == 'zh'
        folder = package / 'tables' / lang
        folder.mkdir(parents=True, exist_ok=True)
        legend = (' Bold marks the lowest error within each comparable group.' if not ix
                  else ' 粗体表示各可比组内的最低误差。')

        def table(name, caption, spec, header, rows, wide=False, ranked=False, colsep=4):
            environment = 'table*' if wide else 'table'
            if ranked:
                caption += legend
            text = (f'\\begin{{{environment}}}[t]\n\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{{colsep}pt}}\n'
                    f'\\caption{{{caption}}}\\label{{tab:{name}}}\n\\begin{{tabular}}{{{spec}}}\\toprule\n'
                    + header + '\\\\\\midrule\n' + '\n'.join(rows)
                    + f'\n\\bottomrule\\end{{tabular}}\n\\end{{{environment}}}\n')
            (folder / f'{name}.tex').write_text(text)

        def ranked_rows(name, records, labels, keys, digits=None, group='all', sd_keys=('mae', 'rmse')):
            minima = {key: min(record[key]['mean'] for record in records.values()) for key in keys}
            audit.append({'table': name, 'language': lang, 'group': group, 'columns': minima,
                'rows': {label: {key: record[key]['mean'] for key in keys} for label, record in records.items()}})
            return [labels[method] + ' & ' + ' & '.join(
                cell(record[key]['mean'], (digits or {}).get(key, 3), record[key]['mean'] == minima[key],
                     record[key]['sd'] if key in sd_keys else None) for key in keys) + r'\\'
                for method, record in records.items()]

        rows = []
        for percent in (1, 5, 10):
            key = str(percent / 100)
            if key not in data['limited']:
                continue
            records = {method: data['limited'][key][method] for method in ('uniform', 'identity_mean', 'nnls_ridge_tuned', 'direct_time', 'ours')}
            transformed = {m: {k: dict(v[k]) for k in METRICS} for m, v in records.items()}
            for value in transformed.values():
                value['wape']['mean'] *= 100
            labels = {method: rf'{percent}\% & {NAMES[method][ix]}' for method in records}
            if rows:
                rows.append(r'\midrule')
            rows += ranked_rows('limited', transformed, labels, METRICS,
                {'wape': 2, 'prefix50': 2, 'top10_mae': 2}, group=f'{percent}%')
        table('limited',
            (r'Astana evaluation errors: 2,182 trips and 87,715 intervals. Seconds except WAPE (\%). Neural rows: mean $\pm$ sample SD of nine runs; baselines: three label masks, with the uniform reference shared across budgets.' if not ix else
             r'Astana评估误差：2,182趟、87,715段。除WAPE（\%）外均为秒。神经方法报告九次运行的均值与样本标准差；基线采用三个标签掩码，均匀分配参照在预算间共用。'),
            'llrrrrr', (r'Budget & Method & MAE & RMSE & WAPE & Prefix50 & Top-10\% MAE' if not ix else
                         r'预算 & 方法 & MAE & RMSE & WAPE & Prefix50 & 高贡献10\% MAE'), rows, wide=True, ranked=True)
        rows = []
        for percent in (1, 5, 10):
            if percent == 5:
                if not data['budget']:
                    continue
                metric = data['budget']['paired']
            else:
                metric = data['frozen']['paired'][str(percent / 100)]['ours_minus_direct']
            rows.append(rf"{percent}\% & {metric['difference']:+.3f} & [{metric['ci95'][0]:+.3f}, {metric['ci95'][1]:+.3f}]\\")
        table('paired', (r'Ours minus direct-prediction MAE (s). Conditional 95\% trip-bootstrap intervals use 5,000 draws and exclude retraining uncertainty.' if not ix else
            r'Ours减直接预测的MAE差（秒）。条件95\%行程bootstrap区间采用5,000次重采样，不包含重新训练的不确定性。'),
            'lrr', r'Budget & Difference & 95\% interval' if not ix else r'预算 & 差值 & 95\%区间', rows)
        layout_names = {'random': ('Random', '随机'), 'prefix_block': (r'Prefix 20\%', r'前20\%'),
                        'moving_blackout': ('Moving blackout', '移动遮挡')}
        rows = []
        for layout in layout_names:
            group_rows = [row for row in data['layout'] if row['setting'] == layout]
            minima = {k: min(float(r[k]) for r in group_rows) for k in ('all_mae', 'later_mae')}
            audit.append({'table': 'layout', 'language': lang, 'group': layout, 'columns': minima})
            for r in group_rows:
                rows.append(f"{layout_names[layout][ix]} & {NAMES[r['method']][ix]} & {r['fit_visible_identities']} & "
                    + cell(float(r['all_mae']), 3, float(r['all_mae']) == minima['all_mae'], float(r['all_sd']))
                    + ' & ' + cell(float(r['later_mae']), 3, float(r['later_mae']) == minima['later_mae']) + r'\\')
        table('layout', (r'Label layouts with 23,720 training labels, full validation labels and three initializations. IDs: visible interval identities; later MAE: final approximately 80\% of each trip (s).' if not ix else
            r'标签布局控制：23,720个训练标签、完整验证标签、三次初始化。身份数为可见区间身份；后段MAE评价各行程约后80\%位置（秒）。'),
            'llrrr', 'Layout & Method & IDs & All MAE & Later MAE' if not ix else '布局 & 方法 & 身份数 & 总体MAE & 后段MAE', rows, wide=True, ranked=True)
        rows = []
        for layout in ('prefix', 'identity_matched_random'):
            records = data['teacher']['coverage'][layout]
            layout_name = {'prefix': (r'First 20\%', r'前20\%标签'), 'identity_matched_random': ('Identity-matched random', '身份计数匹配随机')}[layout][ix]
            labels = {m: layout_name + ' & ' + NAMES[m][ix] for m in records}
            rows += ranked_rows('controlled_coverage', records, labels, ('mae', 'prefix50'), {'prefix50': 2}, group=layout)
        table('controlled_coverage', ('Label-placement control with identical identity-specific label counts; three initializations and limited validation labels. Errors in seconds.' if not ix else
            '同标签总数、身份及每身份标签数量的布局控制：三次初始化、有限验证标签；误差单位为秒。'),
            'llrr', 'Layout & Method & MAE & Prefix50' if not ix else '布局 & 方法 & MAE & Prefix50', rows, wide=True, ranked=True)
        controls = {'ours': ('Ours', 'Ours'), 'no_deviation': ('No correction', '关闭修正'),
                    'no_time': ('No calendar', '去日历特征'), 'shuffle_time': ('Shuffled calendar', '打乱日历特征')}
        labels = {m: controls[m][ix] for m in controls}
        ablation_summary = (data['extension_summary']['ablation'] if data['extension']
                            else data['teacher']['ablation'])
        rows = ranked_rows('finite_ablation', ablation_summary, labels, ('mae', 'prefix50'), {'prefix50': 2})
        table('finite_ablation', (r'Ablations across three 10\% training/validation layouts and three paired initializations per layout. Errors in seconds.' if not ix and data['extension'] else
            r'三个10\%训练/验证标签布局、每个布局三次配对初始化的消融结果。误差单位为秒。' if ix and data['extension'] else
            r'Ablations with one 10\% training/validation mask and three initializations. Errors in seconds.' if not ix else
            r'统一有限训练与验证协议的消融：一组10\%掩码、三次初始化；误差单位为秒。'),
            'lrr', 'Variant & MAE & Prefix50' if not ix else '变体 & MAE & Prefix50', rows, ranked=True)
        dynamics = {r['method']: {'mae': {'mean': float(r['mae_mean']), 'sd': float(r['mae_sd'])},
                    'prefix50': {'mean': float(r['prefix50_mean'])}} for r in data['dynamics'] if r['method'] in controls}
        rows = ranked_rows('mechanism', dynamics, labels, ('mae', 'prefix50'), {'prefix50': 2})
        table('mechanism', (r'Calendar and correction controls with one 10\% training mask, full validation labels and three initializations. Errors in seconds.' if not ix else
            r'日历特征与修正消融：一个10\%训练掩码、完整验证标签、三次初始化；误差单位为秒。'),
            'lrr', 'Variant & MAE & Prefix50' if not ix else '变体 & MAE & Prefix50', rows, ranked=True)
        if data['mechanism']:
            redistribution = (data['extension_summary']['correction_parameterization']
                if data['extension'] else data['mechanism']['summary'])
            records = {v: redistribution[v] for v in VARIANTS}
            labels = {'rho010': (r'Weighted centering & 0.10', r'加权中心化 & 0.10'),
                      'rho025': (r'Weighted centering & 0.25', r'加权中心化 & 0.25'),
                      'centered045_reference': (r'Weighted centering (primary) & 0.45', r'加权中心化（主要设置） & 0.45'),
                      'uncentered045': (r'Without weighted centering & 0.45', r'无加权中心化 & 0.45')}
            rows = ranked_rows('redistribution', records, {v: labels[v][ix] for v in records}, ('mae', 'rmse', 'prefix50'), {'prefix50': 2})
            table('redistribution', (r'Correction parameterizations across three 10\% training/validation layouts and three paired initializations per layout. Errors in seconds; mean $\pm$ sample SD. Uncentered scaling also changes effective amplitude; primary $\rho=0.45$ is retained.' if not ix and data['extension'] else
                r'三个10\%训练/验证标签布局、每个布局三次配对初始化下的修正参数化比较。误差单位为秒，报告均值与样本标准差。无加权中心化后的缩放也改变有效幅度；主要设置仍为$\rho=0.45$。' if ix and data['extension'] else
                r'Correction parameterizations: one 10\% training/validation mask and three initializations. Errors in seconds; mean $\pm$ sample SD. Uncentered scaling also changes effective amplitude; primary $\rho=0.45$ is retained.' if not ix else
                r'修正参数化比较：一组10\%训练与验证掩码、三次初始化。误差单位为秒，报告均值与样本标准差。无加权中心化后的缩放也改变有效幅度；主要设置仍为$\rho=0.45$。'),
                'llrrr', r'Variant & $\rho$ & MAE & RMSE & Prefix50' if not ix else r'变体 & $\rho$ & MAE & RMSE & Prefix50', rows, wide=True, ranked=True)
        prefix_names = {'legacy': ('Ours: padding-inclusive', 'Ours：分母包含填充'),
            'fixed': ('Ours: valid-only', 'Ours：分母仅计有效位置'),
            'no_prefix': ('Ours: no prefix loss', 'Ours：无前缀损失'),
            'fixed_direct': ('Direct prediction: valid-only', '直接预测：分母仅计有效位置')}
        prefix = data['teacher']['singapore']
        rows = ranked_rows('prefix_fixed', prefix, {m: prefix_names[m][ix] for m in prefix}, ('mae', 'prefix50', 'tail_mae'), {'prefix50': 2, 'tail_mae': 2})
        table('prefix_fixed', ('Singapore prefix-normalization check: full training/validation labels, three initializations and no historical inputs. Errors in seconds.' if not ix else
            'Singapore前缀归一化验证：完整训练与验证标签、三次初始化、无额外历史输入；误差单位为秒。'),
            'lrrr', 'Method & MAE & Prefix50 & Long MAE' if not ix else '方法 & MAE & Prefix50 & 长段MAE', rows, wide=True, ranked=True)
        rows = []
        for repeat, seed in enumerate(SEEDS, 1):
            values = [data['teacher']['coverage'][layout][method]['per_seed'][repeat - 1]['mae']
                for layout in ('prefix', 'identity_matched_random') for method in ('ours', 'direct_time')]
            rows.append(str(repeat) + ' & ' + ' & '.join(cell(v) for v in values) + r'\\')
        table('coverage_seeds', ('Individual initialization-repeat MAEs (s) for the identity-count control. P: prefix; R: matched random.' if not ix else
            '身份计数匹配控制的逐初始化重复MAE（秒）。P为前缀布局，R为匹配随机布局。'),
            'rrrrr', 'Repeat & P Ours & P direct & R Ours & R direct' if not ix else '重复 & P Ours & P直接 & R Ours & R直接', rows, wide=True)
        rows = [str(repeat) + ' & ' + ' & '.join(cell(prefix[v]['per_seed'][repeat - 1]['mae']) for v in prefix_names) + r'\\' for repeat in range(1, 4)]
        table('prefix_seeds', 'Individual initialization-repeat Singapore MAEs (s).' if not ix else 'Singapore前缀验证的逐初始化重复MAE（秒）。',
            'rrrrr', 'Repeat & Padding-inclusive & Valid-only & No prefix & Direct prediction' if not ix else '重复 & 含填充分母 & 有效分母 & 无前缀 & 直接预测', rows, wide=True)
        rows = [f"{repeat} & {r['best_step']} & {r['mae']:.3f} & {r['history'][-1]['selection_mae']:.3f}\\\\" for repeat, r in enumerate(data['teacher']['probeta'], 1)]
        table('probeta_extended', 'Fixed 3,840-update check of the ProbETA adaptation; errors in seconds.' if not ix else 'ProbETA改造模型的3,840步固定预算检查；误差单位为秒。',
            'rrrr', 'Repeat & Selected step & Evaluation MAE & Final validation MAE' if not ix else '重复 & 选中步数 & 评估MAE & 末步验证MAE', rows, wide=True)
        timing = data['teacher']['efficiency']['results']
        minimum = min(r['batch_256_median_ms'] for r in timing)
        rows = [f"{NAMES[r['method']][ix]} & " + cell(r['batch_256_median_ms'], 3, r['batch_256_median_ms'] == minimum)
                + f" & [{r['batch_256_iqr_ms'][0]:.3f}, {r['batch_256_iqr_ms'][1]:.3f}]\\\\" for r in timing]
        table('efficiency', ('Same-device 256-trip forward timing (ms), 30 repeats. Bold marks the lowest median.' if not ix else
            '同设备256趟批次前向计时（毫秒），30次重复。粗体表示最低中位数。'),
            'lrr', 'Method & Median & Interquartile range' if not ix else '方法 & 中位数 & 四分位范围', rows)
        convergence = data['frozen']['supplementary']['convergence']
        rows = [(NAMES[m][ix] if m in NAMES else ('Identity inputs', '仅身份输入')[ix]) + ' & '
                + cell(r['mae']['mean'], 3) + ' & '
                + '/'.join(map(str, r['best_steps'])) + r'\\' for m, r in convergence.items()]
        table('convergence', (r'1,920-update training check: one 10\% training mask and full validation labels. Learning rates differ by method; this table does not rank methods. Selected steps follow repeats 1/2/3; MAE in seconds.' if not ix else
            r'1,920步训练检查：一个10\%训练掩码、完整验证标签。各方法学习率不同，本表不用于方法排名。选中步数按重复1/2/3排列；MAE单位为秒。'),
            'lrr', 'Method & MAE & Selected steps' if not ix else '方法 & MAE & 选中步数', rows)
        singapore = data['frozen']['supplementary']['singapore']
        sg_labels = {'ours': NAMES['ours'][ix], 'direct_time': NAMES['direct_time'][ix],
                     'without_prefix_loss': ('No prefix loss', '去前缀损失')[ix]}
        records = {m: singapore[m] for m in sg_labels}
        rows = ranked_rows('singapore', records, sg_labels, ('mae', 'prefix50', 'tail_mae'), sd_keys=('mae', 'prefix50', 'tail_mae'))
        table('singapore', (r'Singapore full-label controls: 10,000 evaluation trips and 197,301 intervals. Seconds; mean $\pm$ sample SD over three initializations.' if not ix else
            'Singapore完整标签控制：10,000趟评估行程、197,301段。单位为秒；三次初始化的均值与样本标准差。'),
            'lrrr', 'Method & MAE & Prefix50 & Tail MAE' if not ix else '方法 & MAE & Prefix50 & 长段MAE', rows, ranked=True)
        r = singapore['no_conservation']
        rows = [('No total adjustment', '不作总量调整')[ix] + ' & ' + ' & '.join(cell(r[k]['mean'], 3, sd=r[k]['sd']) for k in ('mae', 'prefix50', 'tail_mae')) + r'\\']
        table('singapore_unconstrained', ('Unconstrained Singapore output, reported separately because it violates the known-total constraint. Seconds; three initializations.' if not ix else
            '单独报告未满足已知总时长约束的Singapore输出；单位为秒，三次初始化。'),
            'lrrr', 'Output & MAE & Prefix50 & Tail MAE' if not ix else '输出 & MAE & Prefix50 & 长段MAE', rows)
        records = {m: singapore[m] for m in ('history_full', 'without_same_segment_loss')}
        labels = {'history_full': ('History + loss', '历史输入与损失')[ix], 'without_same_segment_loss': ('History only', '仅历史输入')[ix]}
        rows = ranked_rows('history', records, labels, ('mae', 'prefix50', 'tail_mae'), sd_keys=('mae', 'prefix50', 'tail_mae'))
        table('history', (r'Separate Singapore history-input controls with the same strictly-past sources. Seconds; mean $\pm$ sample SD over three initializations.' if not ix else
            'Singapore历史输入控制，两种方法使用相同的严格过去信息源。单位为秒；三次初始化的均值与样本标准差。'),
            'lrrr', 'Method & MAE & Prefix50 & Tail MAE' if not ix else '方法 & MAE & Prefix50 & 长段MAE', rows, ranked=True, colsep=3)
        rows = []
        for bias in (-.1, -.05, -.01, 0., .01, .05, .1):
            proposed, direct = (data['frozen']['stress'][m][str(bias)] for m in ('ours', 'direct_time'))
            rows.append(f'{bias * 100:+.0f}\\% & ' + ' & '.join(f'{v:.3f}' if i < 2 else f'{v:.2f}' for i, v in enumerate(
                (proposed['mae']['mean'], direct['mae']['mean'], 100 * proposed['wape']['mean'], 100 * direct['wape']['mean']))) + r'\\')
        table('stress', (r'Total-bias sensitivity scenarios using frozen 10\% models: nine runs per method. MAE in seconds and WAPE in percent. Bias scenarios are not ranked.' if not ix else
            r'冻结10\%模型的总量偏差敏感性情景，每种方法九次运行。MAE单位为秒，WAPE单位为百分比；不同偏差情景不作排名。'),
            'lrrrr', 'Bias & Ours MAE & Direct MAE & Ours WAPE & Direct WAPE' if not ix else '偏差 & Ours MAE & 直接MAE & Ours WAPE & 直接WAPE', rows)
    return audit


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def export_data(data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / 'source_data.json', data)
    budget_rows = []
    for fraction, summaries in sorted(data['limited'].items(), key=lambda item: float(item[0])):
        for method in STYLES:
            r = summaries[method]
            budget_rows.append({'budget_percent': int(float(fraction) * 100), 'method': method,
                'mae_mean_seconds': r['mae']['mean'], 'mae_sample_sd_seconds': r['mae']['sd'], 'runs': r['runs']})
    write_csv(folder / 'panel_a_budget_summary.csv', budget_rows)
    for filename, key in (('panel_a_budget_observations.csv', 'observations'),
        ('panel_b_paired_differences.csv', 'paired_observations'), ('panel_c_visible_identities.csv', 'identity_counts'),
        ('panel_d_matched_layout.csv', 'coverage_observations'), ('panel_e_redistribution.csv', 'correction_observations'),
        ('panel_f_calendar_ablation.csv', 'ablation_observations')):
        rows = data[key]
        if rows:
            columns = sorted(set().union(*(r.keys() for r in rows)))
            write_csv(folder / filename, [{k: r.get(k) for k in columns} for r in rows])
    ablation_rows = []
    teacher = data['teacher']
    ablation_summary = (data['extension_summary']['ablation'] if data['extension']
                        else teacher['ablation'])
    for key, summary in ablation_summary.items():
        for row in summary.get('per_run', summary.get('per_seed', [])):
            ablation_rows.append({'panel': 'calendar_correction', 'condition': key,
                                  'mask_seed': row.get('mask_seed'), 'seed': row['seed'],
                                  'initialization_repeat': SEEDS.index(row['seed']) + 1,
                                  **{metric: row[metric] for metric in ('mae', 'rmse', 'prefix50')}})
    correction_summary = (data['extension_summary']['correction_parameterization'] if data['extension']
                          else data['mechanism']['summary'])
    for key, summary in correction_summary.items():
        for row in summary['per_run']:
            ablation_rows.append({'panel': 'correction_parameterization', 'condition': key,
                                  'mask_seed': row.get('mask_seed'), 'seed': row['seed'],
                                  'initialization_repeat': SEEDS.index(row['seed']) + 1,
                                  **{metric: row[metric] for metric in ('mae', 'rmse', 'prefix50')}})
    for key, summary in teacher['singapore'].items():
        for row in summary['per_seed']:
            ablation_rows.append({'panel': 'singapore_prefix', 'condition': key,
                                  'initialization_repeat': SEEDS.index(row['seed']) + 1,
                                  **{metric: row[metric] for metric in ('mae', 'rmse', 'prefix50', 'tail_mae')}})
    write_csv(folder / 'ablation_observations.csv', ablation_rows)
    dump(folder / 'data_profile.json', {'comparison_panels': ['MAE', 'RMSE', 'WAPE', 'Prefix50',
            'Top-10% MAE', 'long-interval MAE'],
        'ablation_panels': ['calendar/correction MAE', 'parameterization MAE/RMSE/Prefix50',
            'Singapore prefix-loss MAE and long-interval MAE'],
        'budget_observation_counts': {str(p): {m: sum(r['budget_percent'] == p and r['method'] == m
            for r in data['observations']) for m in STYLES} for p in (1, 5, 10)},
        'uniform_is_one_budget_invariant_evaluation': True,
        'ablation_observation_count': len(ablation_rows),
        'size_inches': {'comparison': list(COMPARISON_SIZE), 'ablation': list(ABLATION_SIZE)},
        'design': 'Two separate 3-by-2 grouped-bar figures; mean and sample SD bars overlay all repeated outcomes.'})
    program = folder / 'build_submission_figures_20261001.py'
    if Path(__file__).resolve() != program.resolve():
        shutil.copyfile(__file__, program)


def layout_audit(fig, minimum_font=6.0):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    issues = []
    for warning in caught:
        if 'Glyph' in str(warning.message) or 'missing from font' in str(warning.message):
            issues.append(str(warning.message))
    visible = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
            issues.append('Text outside canvas: ' + text.get_text())
        if text.get_fontsize() < minimum_font:
            issues.append(f'Text smaller than {minimum_font:g}pt: ' + text.get_text())
        visible.append((text.get_text(), [float(v) for v in box.extents]))
    for ax in fig.axes:
        for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            labels = [label for label in labels if label.get_visible() and label.get_text()]
            for left, right in zip(labels, labels[1:]):
                a, b = left.get_window_extent(renderer), right.get_window_extent(renderer)
                if min(a.x1, b.x1) > max(a.x0, b.x0) + 1 and min(a.y1, b.y1) > max(a.y0, b.y0) + 1:
                    issues.append('Tick labels overlap: ' + left.get_text() + ' / ' + right.get_text())
    return {'status': 'pass' if not issues else 'fail', 'issues': issues,
            'size_inches': list(fig.get_size_inches()), 'minimum_font_pt': minimum_font,
            'text_bounds_pixels': visible}


def plot(data, figures, data_folder):
    teacher = data['teacher']
    extension = data.get('extension') is not None
    condition_runs = 9 if extension else 3
    condition_observations = 36 if extension else 12
    expected = [(method, budget, 9 if method in ('ours', 'direct_time') else
                 1 if method == 'uniform' else 3)
                for method in STYLES for budget in (1, 5, 10)]
    for method, budget, count in expected:
        rows = [r for r in data['observations'] if r['method'] == method and r['budget_percent'] == budget]
        if len(rows) != count or any(not np.isfinite(r['mae']) for r in rows):
            raise ValueError(f'incomplete budget observations: {method}, {budget}%')
    for variant in VARIANTS:
        if sum(r['variant'] == variant for r in data['correction_observations']) != condition_runs:
            raise ValueError('incomplete correction observations: ' + variant)
    if (len(data['paired_observations']) != 27 or len(data['coverage_observations']) != 12
            or len(data['ablation_observations']) != condition_observations
            or len(data['correction_observations']) != condition_observations):
        raise ValueError('incomplete paired, coverage, or ablation observations')
    plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 7, 'axes.labelsize': 6.6,
        'axes.titlesize': 7, 'legend.fontsize': 6.2, 'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'submission-analysis-20261001',
        'mathtext.fontset': 'stix', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'lines.linewidth': 1.,
        'hatch.linewidth': .9, 'savefig.dpi': 600})
    plt.rcParams.update({'font.family': 'Liberation Serif', 'font.size': 7,
        'axes.labelsize': 6.7, 'axes.titlesize': 7.2, 'legend.fontsize': 6.5,
        'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'submission-bars-20261001',
        'mathtext.fontset': 'stix', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'hatch.linewidth': .9, 'savefig.dpi': 600})
    figures.mkdir(parents=True, exist_ok=True)
    methods = ('uniform', 'identity_mean', 'nnls_ridge_tuned', 'direct_time', 'ours')
    short_names = {'uniform': 'Uniform', 'identity_mean': 'ID mean',
                   'nnls_ridge_tuned': 'NNLS-Ridge', 'direct_time': 'Direct', 'ours': 'Ours'}
    metric_specs = (
        ('mae', 'MAE (s)', 1.), ('rmse', 'RMSE (s)', 1.),
        ('wape', 'WAPE (%)', 100.), ('prefix50', 'Prefix50 (s)', 1.),
        ('top10_mae', 'Top-10% MAE (s)', 1.), ('tail_mae', 'Long-interval MAE (s)', 1.),
    )
    colors = {method: STYLES[method][0] for method in STYLES}

    def observed_values(percent, method, metric):
        return [float(row[metric]) for row in data['observations']
                if row['budget_percent'] == percent and row['method'] == method]

    comparison, axes = plt.subplots(3, 2, figsize=COMPARISON_SIZE)
    comparison.subplots_adjust(left=.145, right=.985, top=.91, bottom=.075,
                               wspace=.39, hspace=.56)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], edgecolor='.16',
                                    linewidth=.7,
                                    label=short_names[m]) for m in methods]
    comparison.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.53, .995),
                      ncol=5, frameon=False, handlelength=.85, handleheight=.55,
                      columnspacing=.8, handletextpad=.25, borderaxespad=0.)
    method_width, cluster = .125, .76
    for panel, (ax, (metric, ylabel, scale)) in enumerate(zip(axes.flat, metric_specs)):
        ax.set_title(f'({chr(97 + panel)}) {ylabel.split(" (")[0]}', loc='left', pad=3)
        ax.grid(axis='y', color='.88', linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        for budget_index, percent in enumerate((1, 5, 10)):
            center = budget_index
            start = center - cluster / 2 + method_width / 2
            for method_index, method in enumerate(methods):
                values = observed_values(percent, method, metric)
                values = np.asarray(values, dtype=float) * scale
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError(f'missing {metric} observations for {method} at {percent}%')
                x = start + method_index * (cluster / len(methods))
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
                ax.bar(x, mean, width=method_width * .86, color=colors[method],
                       edgecolor='.16', linewidth=.6, zorder=2)
                if len(values) > 1:
                    ax.errorbar(x, mean, yerr=sd, fmt='none', ecolor='.12',
                                elinewidth=.55, capsize=1.2, capthick=.55, zorder=4)
                jitter = np.linspace(-method_width * .27, method_width * .27, len(values))
                ax.scatter(x + jitter, values, s=5, facecolors='white', edgecolors='.1',
                           linewidths=.45, zorder=5)
        ax.set_xticks([0, 1, 2], ['1%', '5%', '10%'])
        ax.set_xlabel('Visible labels', labelpad=1)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.set_xlim(-.52, 2.52)
        upper = max(float(np.max([r[metric] * scale for r in data['observations']])), 1.)
        ax.set_ylim(0, upper * 1.15)
        ax.tick_params(width=.5, length=2, pad=1)
    comp_report = layout_audit(comparison, minimum_font=6.)
    comp_report['figure_role'] = 'method comparison across label budgets'
    comp_report['method_counts_per_budget'] = {str(p): {m: len(observed_values(p, m, 'mae')) for m in methods}
                                                  for p in (1, 5, 10)}
    if comp_report['issues']:
        raise ValueError('comparison figure layout audit failed: ' + '; '.join(comp_report['issues']))
    save_figure(comparison, figures, data_folder, 'comparison_panels', comp_report)

    # Each ablation panel is a separate, internally comparable experiment.
    ablation, axes = plt.subplots(3, 2, figsize=ABLATION_SIZE)
    ablation.subplots_adjust(left=.16, right=.985, top=.93, bottom=.13,
                             wspace=.38, hspace=.47)
    palette = ('#D55E00', '#0072B2', '#009E73', '#E69F00')
    ablation_hatches = ABLATION_HATCHES
    extension_ablation = (data['extension_summary']['ablation'] if extension
                          else teacher['ablation'])
    extension_correction = (data['extension_summary']['correction_parameterization'] if extension
                            else data['mechanism']['summary'])
    ablation_specs = [
        ('(a) Calendar MAE', ['Ours', 'No corr.', 'No cal.', 'Shuffled'],
         [extension_ablation[k] for k in ('ours', 'no_deviation', 'no_time', 'shuffle_time')],
         'mae', 'MAE (s)', 1.),
        ('(b) Correction MAE', ['.10', '.25', '.45C', '.45U'],
         [extension_correction[k] for k in VARIANTS], 'mae', 'MAE (s)', 1.),
        ('(c) Correction RMSE', ['.10', '.25', '.45C', '.45U'],
         [extension_correction[k] for k in VARIANTS], 'rmse', 'RMSE (s)', 1.),
        ('(d) Correction Prefix50', ['.10', '.25', '.45C', '.45U'],
         [extension_correction[k] for k in VARIANTS], 'prefix50', 'Prefix50 (s)', 1.),
        ('(e) Singapore MAE', ['Pad.', 'Valid', 'No prefix', 'Direct'],
         [teacher['singapore'][k] for k in ('legacy', 'fixed', 'no_prefix', 'fixed_direct')],
         'mae', 'MAE (s)', 1.),
        ('(f) Singapore long MAE', ['Pad.', 'Valid', 'No prefix', 'Direct'],
         [teacher['singapore'][k] for k in ('legacy', 'fixed', 'no_prefix', 'fixed_direct')],
         'tail_mae', 'MAE (s)', 1.),
    ]
    for panel_index, (ax, (title, labels, summaries, metric, ylabel, scale)) in enumerate(
            zip(axes.flat, ablation_specs)):
        expected_runs = condition_runs if panel_index < 4 else 3
        ax.set_title(title, loc='left', pad=3)
        ax.grid(axis='y', color='.88', linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        reference_index = 0 if panel_index < 1 else 2 if panel_index < 4 else 1
        reference_rows = summaries[reference_index].get('per_run',
                         summaries[reference_index].get('per_seed', []))
        def pair_key(row):
            return (row.get('mask_seed', 0), row['seed'])
        reference_values = {pair_key(row): float(row[metric]) for row in reference_rows}
        all_differences = []
        markers = ('o', 's', '^', 'D')
        for index, summary in enumerate(summaries):
            per_run = summary.get('per_run', summary.get('per_seed', []))
            values = np.asarray([(float(row[metric]) - reference_values[pair_key(row)]) * scale
                                 for row in per_run])
            if len(values) != expected_runs:
                raise ValueError(f'expected {expected_runs} paired runs for {title}: {labels[index]}')
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.
            jitter = np.linspace(-.18, .18, len(values))
            ax.scatter(index + jitter, values, s=9, marker=markers[index],
                       facecolors='white', edgecolors=palette[index], linewidths=.65, zorder=4)
            ax.errorbar(index, mean, yerr=sd, fmt='D', markersize=3.4,
                        color='.1', markerfacecolor='.1', elinewidth=.65,
                        capsize=2, capthick=.65, zorder=5)
            all_differences.extend(values.tolist())
        ax.axhline(0, color='.35', linewidth=.65, linestyle=':', zorder=1)
        ax.set_xticks(range(4), labels)
        ax.set_ylabel('Difference (s)', labelpad=1)
        low, high = min(all_differences + [0.]), max(all_differences + [0.])
        span = max(high - low, .03)
        ax.set_ylim(low - span * .28, high + span * .28)
        ax.set_xlim(-.55, 3.55)
        ax.tick_params(axis='x', labelsize=6.0, width=.5, length=2, pad=1)
        ax.tick_params(axis='y', labelsize=6.1, width=.5, length=2, pad=1)
    ablation_report = layout_audit(ablation, minimum_font=6.0)
    ablation_report['figure_role'] = 'paired error differences from preregistered references'
    ablation_report['summary'] = 'mean and sample SD of paired differences; all paired observations shown'
    ablation_report['references'] = {'a': 'Ours', 'b-d': 'rho=.45 centered', 'e-f': 'valid prefix'}
    ablation_report['condition_counts'] = {'calendar_correction': 4, 'redistribution_parameterization': 4,
        'singapore_prefix_objective': 4}
    ablation_report['observations_per_condition'] = {'astana': condition_runs, 'singapore': 3}
    if ablation_report['issues']:
        raise ValueError('ablation figure layout audit failed: ' + '; '.join(ablation_report['issues']))
    save_figure(ablation, figures, data_folder, 'ablation_panels', ablation_report)
    dump(data_folder / 'visual_check.json', {'status': 'pass', 'issues': [],
        'figure_checks': [comp_report, ablation_report], 'single_column_width_inches': SINGLE_COLUMN,
        'layout': '3 rows by 2 columns in each of two separate figures'})
    return {'status': 'pass', 'comparison': comp_report, 'ablation': ablation_report}


def save_figure(fig, figures, data_folder, name, report):
    dump(data_folder / f'{name}_visual_check.json', report)
    base = figures / name
    fig.savefig(base.with_suffix('.png'), dpi=600)
    with Image.open(base.with_suffix('.png')) as raster:
        raster.convert('L').save(figures / f'{name}_grayscale.png', dpi=(600, 600))
    fig.savefig(base.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(base.with_suffix('.svg'), metadata={'Date': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--from-data', type=Path, help='Redraw from a packaged source_data.json; no repository access.')
    parser.add_argument('--tables-only', action='store_true')
    parser.add_argument('--existing-only', action='store_true', help='Format existing verified tables while new gates are pending.')
    args = parser.parse_args()
    if args.from_data:
        data = load(args.from_data)
        folder = args.from_data.resolve().parent
        report = plot(data, folder.parent / 'figures', folder)
    else:
        data = collect(args.repo_root.resolve(), args.package.resolve(), args.existing_only)
        folder = args.package / 'figure_data'
        export_data(data, folder)
        dump(folder / 'table_highlighting.json', write_tables(data, args.package))
        if not args.existing_only:
            source_files = {}
            for source, digest in data['source_files'].items():
                path = Path(source)
                key = str(path.relative_to(args.repo_root.resolve())) if path.is_relative_to(args.repo_root.resolve()) else str(path)
                source_files[key] = digest
            dump(args.package / 'evidence/submission_revision.json', {
                'schema': 'submission_revision_evidence_v1', 'primary_rho': .45,
                'fixed_or_sealed_test_truth_read': False, 'evaluation_labels_used_for_selection': False,
                'units': data['units'], 'limited': data['limited'], 'budget': data['budget'],
                'mechanism': data['mechanism'], 'source_files': source_files,
                'figure_source_data': 'figure_data/source_data.json',
                'figure_source_data_sha256': sha(folder / 'source_data.json')})
        if args.existing_only and not args.tables_only:
            raise ValueError('--existing-only requires --tables-only; incomplete figures cannot be released')
        report = None if args.tables_only else plot(data, args.package / 'figures', folder)
    print(json.dumps({'figure_data': str(folder), 'layout': report['status'] if report else 'not_rendered'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

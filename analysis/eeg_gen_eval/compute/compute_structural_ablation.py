"""Aggregate generator structural (component) ablation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import stats

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.eeg_gen_eval.compute.evaluate import predict_subject
from analysis.eeg_gen_eval.compute.metrics import (
    bandpower_means,
    compute_all_metrics,
    mean_per_channel_pearson,
)
from analysis.eeg_gen_eval.compute.structural_ablation_paths import (
    FREQUENCY_METRIC_KEYS,
    METRIC_COLUMNS,
    METRIC_KEYS,
    SEMANTIC_METRIC_KEYS,
    STRUCTURAL_CONDITIONS,
    STRUCTURAL_ORDER,
    STRUCTURAL_RAW_DIR,
    WAVEFORM_METRIC_KEYS,
    structural_ckpt_path,
    structural_eval_raw_dir,
    structural_metrics_path,
)

_LOWER_IS_BETTER = frozenset({'test_mse', 'fft_l1'})


def _load_subject_metrics(condition: str, sub: int) -> Optional[Dict]:
    path = structural_metrics_path(condition, sub)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_structural_frequency_metrics(
    condition: str,
    sub: int,
    device: Optional[torch.device] = None,
    force: bool = False,
) -> Optional[Dict[str, float]]:
    """Waveform metrics from saved generator checkpoints (per-ch mean r, frequency)."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    raw_dir = structural_eval_raw_dir(condition, sub)
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, 'frequency_metrics.json')
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            return json.load(f)

    ckpt = structural_ckpt_path(condition, sub)
    if not os.path.isfile(ckpt):
        return None

    y_pred, y_true = predict_subject(sub, device, ckpt)
    summary = compute_all_metrics(y_pred, y_true)['summary']
    bp_corr_mean, bp_err_mean = bandpower_means(
        summary['bandpower_correlation'],
        summary['bandpower_abs_error'],
    )
    result = {
        'subject': f'sub-{sub:02d}',
        'condition': condition,
        'test_corr': mean_per_channel_pearson(y_pred, y_true),
        'bandpower_corr': bp_corr_mean,
        'bandpower_abs_error': bp_err_mean,
        'fft_l1': float(summary['fft_l1_magnitude_error']),
        'fft_mse': float(summary['fft_mse_magnitude_error']),
        'checkpoint': ckpt,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    return result


def collect_rows(
    subs: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    force_freq: bool = False,
) -> List[Dict]:
    subs = subs or list(range(1, 11))
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rows = []
    for cond in STRUCTURAL_ORDER:
        meta = STRUCTURAL_CONDITIONS[cond]
        for sub in subs:
            m = _load_subject_metrics(cond, sub)
            if m is None:
                continue
            row = {
                'condition': cond,
                'label': meta['label'],
                'generator_arch': meta['generator_arch'],
                'per_channel_heads': meta['per_channel_heads'],
                'subject': f'sub-{sub:02d}',
            }
            for k in SEMANTIC_METRIC_KEYS:
                row[k] = m.get(k)
            row['test_mse'] = m.get('test_mse')
            freq = compute_structural_frequency_metrics(cond, sub, device, force=force_freq)
            if freq:
                row['test_corr'] = freq.get('test_corr')
                for k in FREQUENCY_METRIC_KEYS:
                    row[k] = freq.get(k)
            else:
                row['test_corr'] = m.get('test_corr')
                for k in FREQUENCY_METRIC_KEYS:
                    row[k] = None
            for k in ('test_retrieval_top1', 'test_retrieval_top5'):
                row[k] = m.get(k)
            rows.append(row)
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    out = []
    for cond in STRUCTURAL_ORDER:
        meta = STRUCTURAL_CONDITIONS[cond]
        sub_rows = [r for r in rows if r['condition'] == cond]
        if not sub_rows:
            continue
        entry = {
            'condition': cond,
            'label': meta['label'],
            'generator_arch': meta['generator_arch'],
            'per_channel_heads': meta['per_channel_heads'],
            'n_subjects': len(sub_rows),
        }
        for k in METRIC_KEYS:
            vals = [float(r[k]) for r in sub_rows if r.get(k) is not None and r[k] == r[k]]
            if not vals:
                entry[f'{k}_mean'] = None
                entry[f'{k}_std'] = None
                entry[f'{k}_sem'] = None
                continue
            entry[f'{k}_mean'] = float(np.mean(vals))
            entry[f'{k}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            entry[f'{k}_sem'] = entry[f'{k}_std'] / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out.append(entry)
    return out


def paired_tests(rows: List[Dict]) -> List[Dict]:
    """Wilcoxon signed-rank: each ablation vs S0_full."""
    full_rows = {r['subject']: r for r in rows if r['condition'] == 'S0_full'}
    if not full_rows:
        return []

    n_comparisons = (len(STRUCTURAL_ORDER) - 1) * len(METRIC_KEYS)
    alpha = 0.05 / max(n_comparisons, 1)
    results = []

    for cond in STRUCTURAL_ORDER[1:]:
        ab_rows = {r['subject']: r for r in rows if r['condition'] == cond}
        common = sorted(set(full_rows) & set(ab_rows))
        if len(common) < 3:
            continue
        entry = {
            'condition': cond,
            'label': STRUCTURAL_CONDITIONS[cond]['label'],
            'n_pairs': len(common),
            'bonferroni_alpha': alpha,
        }
        for k in METRIC_KEYS:
            full_vals = np.array([float(full_rows[s][k]) for s in common])
            ab_vals = np.array([float(ab_rows[s][k]) for s in common])
            if k in _LOWER_IS_BETTER:
                _, p = stats.wilcoxon(ab_vals, full_vals, alternative='greater')
                direction = 'worse_if_higher'
            else:
                _, p = stats.wilcoxon(ab_vals, full_vals, alternative='less')
                direction = 'worse_if_lower'
            entry[f'{k}_p'] = float(p)
            entry[f'{k}_significant'] = bool(p < alpha)
            entry[f'{k}_direction'] = direction
            entry[f'{k}_delta_mean'] = float(np.mean(ab_vals - full_vals))
        results.append(entry)
    return results


def _fmt_float(mean: float, std: float, digits: int = 3) -> str:
    return f'{mean:.{digits}f}±{std:.{digits}f}'


def _sig_stars(p: float, alpha: float) -> str:
    if p != p or p >= alpha:
        return ''
    if p < alpha / 10:
        return '***'
    if p < alpha / 3:
        return '**'
    return '*'


def _tests_by_condition(tests: List[Dict]) -> Dict[str, Dict]:
    return {t['condition']: t for t in tests}


def build_unified_table_rows(
    summary: List[Dict],
    tests: List[Dict],
) -> List[Dict]:
    """One row per condition; ablation rows include p-values vs Full."""
    alpha = tests[0]['bonferroni_alpha'] if tests else 0.05
    test_map = _tests_by_condition(tests)
    rows = []
    for s in summary:
        cond = s['condition']
        row = {
            'condition': cond,
            'label': s['label'],
            'n_subjects': s['n_subjects'],
            'bonferroni_alpha': alpha,
        }
        t = test_map.get(cond)
        for key, header, digits, _lower in METRIC_COLUMNS:
            mean = s[f'{key}_mean']
            std = s[f'{key}_std']
            row[f'{key}_mean'] = mean
            row[f'{key}_std'] = std
            row[f'{key}_display'] = _fmt_float(mean, std, digits)
            if cond == 'S0_full' or t is None:
                row[f'{key}_p'] = None
                row[f'{key}_sig'] = ''
                row[f'{key}_cell'] = row[f'{key}_display']
            else:
                p = float(t[f'{key}_p'])
                sig = _sig_stars(p, alpha)
                row[f'{key}_p'] = p
                row[f'{key}_sig'] = sig
                row[f'{key}_cell'] = f"{row[f'{key}_display']}{sig}"
                row[f'{key}_delta'] = float(t[f'{key}_delta_mean'])
        rows.append(row)
    return rows


def write_unified_table_md(table_rows: List[Dict], path: str) -> None:
    alpha = table_rows[0]['bonferroni_alpha'] if table_rows else 0.05
    headers = ['Condition'] + [h for _, h, _, _ in METRIC_COLUMNS]
    lines = [
        '# Generator structural ablation — unified metrics table',
        '',
        f'Mean ± s.d. over 10 subjects. Pearson r: per-channel r (pooled over trials × time) '
        f'averaged across 63 channels. Ablation rows: Wilcoxon signed-rank vs Full; '
        f'Bonferroni α={alpha:.4g} across {len(METRIC_COLUMNS)} metrics × '
        f'{len(STRUCTURAL_ORDER) - 1} ablations. '
        f'Significance: * p<{alpha:.4g}, ** p<{alpha / 3:.4g}, *** p<{alpha / 10:.4g}.',
        '',
        '| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join(['---'] * len(headers)) + '|',
    ]
    for row in table_rows:
        cells = [row['label']] + [row[f'{key}_cell'] for key, _, _, _ in METRIC_COLUMNS]
        lines.append('| ' + ' | '.join(cells) + ' |')

    lines.extend([
        '',
        '## p-values vs Full (ablation conditions only)',
        '',
        '| Condition | ' + ' | '.join(h for _, h, _, _ in METRIC_COLUMNS) + ' |',
        '|' + '|'.join(['---'] * (len(METRIC_COLUMNS) + 1)) + '|',
    ])
    for row in table_rows:
        if row['condition'] == 'S0_full':
            continue
        p_cells = []
        for key, _, _, _ in METRIC_COLUMNS:
            p = row[f'{key}_p']
            sig = row[f'{key}_sig']
            p_cells.append(f'{p:.4g}{sig}' if p is not None else '—')
        lines.append(f"| {row['label']} | " + ' | '.join(p_cells) + ' |')
    lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def write_unified_table_csv(table_rows: List[Dict], path: str) -> None:
    fieldnames = ['condition', 'label', 'n_subjects', 'bonferroni_alpha']
    for key, header, _, _ in METRIC_COLUMNS:
        fieldnames.extend([
            f'{key}_mean', f'{key}_std', f'{key}_display', f'{key}_p', f'{key}_sig', f'{key}_cell',
        ])
        if any(r.get(f'{key}_delta') is not None for r in table_rows):
            fieldnames.append(f'{key}_delta')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(table_rows)


def write_unified_table_tex(table_rows: List[Dict], path: str) -> None:
    alpha = table_rows[0]['bonferroni_alpha'] if table_rows else 0.05
    col_spec = 'l' + 'c' * len(METRIC_COLUMNS)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        rf'\caption{{Generator structural ablation (THINGS-EEG, Final two-stage). '
        rf'Mean$\pm$s.d. over 10 subjects. '
        rf'Wilcoxon vs Full; Bonferroni $\alpha={alpha:.4g}$. '
        rf'$^{{*}}$/$^{{**}}$/$^{{***}}$: $p<\alpha$, $p<\alpha/3$, $p<\alpha/10$.}}',
        r'\label{tab:structural_ablation}',
        rf'\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        'Condition & ' + ' & '.join(h.replace('↑', r'$\uparrow$').replace('↓', r'$\downarrow$')
                                     for _, h, _, _ in METRIC_COLUMNS) + r' \\',
        r'\midrule',
    ]
    for row in table_rows:
        label = row['label'].replace('w/o ', r'w/o ').replace('Full (Ours)', r'\textbf{Full (Ours)}')
        if row['condition'] == 'S0_full':
            label = r'\textbf{Full (Ours)}'
        cells = [label] + [row[f'{key}_cell'] for key, _, _, _ in METRIC_COLUMNS]
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ])
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def write_markdown(summary: List[Dict], tests: List[Dict], path: str) -> None:
    """Detailed paired statistics (unified table written separately)."""
    alpha = tests[0]['bonferroni_alpha'] if tests else 0.05
    lines = [
        '# Generator structural ablation — detailed statistics',
        '',
        'Primary table: `structural_ablation_table.md` (all metrics + significance).',
        '',
        f'Wilcoxon signed-rank vs Full; Bonferroni α={alpha:.4g}.',
        '',
    ]
    for t in tests:
        lines.append(f"### {t['label']} (n={t['n_pairs']})")
        for k in METRIC_KEYS:
            _append_test_line(lines, t, k)
        lines.append('')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _append_test_line(lines: List[str], t: Dict, k: str) -> None:
    p_key = f'{k}_p'
    if p_key in t:
        sig = '*' if t.get(f'{k}_significant') else ''
        lines.append(f"- {k}: p={t[p_key]:.4g}{sig}, Δmean={t[f'{k}_delta_mean']:.4g}")


def compute_structural_ablation(
    subs: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    force_freq: bool = False,
) -> Dict:
    rows = collect_rows(subs, device=device, force_freq=force_freq)
    if not rows:
        raise SystemExit(
            'No structural ablation metrics found. Train first:\n'
            '  bash scripts/train_structural_ablation.sh'
        )
    summary = summarize(rows)
    tests = paired_tests(rows)
    return {'summary': {s['condition']: s for s in summary}, 'tests': tests, 'rows': rows}


def main():
    p = argparse.ArgumentParser(description='Aggregate generator structural ablation')
    p.add_argument('--all', action='store_true', help='All 10 subjects')
    p.add_argument('--sub', type=int, default=None)
    p.add_argument('--force_freq', action='store_true', help='Recompute frequency metrics from checkpoints')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    subs = list(range(1, 11)) if args.all or args.sub is None else [args.sub]

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    data = compute_structural_ablation(subs, device=device, force_freq=args.force_freq)
    summary_list = [data['summary'][c] for c in STRUCTURAL_ORDER if c in data['summary']]
    rows = data['rows']
    tests = data['tests']

    os.makedirs(STRUCTURAL_RAW_DIR, exist_ok=True)
    with open(os.path.join(STRUCTURAL_RAW_DIR, 'summary.json'), 'w') as f:
        json.dump({'summary': data['summary'], 'tests': tests}, f, indent=2)

    csv_path = os.path.join(STRUCTURAL_RAW_DIR, 'structural_ablation.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sum_path = os.path.join(STRUCTURAL_RAW_DIR, 'structural_ablation_summary.csv')
    with open(sum_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_list[0].keys()))
        writer.writeheader()
        writer.writerows(summary_list)

    table_rows = build_unified_table_rows(summary_list, tests)
    write_unified_table_md(
        table_rows,
        os.path.join(STRUCTURAL_RAW_DIR, 'structural_ablation_table.md'),
    )
    write_unified_table_csv(
        table_rows,
        os.path.join(STRUCTURAL_RAW_DIR, 'structural_ablation_table.csv'),
    )
    write_unified_table_tex(
        table_rows,
        os.path.join(STRUCTURAL_RAW_DIR, 'structural_ablation_table.tex'),
    )

    write_markdown(
        summary_list,
        tests,
        os.path.join(STRUCTURAL_RAW_DIR, 'structural_ablation.md'),
    )

    print(f'Wrote {len(rows)} rows, {len(summary_list)} conditions -> {STRUCTURAL_RAW_DIR}')
    for s in summary_list:
        print(
            f"  {s['label']:28s}  n={s['n_subjects']}  "
            f"mse={s['test_mse_mean']:.4f}  r={s['test_corr_mean']:.3f}  "
            f"bp_corr={s['bandpower_corr_mean']:.3f}  "
            f"sem_cos={s['test_semantic_cosine_mean']:.3f}"
        )


if __name__ == '__main__':
    main()

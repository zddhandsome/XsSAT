"""
消融实验结果分析脚本

解析 test 日志，生成 Markdown/LaTeX 格式消融表 (mean ± std)。

Usage:
    python scripts/analyze_ablation.py [--result_dir results/ablation]
"""

import os
import re
import argparse
import numpy as np
from collections import defaultdict


EXPERIMENTS = {
    'A0_full':            'Full Model',
    'A1_no_multichannel': 'w/o MC-VSM',
    'A2_no_negation':     'w/o Negation',
    'A3_full_attention':  'w/o Axial Attn',
    'A4_mean_pool':       'w/o Semantic RO',
    'A5_no_polarity':     'w/o Polarity',
    'A6_no_core_loss':    'w/o Core Loss',
}

SEEDS = [42, 123, 456]


def parse_test_log(log_path):
    """从 test log 中提取指标"""
    metrics = {}
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # SAT Accuracy
    m = re.search(r'Accuracy:\s+([\d.]+)%', content)
    if m:
        metrics['sat_acc'] = float(m.group(1))

    # F1
    m = re.search(r'F1 Score:\s+([\d.]+)%', content)
    if m:
        metrics['f1'] = float(m.group(1))

    # ROC-AUC
    m = re.search(r'ROC-AUC:\s+([\d.]+)', content)
    if m:
        metrics['auc_roc'] = float(m.group(1))

    # Core F1
    m = re.search(r'Avg Core F1:\s+([\d.]+)%', content)
    if m:
        metrics['core_f1'] = float(m.group(1))

    # Total parameters
    m = re.search(r'Total parameters:\s+([\d,]+)', content)
    if m:
        params = int(m.group(1).replace(',', ''))
        metrics['params_k'] = params / 1000.0

    return metrics if metrics else None


def format_mean_std(values, fmt='.2f'):
    """格式化 mean ± std"""
    if not values:
        return '-'
    mean = np.mean(values)
    if len(values) > 1:
        std = np.std(values, ddof=1)
        return f'{mean:{fmt}} ± {std:{fmt}}'
    return f'{mean:{fmt}}'


def generate_markdown_table(results):
    """生成 Markdown 格式消融表"""
    header = '| ID | Method | SAT Acc (%) | F1 (%) | AUC-ROC | Core F1 (%) | Params (K) |'
    separator = '|---|---|---|---|---|---|---|'

    rows = [header, separator]
    for exp_id, exp_name in EXPERIMENTS.items():
        if exp_id not in results:
            rows.append(f'| {exp_id} | {exp_name} | - | - | - | - | - |')
            continue

        data = results[exp_id]
        sat_acc = format_mean_std(data.get('sat_acc', []))
        f1 = format_mean_std(data.get('f1', []))
        auc_roc = format_mean_std(data.get('auc_roc', []), '.4f')
        core_f1 = format_mean_std(data.get('core_f1', []))
        params = f"{data['params_k'][0]:.1f}" if data.get('params_k') else '-'

        rows.append(f'| {exp_id} | {exp_name} | {sat_acc} | {f1} | {auc_roc} | {core_f1} | {params} |')

    return '\n'.join(rows)


def generate_latex_table(results):
    """生成 LaTeX 格式消融表"""
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Ablation study on SR(3-10). Mean $\pm$ std over 3 seeds.}',
        r'\label{tab:ablation}',
        r'\begin{tabular}{l l c c c c c}',
        r'\toprule',
        r'ID & Method & SAT Acc (\%) & F1 (\%) & AUC-ROC & Core F1 (\%) & Params (K) \\',
        r'\midrule',
    ]

    for exp_id, exp_name in EXPERIMENTS.items():
        if exp_id not in results:
            lines.append(f'{exp_id} & {exp_name} & - & - & - & - & - \\\\')
            continue

        data = results[exp_id]
        sat_acc = format_mean_std(data.get('sat_acc', []))
        f1 = format_mean_std(data.get('f1', []))
        auc_roc = format_mean_std(data.get('auc_roc', []), '.4f')
        core_f1 = format_mean_std(data.get('core_f1', []))
        params = f"{data['params_k'][0]:.1f}" if data.get('params_k') else '-'

        # Bold best results for A0
        if exp_id == 'A0_full':
            sat_acc = f'\\textbf{{{sat_acc}}}'
            f1 = f'\\textbf{{{f1}}}'

        lines.append(f'{exp_id} & {exp_name} & {sat_acc} & {f1} & {auc_roc} & {core_f1} & {params} \\\\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation results')
    parser.add_argument('--result_dir', type=str, default='results/ablation',
                        help='消融实验结果目录')
    parser.add_argument('--format', type=str, default='both',
                        choices=['markdown', 'latex', 'both'],
                        help='输出格式')
    args = parser.parse_args()

    results = defaultdict(lambda: defaultdict(list))

    for exp_id in EXPERIMENTS:
        for seed in SEEDS:
            log_file = os.path.join(args.result_dir, f'{exp_id}_seed{seed}_test.log')
            metrics = parse_test_log(log_file)
            if metrics:
                for k, v in metrics.items():
                    results[exp_id][k].append(v)

    if not results:
        print(f"No results found in {args.result_dir}")
        print("Please run experiments first: bash scripts/run_ablation.sh")
        return

    print("=" * 70)
    print("GeoSATformer v2 Ablation Study Results")
    print("=" * 70)

    found_count = sum(1 for exp_id in EXPERIMENTS if exp_id in results)
    total_runs = sum(len(results[exp_id].get('sat_acc', [])) for exp_id in EXPERIMENTS)
    print(f"Found {found_count}/{len(EXPERIMENTS)} experiments, {total_runs} total runs\n")

    if args.format in ('markdown', 'both'):
        print("### Markdown Table\n")
        md_table = generate_markdown_table(results)
        print(md_table)
        # Save to file
        out_path = os.path.join(args.result_dir, 'ablation_table.md')
        with open(out_path, 'w') as f:
            f.write(md_table)
        print(f"\nSaved to: {out_path}")

    if args.format in ('latex', 'both'):
        print("\n\n### LaTeX Table\n")
        latex_table = generate_latex_table(results)
        print(latex_table)
        # Save to file
        out_path = os.path.join(args.result_dir, 'ablation_table.tex')
        with open(out_path, 'w') as f:
            f.write(latex_table)
        print(f"\nSaved to: {out_path}")

    # Print delta from baseline
    if 'A0_full' in results and results['A0_full'].get('sat_acc'):
        baseline_acc = np.mean(results['A0_full']['sat_acc'])
        print("\n\n### Delta from Baseline (SAT Acc)")
        print("-" * 40)
        for exp_id, exp_name in EXPERIMENTS.items():
            if exp_id == 'A0_full':
                continue
            if exp_id in results and results[exp_id].get('sat_acc'):
                delta = np.mean(results[exp_id]['sat_acc']) - baseline_acc
                sign = '+' if delta >= 0 else ''
                print(f"  {exp_name:20s}: {sign}{delta:.2f}%")


if __name__ == '__main__':
    main()

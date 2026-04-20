"""
分析数据集的 c/v ratio 分布
"""
import torch
import numpy as np
from pathlib import Path

def analyze_dataset(data_path):
    """分析单个数据集文件的 c/v ratio"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {data_path}")
    print(f"{'='*60}")

    data = torch.load(data_path)
    metadata = data['metadata']

    cv_ratios = []
    num_vars_list = []
    num_clauses_list = []

    for meta in metadata:
        if 'error' not in meta:
            num_vars = meta['num_vars']
            num_clauses = meta['num_clauses']

            if num_vars > 0:
                cv_ratio = num_clauses / num_vars
                cv_ratios.append(cv_ratio)
                num_vars_list.append(num_vars)
                num_clauses_list.append(num_clauses)

    cv_ratios = np.array(cv_ratios)
    num_vars_list = np.array(num_vars_list)
    num_clauses_list = np.array(num_clauses_list)

    # 统计信息
    print(f"\nDataset size: {len(metadata)} instances")
    print(f"Valid instances: {len(cv_ratios)}")

    print(f"\n--- Number of Variables ---")
    print(f"  Min: {num_vars_list.min()}")
    print(f"  Max: {num_vars_list.max()}")
    print(f"  Mean: {num_vars_list.mean():.2f}")
    print(f"  Median: {np.median(num_vars_list):.2f}")

    print(f"\n--- Number of Clauses ---")
    print(f"  Min: {num_clauses_list.min()}")
    print(f"  Max: {num_clauses_list.max()}")
    print(f"  Mean: {num_clauses_list.mean():.2f}")
    print(f"  Median: {np.median(num_clauses_list):.2f}")

    print(f"\n--- C/V Ratio ---")
    print(f"  Min: {cv_ratios.min():.4f}")
    print(f"  Max: {cv_ratios.max():.4f}")
    print(f"  Mean: {cv_ratios.mean():.4f}")
    print(f"  Median: {np.median(cv_ratios):.4f}")
    print(f"  Std: {cv_ratios.std():.4f}")

    # 分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n--- C/V Ratio Percentiles ---")
    for p in percentiles:
        value = np.percentile(cv_ratios, p)
        print(f"  {p}th percentile: {value:.4f}")

    # SAT/UNSAT 分布
    sat_labels = data['sat_label'].numpy()
    sat_count = (sat_labels == 1).sum()
    unsat_count = (sat_labels == 0).sum()
    print(f"\n--- SAT/UNSAT Distribution ---")
    print(f"  SAT instances: {sat_count} ({sat_count/len(sat_labels)*100:.1f}%)")
    print(f"  UNSAT instances: {unsat_count} ({unsat_count/len(sat_labels)*100:.1f}%)")

    # C/V ratio 直方图分布
    print(f"\n--- C/V Ratio Distribution (histogram) ---")
    bins = [0, 2, 3, 4, 4.26, 4.5, 5, 6, 100]
    hist, _ = np.histogram(cv_ratios, bins=bins)
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(cv_ratios) * 100
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count} ({pct:.1f}%)")

    return cv_ratios, num_vars_list, num_clauses_list, sat_labels


if __name__ == '__main__':
    data_dir = Path('data_test_cv3')

    all_results = {}

    for split in ['train', 'val', 'test']:
        data_path = data_dir / f'{split}.pt'
        if data_path.exists():
            cv_ratios, num_vars, num_clauses, sat_labels = analyze_dataset(data_path)
            all_results[split] = {
                'cv_ratios': cv_ratios,
                'num_vars': num_vars,
                'num_clauses': num_clauses,
                'sat_labels': sat_labels
            }

    # 总体统计
    if all_results:
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS")
        print(f"{'='*60}")

        all_cv = np.concatenate([v['cv_ratios'] for v in all_results.values()])
        all_vars = np.concatenate([v['num_vars'] for v in all_results.values()])
        all_clauses = np.concatenate([v['num_clauses'] for v in all_results.values()])

        print(f"\nTotal instances: {len(all_cv)}")
        print(f"\nVariables range: [{all_vars.min()}, {all_vars.max()}]")
        print(f"Clauses range: [{all_clauses.min()}, {all_clauses.max()}]")
        print(f"\nOverall C/V ratio:")
        print(f"  Mean: {all_cv.mean():.4f}")
        print(f"  Median: {np.median(all_cv):.4f}")
        print(f"  Std: {all_cv.std():.4f}")
        print(f"  Range: [{all_cv.min():.4f}, {all_cv.max():.4f}]")

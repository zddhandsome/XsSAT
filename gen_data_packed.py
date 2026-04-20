"""
生成打包的 SAT 数据集（单个 .pt 文件）

将所有数据和标签打包成单个文件，避免大量小文件导致的性能问题。

生成的文件：
- train.pt: 训练集
- val.pt: 验证集
- test.pt: 测试集

每个 .pt 文件包含：
- vsm: Variable Space Matrix [N, max_clauses, max_vars]
- clause_mask: 子句掩码 [N, max_clauses]
- var_mask: 变量掩码 [N, max_vars]
- sat_label: SAT/UNSAT 标签 [N]
- core_labels: UNSAT Core 标签 [N, max_clauses]
- vsids_labels: VSIDS 分数 [N, max_vars]
- metadata: 元数据列表

需要安装: pip install python-sat
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.generator import RandomSATGenerator, UNSATGenerator
from src.data.vsm_encoder import VSMEncoder


def solve_and_get_labels(clauses: List[List[int]], num_vars: int, num_clauses: int) -> dict:
    """
    求解 CNF 并生成标签

    Args:
        clauses: 子句列表
        num_vars: 变量数
        num_clauses: 子句数

    Returns:
        包含标签信息的字典
    """
    try:
        from pysat.solvers import Solver
        from pysat.formula import CNF
    except ImportError:
        raise ImportError("请安装 python-sat: pip install python-sat")

    result = {
        'sat_label': None,
        'core': None,
        'vsids': None,
        'error': None
    }

    # 创建 CNF 对象
    cnf = CNF()
    for clause in clauses:
        cnf.append(clause)

    # 使用 Glucose4 求解
    solver = Solver(name='glucose4')

    try:
        for clause in clauses:
            solver.add_clause(clause)

        is_sat = solver.solve()

        if is_sat is True:
            result['sat_label'] = 1
            # SAT 实例：生成基于变量出现频率的伪 VSIDS 分数
            var_freq = np.zeros(num_vars, dtype=np.float32)
            for clause in clauses:
                for lit in clause:
                    var_freq[abs(lit) - 1] += 1
            if var_freq.max() > 0:
                var_freq = var_freq / var_freq.max()
            result['vsids'] = var_freq
            # SAT 实例没有 core
            result['core'] = np.zeros(num_clauses, dtype=np.float32)

        elif is_sat is False:
            result['sat_label'] = 0
            # UNSAT 实例：提取 MUS
            mus_indices = extract_mus(cnf)
            core = np.zeros(num_clauses, dtype=np.float32)
            for idx in mus_indices:
                if idx - 1 < num_clauses:  # 1-indexed to 0-indexed
                    core[idx - 1] = 1.0
            result['core'] = core

            # 生成 VSIDS 分数
            var_freq = np.zeros(num_vars, dtype=np.float32)
            for clause in clauses:
                for lit in clause:
                    var_freq[abs(lit) - 1] += 1
            if var_freq.max() > 0:
                var_freq = var_freq / var_freq.max()
            result['vsids'] = var_freq
        else:
            result['error'] = 'UNKNOWN'

    except Exception as e:
        result['error'] = str(e)
    finally:
        solver.delete()

    return result


def extract_mus(cnf) -> List[int]:
    """提取最小不可满足子集 (MUS)"""
    try:
        from pysat.examples.musx import MUSX

        musx = MUSX(cnf, verbosity=0)
        mus = musx.compute()
        musx.delete()

        if mus:
            return mus
        else:
            return list(range(1, len(cnf.clauses) + 1))
    except Exception:
        # 简单删除法
        return extract_mus_simple(cnf)


def extract_mus_simple(cnf) -> List[int]:
    """简单的 MUS 提取（删除法）"""
    from pysat.solvers import Solver

    clauses = list(cnf.clauses)
    mus_indices = list(range(len(clauses)))

    i = 0
    while i < len(mus_indices):
        test_indices = mus_indices[:i] + mus_indices[i+1:]
        test_clauses = [clauses[j] for j in test_indices]

        solver = Solver(name='glucose4')
        for c in test_clauses:
            solver.add_clause(c)

        is_sat = solver.solve()
        solver.delete()

        if is_sat is False:
            mus_indices = test_indices
        else:
            i += 1

    return [idx + 1 for idx in mus_indices]


def generate_packed_dataset(
    output_dir: str,
    num_train: int = 10000,
    num_val: int = 1000,
    num_test: int = 1000,
    max_clauses: int = 200,
    max_vars: int = 104,
    k: int = 3,
    min_vars: int = 10,
    gen_max_vars: int = 100,
    cv_ratio: float = 4.26,
    sat_ratio: float = 0.5,
    seed: int = 42
):
    """
    生成打包的数据集

    Args:
        output_dir: 输出目录
        num_train/val/test: 各数据集大小
        max_clauses: VSM 最大子句数
        max_vars: VSM 最大变量数
        k: k-SAT
        min_vars: 生成时最小变量数
        gen_max_vars: 生成时最大变量数
        cv_ratio: clause/variable 比率
        sat_ratio: 随机 SAT 实例比例
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 确保生成的实例不超过 VSM 限制
    # gen_max_vars 不能超过 max_vars
    actual_gen_max_vars = min(gen_max_vars, max_vars)
    # 根据 cv_ratio 估算最大子句数，确保不超过 max_clauses
    max_possible_clauses = int(actual_gen_max_vars * (cv_ratio + 0.5))
    if max_possible_clauses > max_clauses:
        # 调整 gen_max_vars 使生成的子句数不超过 max_clauses
        actual_gen_max_vars = int(max_clauses / (cv_ratio + 0.5))
        print(f"Warning: Adjusted gen_max_vars to {actual_gen_max_vars} to fit max_clauses={max_clauses}")

    # 创建生成器
    random_gen = RandomSATGenerator(
        k=k,
        min_vars=min_vars,
        max_vars=actual_gen_max_vars,
        cv_ratio=cv_ratio
    )
    unsat_gen = UNSATGenerator(
        min_pigeons=3,
        max_pigeons=min(10, actual_gen_max_vars // 10)
    )

    # VSM 编码器
    vsm_encoder = VSMEncoder(max_clauses, max_vars)

    for split, num_instances in [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]:
        print(f"\n{'='*60}")
        print(f"Generating {split} set: {num_instances} instances")
        print(f"{'='*60}")

        # 预分配数组
        all_vsm = np.zeros((num_instances, max_clauses, max_vars), dtype=np.float32)
        all_clause_mask = np.zeros((num_instances, max_clauses), dtype=np.bool_)
        all_var_mask = np.zeros((num_instances, max_vars), dtype=np.bool_)
        all_sat_label = np.zeros(num_instances, dtype=np.int64)
        all_core_labels = np.zeros((num_instances, max_clauses), dtype=np.float32)
        all_vsids_labels = np.zeros((num_instances, max_vars), dtype=np.float32)
        metadata = []

        sat_count = 0
        unsat_count = 0
        error_count = 0

        for i in tqdm(range(num_instances), desc=f"Generating {split}"):
            # 生成实例
            if random.random() < sat_ratio:
                clauses, num_vars, num_clauses = random_gen.generate()
            else:
                clauses, num_vars, num_clauses = unsat_gen.generate()

            # 编码为 VSM
            vsm, clause_mask, var_mask = vsm_encoder.encode(clauses, num_vars, num_clauses)

            # 求解并获取标签
            labels = solve_and_get_labels(clauses, num_vars, num_clauses)

            if labels['error'] is not None:
                error_count += 1
                # 使用默认值
                all_vsm[i] = vsm
                all_clause_mask[i] = clause_mask
                all_var_mask[i] = var_mask
                all_sat_label[i] = 0
                metadata.append({
                    'num_vars': num_vars,
                    'num_clauses': num_clauses,
                    'error': labels['error']
                })
                continue

            # 存储数据
            all_vsm[i] = vsm
            all_clause_mask[i] = clause_mask
            all_var_mask[i] = var_mask
            all_sat_label[i] = labels['sat_label']

            # 存储 core labels（截断到 max_clauses）
            core = labels['core']
            core_len = min(len(core), max_clauses)
            all_core_labels[i, :core_len] = core[:core_len]

            # 存储 vsids labels（截断到 max_vars）
            vsids = labels['vsids']
            vsids_len = min(len(vsids), max_vars)
            all_vsids_labels[i, :vsids_len] = vsids[:vsids_len]

            if labels['sat_label'] == 1:
                sat_count += 1
            else:
                unsat_count += 1

            metadata.append({
                'num_vars': num_vars,
                'num_clauses': num_clauses
            })

        # 保存为 .pt 文件
        data = {
            'vsm': torch.from_numpy(all_vsm),
            'clause_mask': torch.from_numpy(all_clause_mask),
            'var_mask': torch.from_numpy(all_var_mask),
            'sat_label': torch.from_numpy(all_sat_label),
            'core_labels': torch.from_numpy(all_core_labels),
            'vsids_labels': torch.from_numpy(all_vsids_labels),
            'metadata': metadata,
            'config': {
                'max_clauses': max_clauses,
                'max_vars': max_vars,
                'k': k,
                'cv_ratio': cv_ratio
            }
        }

        save_path = output_path / f'{split}.pt'
        torch.save(data, save_path)

        file_size = save_path.stat().st_size / (1024 * 1024)

        print(f"\n{split} set statistics:")
        print(f"  SAT instances: {sat_count}")
        print(f"  UNSAT instances: {unsat_count}")
        print(f"  Errors: {error_count}")
        print(f"  Saved to: {save_path}")
        print(f"  File size: {file_size:.2f} MB")

    print(f"\n{'='*60}")
    print("Dataset generation completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def convert_existing_to_packed(
    input_dir: str,
    output_dir: str,
    max_clauses: int = 200,
    max_vars: int = 104
):
    """
    将已有的分散文件数据集转换为打包格式

    Args:
        input_dir: 输入目录（包含 train/val/test 子目录）
        output_dir: 输出目录
        max_clauses: 最大子句数
        max_vars: 最大变量数
    """
    from src.data import SATDataset

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        split_dir = input_path / split
        if not split_dir.exists():
            print(f"Skipping {split}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Converting {split} set...")
        print(f"{'='*60}")

        # 加载数据集
        dataset = SATDataset(
            str(split_dir),
            max_clauses=max_clauses,
            max_vars=max_vars,
            load_labels=True
        )

        num_instances = len(dataset)
        print(f"Found {num_instances} instances")

        # 预分配数组
        all_vsm = np.zeros((num_instances, max_clauses, max_vars), dtype=np.float32)
        all_clause_mask = np.zeros((num_instances, max_clauses), dtype=np.bool_)
        all_var_mask = np.zeros((num_instances, max_vars), dtype=np.bool_)
        all_sat_label = np.zeros(num_instances, dtype=np.int64)
        all_core_labels = np.zeros((num_instances, max_clauses), dtype=np.float32)
        all_vsids_labels = np.zeros((num_instances, max_vars), dtype=np.float32)
        metadata = []

        for i in tqdm(range(num_instances), desc=f"Converting {split}"):
            sample = dataset[i]

            all_vsm[i] = sample['vsm'].numpy()
            all_clause_mask[i] = sample['clause_mask'].numpy()
            all_var_mask[i] = sample['var_mask'].numpy()

            if 'sat_label' in sample:
                all_sat_label[i] = sample['sat_label'].item()

            if 'core_labels' in sample:
                all_core_labels[i] = sample['core_labels'].numpy()

            if 'vsids_labels' in sample:
                all_vsids_labels[i] = sample['vsids_labels'].numpy()

            metadata.append({
                'name': sample.get('name', f'instance_{i}'),
                'num_vars': sample.get('num_vars', 0),
                'num_clauses': sample.get('num_clauses', 0)
            })

        # 保存
        data = {
            'vsm': torch.from_numpy(all_vsm),
            'clause_mask': torch.from_numpy(all_clause_mask),
            'var_mask': torch.from_numpy(all_var_mask),
            'sat_label': torch.from_numpy(all_sat_label),
            'core_labels': torch.from_numpy(all_core_labels),
            'vsids_labels': torch.from_numpy(all_vsids_labels),
            'metadata': metadata,
            'config': {
                'max_clauses': max_clauses,
                'max_vars': max_vars
            }
        }

        save_path = output_path / f'{split}.pt'
        torch.save(data, save_path)

        file_size = save_path.stat().st_size / (1024 * 1024)
        print(f"Saved to: {save_path} ({file_size:.2f} MB)")

    print(f"\n{'='*60}")
    print("Conversion completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate packed SAT dataset')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # 生成新数据集
    gen_parser = subparsers.add_parser('generate', help='Generate new packed dataset')
    gen_parser.add_argument('--output_dir', type=str, default='data_packed',
                           help='Output directory')
    gen_parser.add_argument('--num_train', type=int, default=10000,
                           help='Number of training instances')
    gen_parser.add_argument('--num_val', type=int, default=1000,
                           help='Number of validation instances')
    gen_parser.add_argument('--num_test', type=int, default=1000,
                           help='Number of test instances')
    gen_parser.add_argument('--max_clauses', type=int, default=500,
                           help='Max clauses in VSM')
    gen_parser.add_argument('--max_vars', type=int, default=167,
                           help='Max variables in VSM')
    gen_parser.add_argument('--k', type=int, default=3,
                           help='k for k-SAT')
    gen_parser.add_argument('--min_vars', type=int, default=70,
                           help='Min variables when generating')
    gen_parser.add_argument('--gen_max_vars', type=int, default=167,
                           help='Max variables when generating')
    gen_parser.add_argument('--cv_ratio', type=float, default=4.26,
                           help='Clause/variable ratio')
    gen_parser.add_argument('--sat_ratio', type=float, default=0.5,
                           help='Ratio of random SAT instances')
    gen_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')

    # 转换已有数据集
    conv_parser = subparsers.add_parser('convert', help='Convert existing dataset to packed format')
    conv_parser.add_argument('--input_dir', type=str, required=True,
                            help='Input directory with train/val/test subdirs')
    conv_parser.add_argument('--output_dir', type=str, default='data_packed',
                            help='Output directory')
    conv_parser.add_argument('--max_clauses', type=int, default=200,
                            help='Max clauses')
    conv_parser.add_argument('--max_vars', type=int, default=104,
                            help='Max variables')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_packed_dataset(
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            max_clauses=args.max_clauses,
            max_vars=args.max_vars,
            k=args.k,
            min_vars=args.min_vars,
            gen_max_vars=args.gen_max_vars,
            cv_ratio=args.cv_ratio,
            sat_ratio=args.sat_ratio,
            seed=args.seed
        )
    elif args.command == 'convert':
        convert_existing_to_packed(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_clauses=args.max_clauses,
            max_vars=args.max_vars
        )
    else:
        parser.print_help()

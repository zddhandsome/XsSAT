"""
快速生成打包的 SAT 数据集（多进程并行版本）

优化点：
1. 多进程并行生成和求解
2. 简化 MUS 提取（使用近似方法）
3. 批量保存减少 I/O

用法：
    python gen_data_packed_fast.py generate --num_train 10000 --num_workers 8
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
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.legacy.data.generator import RandomSATGenerator, UNSATGenerator
from src.data.vsm_encoder import VSMEncoder


def generate_fixed_sat_instance(num_vars: int, num_clauses: int, k: int = 3) -> List[List[int]]:
    """
    生成固定 var 和 clause 数量的 k-SAT 实例

    Args:
        num_vars: 固定的变量数
        num_clauses: 固定的子句数
        k: 每个子句的文字数

    Returns:
        clauses: 子句列表
    """
    clauses = []
    for _ in range(num_clauses):
        # 随机选择 k 个不同的变量
        vars_in_clause = random.sample(range(1, num_vars + 1), min(k, num_vars))
        # 随机决定每个变量的极性
        clause = [var if random.random() < 0.5 else -var for var in vars_in_clause]
        clauses.append(clause)
    return clauses


def compute_clause_length_stats(clauses: List[List[int]]) -> Dict[str, float]:
    """统计一个 CNF 实例中的子句长度分布摘要。"""
    if not clauses:
        return {
            'min_clause_len': 0,
            'max_clause_len': 0,
            'avg_clause_len': 0.0,
        }

    clause_lengths = [len(clause) for clause in clauses]
    return {
        'min_clause_len': min(clause_lengths),
        'max_clause_len': max(clause_lengths),
        'avg_clause_len': float(sum(clause_lengths) / len(clause_lengths)),
    }


def build_clause_len_profile(profile_dir: str) -> Dict[int, float]:
    """
    从一组 CNF 文件中统计经验子句长度分布。

    Returns:
        {clause_len: probability}
    """
    profile_path = Path(profile_dir)
    if not profile_path.exists():
        raise ValueError(f"Clause length profile dir does not exist: {profile_path}")
    if not profile_path.is_dir():
        raise ValueError(f"Clause length profile path is not a directory: {profile_path}")

    counter: Dict[int, int] = {}
    cnf_files = sorted(profile_path.glob('*.cnf'))
    if not cnf_files:
        raise ValueError(f"No CNF files found under clause length profile dir: {profile_path}")

    total = 0
    for cnf_path in cnf_files:
        current_clause = []
        with cnf_path.open('r') as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith('c') or line.startswith('p') or line.startswith('%'):
                    continue

                for literal_str in line.split():
                    literal = int(literal_str)
                    if literal == 0:
                        if current_clause:
                            clause_len = len(current_clause)
                            counter[clause_len] = counter.get(clause_len, 0) + 1
                            total += 1
                            current_clause = []
                    else:
                        current_clause.append(literal)

        if current_clause:
            clause_len = len(current_clause)
            counter[clause_len] = counter.get(clause_len, 0) + 1
            total += 1

    if total == 0:
        raise ValueError(f"No clauses found under clause length profile dir: {profile_path}")

    return {
        clause_len: count / total
        for clause_len, count in sorted(counter.items())
    }


def sample_clause_len(num_vars: int, config: Dict) -> int:
    """根据配置采样单个子句长度。"""
    if num_vars <= 0:
        return 0
    if num_vars < 2:
        return num_vars

    clause_len_mode = config.get('clause_len_mode', 'fixed')

    if clause_len_mode == 'fixed':
        return min(config.get('k', 3), num_vars)

    if clause_len_mode == 'uniform':
        min_clause_len = config.get('min_clause_len', 2)
        max_clause_len = min(config.get('max_clause_len', min_clause_len), num_vars)
        if max_clause_len < min_clause_len:
            min_clause_len = max_clause_len
        return random.randint(min_clause_len, max_clause_len)

    if clause_len_mode == 'neurosat':
        values = config.get('clause_len_values', [])
        probs = config.get('clause_len_probs', [])
        candidates = [
            (clause_len, prob)
            for clause_len, prob in zip(values, probs)
            if 2 <= clause_len <= num_vars
        ]
        if not candidates:
            return min(config.get('k', 3), num_vars)

        candidate_values = [clause_len for clause_len, _ in candidates]
        candidate_probs = np.asarray([prob for _, prob in candidates], dtype=np.float64)
        candidate_probs /= candidate_probs.sum()
        return int(np.random.choice(candidate_values, p=candidate_probs))

    raise ValueError(f"Unknown clause_len_mode: {clause_len_mode}")


def generate_random_cnf_instance(num_vars: int, num_clauses: int, config: Dict) -> List[List[int]]:
    """生成支持 mixed clause length 的随机 CNF 实例。"""
    clauses = []
    for _ in range(num_clauses):
        clause_len = sample_clause_len(num_vars, config)
        if clause_len <= 0:
            continue

        vars_in_clause = random.sample(range(1, num_vars + 1), clause_len)
        clause = [var if random.random() < 0.5 else -var for var in vars_in_clause]
        clauses.append(clause)

    return clauses


def solve_single_instance_fixed(args_tuple):
    """
    生成固定 var/clause 数量的单个实例（用于多进程）

    Args:
        args_tuple: (instance_id, config_dict)

    Returns:
        处理结果字典
    """
    instance_id, config = args_tuple

    # 设置随机种子
    seed = config['seed'] + instance_id
    random.seed(seed)
    np.random.seed(seed)

    try:
        from pysat.solvers import Solver
    except ImportError:
        return {'error': 'pysat not installed', 'instance_id': instance_id}

    # 固定的参数
    num_vars = config['fixed_vars']
    num_clauses = config['fixed_clauses']
    k = config['k']

    # VSM 编码器
    vsm_encoder = VSMEncoder(config['max_clauses'], config['max_vars'])

    # 生成固定大小的实例
    clauses = generate_fixed_sat_instance(num_vars, num_clauses, k)

    # 编码为 VSM
    vsm, clause_mask, var_mask = vsm_encoder.encode(clauses, num_vars, num_clauses)

    # 求解
    result = {
        'instance_id': instance_id,
        'vsm': vsm,
        'clause_mask': clause_mask,
        'var_mask': var_mask,
        'num_vars': num_vars,
        'num_clauses': num_clauses,
        'sat_label': None,
        'core': None,
        'vsids': None,
        'error': None
    }

    solver = Solver(name='glucose4')

    try:
        for clause in clauses:
            solver.add_clause(clause)

        is_sat = solver.solve()

        # 计算变量频率（用于 VSIDS）
        var_freq = np.zeros(num_vars, dtype=np.float32)
        for clause in clauses:
            for lit in clause:
                var_freq[abs(lit) - 1] += 1
        if var_freq.max() > 0:
            var_freq = var_freq / var_freq.max()
        result['vsids'] = var_freq

        if is_sat is True:
            result['sat_label'] = 1
            result['core'] = np.zeros(num_clauses, dtype=np.float32)
        elif is_sat is False:
            result['sat_label'] = 0
            core = np.zeros(num_clauses, dtype=np.float32)
            try:
                unsat_core = solver.get_core()
                if unsat_core:
                    for idx in unsat_core:
                        if 0 <= idx - 1 < num_clauses:
                            core[idx - 1] = 1.0
                else:
                    core = fast_approximate_mus(clauses, num_clauses)
            except:
                core = fast_approximate_mus(clauses, num_clauses)
            result['core'] = core
        else:
            result['error'] = 'UNKNOWN'

    except Exception as e:
        result['error'] = str(e)
    finally:
        solver.delete()

    return result


def solve_single_instance(args_tuple):
    """
    单个实例的生成和求解（用于多进程）

    Args:
        args_tuple: (instance_id, config_dict)

    Returns:
        处理结果字典
    """
    instance_id, config = args_tuple

    # 设置随机种子（每个进程不同）
    seed = config['seed'] + instance_id
    random.seed(seed)
    np.random.seed(seed)

    try:
        from pysat.solvers import Solver
        from pysat.formula import CNF
    except ImportError:
        return {'error': 'pysat not installed', 'instance_id': instance_id}

    # 创建生成器
    random_gen = RandomSATGenerator(
        k=config['k'],
        min_vars=config['min_vars'],
        max_vars=config['gen_max_vars'],
        cv_ratio=config['cv_ratio']
    )
    unsat_gen = UNSATGenerator(
        min_pigeons=3,
        max_pigeons=min(10, config['gen_max_vars'] // 10)
    )

    # VSM 编码器
    vsm_encoder = VSMEncoder(config['max_clauses'], config['max_vars'])

    # 生成实例
    if random.random() < config['sat_ratio']:
        clauses, num_vars, num_clauses = random_gen.generate()
    else:
        clauses, num_vars, num_clauses = unsat_gen.generate()

    # 编码为 VSM
    vsm, clause_mask, var_mask = vsm_encoder.encode(clauses, num_vars, num_clauses)

    # 求解
    result = {
        'instance_id': instance_id,
        'vsm': vsm,
        'clause_mask': clause_mask,
        'var_mask': var_mask,
        'num_vars': num_vars,
        'num_clauses': num_clauses,
        'sat_label': None,
        'core': None,
        'vsids': None,
        'error': None
    }

    solver = Solver(name='glucose4')

    try:
        for clause in clauses:
            solver.add_clause(clause)

        is_sat = solver.solve()

        # 计算变量频率（用于 VSIDS）
        var_freq = np.zeros(num_vars, dtype=np.float32)
        for clause in clauses:
            for lit in clause:
                var_freq[abs(lit) - 1] += 1
        if var_freq.max() > 0:
            var_freq = var_freq / var_freq.max()
        result['vsids'] = var_freq

        if is_sat is True:
            result['sat_label'] = 1
            result['core'] = np.zeros(num_clauses, dtype=np.float32)
        elif is_sat is False:
            result['sat_label'] = 0
            # 快速 MUS 近似：使用 solver 的 core（如果支持）
            core = np.zeros(num_clauses, dtype=np.float32)
            try:
                # 尝试获取 unsat core
                unsat_core = solver.get_core()
                if unsat_core:
                    for idx in unsat_core:
                        if 0 <= idx - 1 < num_clauses:
                            core[idx - 1] = 1.0
                else:
                    # 如果没有 core，使用简化的启发式方法
                    core = fast_approximate_mus(clauses, num_clauses)
            except:
                # 使用启发式方法
                core = fast_approximate_mus(clauses, num_clauses)
            result['core'] = core
        else:
            result['error'] = 'UNKNOWN'

    except Exception as e:
        result['error'] = str(e)
    finally:
        solver.delete()

    return result


def fast_approximate_mus(clauses: List[List[int]], num_clauses: int) -> np.ndarray:
    """
    快速近似 MUS（不精确但快速）

    使用启发式：短子句和包含高频变量的子句更可能在 MUS 中
    """
    core = np.zeros(num_clauses, dtype=np.float32)

    # 计算每个子句的"重要性"分数
    var_count = {}
    for clause in clauses:
        for lit in clause:
            var = abs(lit)
            var_count[var] = var_count.get(var, 0) + 1

    scores = []
    for i, clause in enumerate(clauses):
        # 短子句更重要
        length_score = 1.0 / len(clause)
        # 包含高频变量的子句更重要
        var_score = sum(var_count.get(abs(lit), 0) for lit in clause) / len(clause)
        scores.append((i, length_score * var_score))

    # 选择分数最高的子句作为近似 MUS
    scores.sort(key=lambda x: x[1], reverse=True)
    num_select = max(1, num_clauses // 3)  # 选择约 1/3 的子句

    for i in range(min(num_select, len(scores))):
        core[scores[i][0]] = 1.0

    return core


def generate_packed_dataset_parallel(
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
    seed: int = 42,
    num_workers: int = None
):
    """
    并行生成打包的数据集
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Using {num_workers} workers")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 调整参数
    actual_gen_max_vars = min(gen_max_vars, max_vars)
    max_possible_clauses = int(actual_gen_max_vars * (cv_ratio + 0.5))
    if max_possible_clauses > max_clauses:
        actual_gen_max_vars = int(max_clauses / (cv_ratio + 0.5))
        print(f"Warning: Adjusted gen_max_vars to {actual_gen_max_vars}")

    config = {
        'max_clauses': max_clauses,
        'max_vars': max_vars,
        'k': k,
        'min_vars': min_vars,
        'gen_max_vars': actual_gen_max_vars,
        'cv_ratio': cv_ratio,
        'sat_ratio': sat_ratio,
        'seed': seed
    }

    for split, num_instances in [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]:
        print(f"\n{'='*60}")
        print(f"Generating {split} set: {num_instances} instances")
        print(f"{'='*60}")

        start_time = time.time()

        # 准备参数
        args_list = [(i, config.copy()) for i in range(num_instances)]
        # 每个 split 使用不同的种子偏移
        split_offset = {'train': 0, 'val': 100000, 'test': 200000}
        for i in range(len(args_list)):
            args_list[i][1]['seed'] = seed + split_offset[split] + i

        # 并行处理
        results = []
        with Pool(num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(solve_single_instance, args_list),
                total=num_instances,
                desc=f"Generating {split}"
            ):
                results.append(result)

        # 按 instance_id 排序
        results.sort(key=lambda x: x['instance_id'])

        # 收集结果
        all_vsm = np.zeros((num_instances, 2, max_clauses, max_vars), dtype=np.float32)
        all_clause_mask = np.zeros((num_instances, max_clauses), dtype=np.bool_)
        all_var_mask = np.zeros((num_instances, max_vars), dtype=np.bool_)
        all_sat_label = np.zeros(num_instances, dtype=np.int64)
        all_core_labels = np.zeros((num_instances, max_clauses), dtype=np.float32)
        all_vsids_labels = np.zeros((num_instances, max_vars), dtype=np.float32)
        metadata = []

        sat_count = 0
        unsat_count = 0
        error_count = 0

        for i, result in enumerate(results):
            if result.get('error'):
                error_count += 1
                metadata.append({
                    'num_vars': result.get('num_vars', 0),
                    'num_clauses': result.get('num_clauses', 0),
                    'error': result['error']
                })
                continue

            all_vsm[i] = result['vsm']
            all_clause_mask[i] = result['clause_mask']
            all_var_mask[i] = result['var_mask']
            all_sat_label[i] = result['sat_label']

            # Core labels
            core = result['core']
            core_len = min(len(core), max_clauses)
            all_core_labels[i, :core_len] = core[:core_len]

            # VSIDS labels
            vsids = result['vsids']
            vsids_len = min(len(vsids), max_vars)
            all_vsids_labels[i, :vsids_len] = vsids[:vsids_len]

            if result['sat_label'] == 1:
                sat_count += 1
            else:
                unsat_count += 1

            metadata.append({
                'num_vars': result['num_vars'],
                'num_clauses': result['num_clauses']
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
            'config': config
        }

        save_path = output_path / f'{split}.pt'
        torch.save(data, save_path)

        elapsed = time.time() - start_time
        file_size = save_path.stat().st_size / (1024 * 1024)

        print(f"\n{split} set statistics:")
        print(f"  SAT instances: {sat_count}")
        print(f"  UNSAT instances: {unsat_count}")
        print(f"  Errors: {error_count}")
        print(f"  Time: {elapsed:.1f}s ({num_instances/elapsed:.1f} instances/s)")
        print(f"  Saved to: {save_path}")
        print(f"  File size: {file_size:.2f} MB")

    print(f"\n{'='*60}")
    print("Dataset generation completed!")
    print(f"{'='*60}")


def solve_single_instance_mixed_vars(args_tuple):
    """
    生成变化 var 数量（固定 cv_ratio）的单个实例（用于多进程）

    Args:
        args_tuple: (instance_id, config_dict)

    Returns:
        处理结果字典
    """
    instance_id, config = args_tuple

    seed = config['seed'] + instance_id
    random.seed(seed)
    np.random.seed(seed)

    try:
        from pysat.solvers import Solver
    except ImportError:
        return {'error': 'pysat not installed', 'instance_id': instance_id}

    min_vars = config['min_vars']
    max_vars_gen = config['max_vars_gen']
    cv_ratio = config['cv_ratio']
    k = config['k']
    max_clauses = config['max_clauses']
    max_vars = config['max_vars']

    # 随机采样 num_vars
    num_vars = random.randint(min_vars, max_vars_gen)
    num_clauses = int(num_vars * cv_ratio)

    vsm_encoder = VSMEncoder(max_clauses, max_vars)

    clauses = generate_random_cnf_instance(num_vars, num_clauses, config)
    vsm, clause_mask, var_mask = vsm_encoder.encode(clauses, num_vars, num_clauses)
    clause_length_stats = compute_clause_length_stats(clauses)

    result = {
        'instance_id': instance_id,
        'vsm': vsm,
        'clause_mask': clause_mask,
        'var_mask': var_mask,
        'num_vars': num_vars,
        'num_clauses': num_clauses,
        'sat_label': None,
        'core': None,
        'vsids': None,
        'error': None,
        **clause_length_stats,
    }

    solver = Solver(name='glucose4')

    try:
        for clause in clauses:
            solver.add_clause(clause)

        is_sat = solver.solve()

        var_freq = np.zeros(num_vars, dtype=np.float32)
        for clause in clauses:
            for lit in clause:
                var_freq[abs(lit) - 1] += 1
        if var_freq.max() > 0:
            var_freq = var_freq / var_freq.max()
        result['vsids'] = var_freq

        if is_sat is True:
            result['sat_label'] = 1
            result['core'] = np.zeros(num_clauses, dtype=np.float32)
        elif is_sat is False:
            result['sat_label'] = 0
            core = np.zeros(num_clauses, dtype=np.float32)
            try:
                unsat_core = solver.get_core()
                if unsat_core:
                    for idx in unsat_core:
                        if 0 <= idx - 1 < num_clauses:
                            core[idx - 1] = 1.0
                else:
                    core = fast_approximate_mus(clauses, num_clauses)
            except:
                core = fast_approximate_mus(clauses, num_clauses)
            result['core'] = core
        else:
            result['error'] = 'UNKNOWN'

    except Exception as e:
        result['error'] = str(e)
    finally:
        solver.delete()

    return result


def generate_mixed_vars_dataset_parallel(
    output_dir: str,
    num_train: int = 10000,
    num_val: int = 1000,
    num_test: int = 1000,
    min_vars: int = 40,
    max_vars_gen: int = 100,
    cv_ratio: float = 4.26,
    max_clauses: int = 550,
    max_vars: int = 100,
    k: int = 3,
    seed: int = 42,
    num_workers: int = None,
    balanced: bool = False,
    clause_len_mode: str = 'fixed',
    min_clause_len: int = 2,
    max_clause_len: int = 8,
    clause_len_profile_dir: str = 'neuro_data/train',
):
    """
    生成固定 cv_ratio、变化 var 数量的数据集（train/val/test 三个 split）

    Args:
        output_dir: 输出目录
        num_train/val/test: 各 split 的实例数
        min_vars: 最小变量数
        max_vars_gen: 生成时的最大变量数（不超过 max_vars）
        cv_ratio: 固定的 clause/variable ratio
        max_clauses: VSM padding 维度（子句）
        max_vars: VSM padding 维度（变量）
        k: k-SAT 的 k 值
        seed: 随机种子
        num_workers: 并行 worker 数
        balanced: 是否强制 SAT:UNSAT = 1:1
        clause_len_mode: 子句长度模式 (fixed / uniform / neurosat)
        min_clause_len: uniform 模式的最小子句长度
        max_clause_len: uniform 模式的最大子句长度
        clause_len_profile_dir: neurosat 模式的经验分布目录
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    valid_clause_len_modes = {'fixed', 'uniform', 'neurosat'}
    if clause_len_mode not in valid_clause_len_modes:
        raise ValueError(
            f"Unknown clause_len_mode={clause_len_mode}. "
            f"Expected one of {sorted(valid_clause_len_modes)}."
        )
    if min_clause_len > max_clause_len:
        raise ValueError(
            f"min_clause_len={min_clause_len} must be <= max_clause_len={max_clause_len}."
        )
    if max_clause_len < 2:
        raise ValueError(f"max_clause_len={max_clause_len} must be >= 2.")

    max_vars_gen = min(max_vars_gen, max_vars)
    # 最大可能子句数需要能放进 max_clauses
    max_possible_clauses = int(max_vars_gen * cv_ratio)
    if max_possible_clauses > max_clauses:
        raise ValueError(
            f"max_vars_gen={max_vars_gen} * cv_ratio={cv_ratio} = {max_possible_clauses} "
            f"> max_clauses={max_clauses}. 请增大 max_clauses 或减小 max_vars_gen/cv_ratio。"
        )

    print(f"{'='*60}")
    print(f"Generating MIXED-VARS dataset (fixed cv_ratio):")
    print(f"  num_vars range: [{min_vars}, {max_vars_gen}]")
    print(f"  cv_ratio (fixed): {cv_ratio}")
    print(f"  max_clauses (padding): {max_clauses}")
    print(f"  max_vars (padding): {max_vars}")
    print(f"  clause_len_mode: {clause_len_mode}")
    if clause_len_mode == 'fixed':
        print(f"  k-SAT: {k}")
    elif clause_len_mode == 'uniform':
        print(f"  clause_len_range: [{min_clause_len}, {max_clause_len}]")
    elif clause_len_mode == 'neurosat':
        print(f"  clause_len_profile_dir: {clause_len_profile_dir}")
        print(f"  fallback_k_if_needed: {k}")
    print(f"  Workers: {num_workers}")
    print(f"  Balanced (SAT:UNSAT=1:1): {balanced}")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clause_len_profile = None
    clause_len_values = []
    clause_len_probs = []
    if clause_len_mode == 'neurosat':
        clause_len_profile = build_clause_len_profile(clause_len_profile_dir)
        clause_len_values = list(clause_len_profile.keys())
        clause_len_probs = list(clause_len_profile.values())

    config = {
        'max_clauses': max_clauses,
        'max_vars': max_vars,
        'min_vars': min_vars,
        'max_vars_gen': max_vars_gen,
        'cv_ratio': cv_ratio,
        'k': k,
        'seed': seed,
        'balanced': balanced,
        'clause_len_mode': clause_len_mode,
        'min_clause_len': min_clause_len,
        'max_clause_len': max_clause_len,
        'clause_len_profile_dir': clause_len_profile_dir,
        'clause_len_profile': clause_len_profile,
        'clause_len_values': clause_len_values,
        'clause_len_probs': clause_len_probs,
    }

    split_offset = {'train': 0, 'val': 100000, 'test': 200000}

    for split, num_instances in [('train', num_train), ('val', num_val), ('test', num_test)]:
        print(f"\n{'='*60}")
        print(f"Generating {split} set: {num_instances} instances")
        print(f"{'='*60}")

        start_time = time.time()

        if balanced:
            # 平衡模式：分别收集 SAT 和 UNSAT 实例，直到各凑齐 num_instances/2 个
            need_sat = num_instances // 2
            need_unsat = num_instances - need_sat  # 处理奇数情况
            sat_results = []
            unsat_results = []
            batch_id = 0
            total_generated = 0

            print(f"  Balanced mode: need {need_sat} SAT + {need_unsat} UNSAT")

            while len(sat_results) < need_sat or len(unsat_results) < need_unsat:
                # 每批生成一定数量的实例
                still_need = (need_sat - len(sat_results)) + (need_unsat - len(unsat_results))
                # 多生成一些以弥补比例偏差（至少多 20%）
                batch_size = max(still_need, int(still_need * 1.2))
                batch_size = min(batch_size, still_need * 3)  # 防止过度生成

                args_list = []
                for i in range(batch_size):
                    cfg = config.copy()
                    cfg['seed'] = seed + split_offset[split] + total_generated + i
                    args_list.append((total_generated + i, cfg))

                batch_results = []
                with Pool(num_workers) as pool:
                    for result in tqdm(
                        pool.imap_unordered(solve_single_instance_mixed_vars, args_list),
                        total=batch_size,
                        desc=f"Generating {split} batch {batch_id} "
                             f"(SAT: {len(sat_results)}/{need_sat}, "
                             f"UNSAT: {len(unsat_results)}/{need_unsat})"
                    ):
                        batch_results.append(result)

                total_generated += batch_size
                batch_id += 1

                for result in batch_results:
                    if result.get('error'):
                        continue
                    if result['sat_label'] == 1 and len(sat_results) < need_sat:
                        sat_results.append(result)
                    elif result['sat_label'] == 0 and len(unsat_results) < need_unsat:
                        unsat_results.append(result)

                print(f"  After batch {batch_id}: SAT={len(sat_results)}/{need_sat}, "
                      f"UNSAT={len(unsat_results)}/{need_unsat}, "
                      f"total generated={total_generated}")

            # 合并并打乱
            selected = sat_results[:need_sat] + unsat_results[:need_unsat]
            random.seed(seed + split_offset[split])
            random.shuffle(selected)
            # 重新赋 instance_id
            for idx, r in enumerate(selected):
                r['instance_id'] = idx
            results = selected

            print(f"  Total instances generated to achieve balance: {total_generated}")
        else:
            # 原始模式：直接生成
            args_list = []
            for i in range(num_instances):
                cfg = config.copy()
                cfg['seed'] = seed + split_offset[split] + i
                args_list.append((i, cfg))

            results = []
            with Pool(num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(solve_single_instance_mixed_vars, args_list),
                    total=num_instances,
                    desc=f"Generating {split}"
                ):
                    results.append(result)

            results.sort(key=lambda x: x['instance_id'])

        all_vsm = np.zeros((num_instances, 2, max_clauses, max_vars), dtype=np.float32)
        all_clause_mask = np.zeros((num_instances, max_clauses), dtype=np.bool_)
        all_var_mask = np.zeros((num_instances, max_vars), dtype=np.bool_)
        all_sat_label = np.zeros(num_instances, dtype=np.int64)
        all_core_labels = np.zeros((num_instances, max_clauses), dtype=np.float32)
        all_vsids_labels = np.zeros((num_instances, max_vars), dtype=np.float32)
        metadata = []

        sat_count = 0
        unsat_count = 0
        error_count = 0

        for i, result in enumerate(results):
            if result.get('error'):
                error_count += 1
                metadata.append({
                    'num_vars': result.get('num_vars', 0),
                    'num_clauses': result.get('num_clauses', 0),
                    'min_clause_len': result.get('min_clause_len', 0),
                    'max_clause_len': result.get('max_clause_len', 0),
                    'avg_clause_len': result.get('avg_clause_len', 0.0),
                    'error': result['error']
                })
                continue

            all_vsm[i] = result['vsm']
            all_clause_mask[i] = result['clause_mask']
            all_var_mask[i] = result['var_mask']
            all_sat_label[i] = result['sat_label']

            core = result['core']
            core_len = min(len(core), max_clauses)
            all_core_labels[i, :core_len] = core[:core_len]

            vsids = result['vsids']
            vsids_len = min(len(vsids), max_vars)
            all_vsids_labels[i, :vsids_len] = vsids[:vsids_len]

            if result['sat_label'] == 1:
                sat_count += 1
            else:
                unsat_count += 1

            metadata.append({
                'num_vars': result['num_vars'],
                'num_clauses': result['num_clauses'],
                'min_clause_len': result['min_clause_len'],
                'max_clause_len': result['max_clause_len'],
                'avg_clause_len': result['avg_clause_len'],
            })

        data = {
            'vsm': torch.from_numpy(all_vsm),
            'clause_mask': torch.from_numpy(all_clause_mask),
            'var_mask': torch.from_numpy(all_var_mask),
            'sat_label': torch.from_numpy(all_sat_label),
            'core_labels': torch.from_numpy(all_core_labels),
            'vsids_labels': torch.from_numpy(all_vsids_labels),
            'metadata': metadata,
            'config': {**config, 'split': split}
        }

        save_path = output_path / f'{split}.pt'
        torch.save(data, save_path)

        elapsed = time.time() - start_time
        file_size = save_path.stat().st_size / (1024 * 1024)

        print(f"\n{split} set statistics:")
        print(f"  SAT instances: {sat_count} ({100*sat_count/num_instances:.1f}%)")
        print(f"  UNSAT instances: {unsat_count} ({100*unsat_count/num_instances:.1f}%)")
        print(f"  Errors: {error_count}")
        print(f"  Time: {elapsed:.1f}s ({num_instances/elapsed:.1f} instances/s)")
        print(f"  Saved to: {save_path}")
        print(f"  File size: {file_size:.2f} MB")

    print(f"\n{'='*60}")
    print("Dataset generation completed!")
    print(f"{'='*60}")


def generate_fixed_dataset_parallel(
    output_dir: str,
    num_instances: int = 1000,
    fixed_vars: int = 100,
    fixed_clauses: int = None,
    cv_ratio: float = 4.26,
    max_clauses: int = 500,
    max_vars: int = 250,
    k: int = 3,
    seed: int = 42,
    num_workers: int = None,
    output_name: str = 'test'
):
    """
    生成固定 var/clause 数量的数据集

    Args:
        output_dir: 输出目录
        num_instances: 实例数量
        fixed_vars: 固定的变量数
        fixed_clauses: 固定的子句数（如果为 None，则根据 cv_ratio 计算）
        cv_ratio: clause/variable ratio（当 fixed_clauses 为 None 时使用）
        max_clauses: VSM 最大子句数（用于 padding）
        max_vars: VSM 最大变量数（用于 padding）
        k: k-SAT 的 k 值
        seed: 随机种子
        num_workers: 并行 worker 数
        output_name: 输出文件名（不含扩展名）
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # 如果没有指定 fixed_clauses，则根据 cv_ratio 计算
    if fixed_clauses is None:
        fixed_clauses = int(fixed_vars * cv_ratio)

    # 检查是否超出 max 限制
    if fixed_vars > max_vars:
        print(f"Warning: fixed_vars ({fixed_vars}) > max_vars ({max_vars}), adjusting max_vars")
        max_vars = fixed_vars + 10
    if fixed_clauses > max_clauses:
        print(f"Warning: fixed_clauses ({fixed_clauses}) > max_clauses ({max_clauses}), adjusting max_clauses")
        max_clauses = fixed_clauses + 10

    actual_cv_ratio = fixed_clauses / fixed_vars

    print(f"{'='*60}")
    print(f"Generating FIXED dataset:")
    print(f"  Instances: {num_instances}")
    print(f"  Fixed vars: {fixed_vars}")
    print(f"  Fixed clauses: {fixed_clauses}")
    print(f"  Actual CV ratio: {actual_cv_ratio:.4f}")
    print(f"  k-SAT: {k}")
    print(f"  Max vars (padding): {max_vars}")
    print(f"  Max clauses (padding): {max_clauses}")
    print(f"  Workers: {num_workers}")
    print(f"{'='*60}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = {
        'max_clauses': max_clauses,
        'max_vars': max_vars,
        'k': k,
        'fixed_vars': fixed_vars,
        'fixed_clauses': fixed_clauses,
        'cv_ratio': actual_cv_ratio,
        'seed': seed
    }

    start_time = time.time()

    # 准备参数
    args_list = [(i, config.copy()) for i in range(num_instances)]
    for i in range(len(args_list)):
        args_list[i][1]['seed'] = seed + i

    # 并行处理
    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(solve_single_instance_fixed, args_list),
            total=num_instances,
            desc="Generating fixed dataset"
        ):
            results.append(result)

    # 按 instance_id 排序
    results.sort(key=lambda x: x['instance_id'])

    # 收集结果
    all_vsm = np.zeros((num_instances, 2, max_clauses, max_vars), dtype=np.float32)
    all_clause_mask = np.zeros((num_instances, max_clauses), dtype=np.bool_)
    all_var_mask = np.zeros((num_instances, max_vars), dtype=np.bool_)
    all_sat_label = np.zeros(num_instances, dtype=np.int64)
    all_core_labels = np.zeros((num_instances, max_clauses), dtype=np.float32)
    all_vsids_labels = np.zeros((num_instances, max_vars), dtype=np.float32)
    metadata = []

    sat_count = 0
    unsat_count = 0
    error_count = 0

    for i, result in enumerate(results):
        if result.get('error'):
            error_count += 1
            metadata.append({
                'num_vars': result.get('num_vars', 0),
                'num_clauses': result.get('num_clauses', 0),
                'error': result['error']
            })
            continue

        all_vsm[i] = result['vsm']
        all_clause_mask[i] = result['clause_mask']
        all_var_mask[i] = result['var_mask']
        all_sat_label[i] = result['sat_label']

        # Core labels
        core = result['core']
        core_len = min(len(core), max_clauses)
        all_core_labels[i, :core_len] = core[:core_len]

        # VSIDS labels
        vsids = result['vsids']
        vsids_len = min(len(vsids), max_vars)
        all_vsids_labels[i, :vsids_len] = vsids[:vsids_len]

        if result['sat_label'] == 1:
            sat_count += 1
        else:
            unsat_count += 1

        metadata.append({
            'num_vars': result['num_vars'],
            'num_clauses': result['num_clauses']
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
        'config': config
    }

    save_path = output_path / f'{output_name}.pt'
    torch.save(data, save_path)

    elapsed = time.time() - start_time
    file_size = save_path.stat().st_size / (1024 * 1024)

    print(f"\nDataset statistics:")
    print(f"  SAT instances: {sat_count} ({100*sat_count/num_instances:.1f}%)")
    print(f"  UNSAT instances: {unsat_count} ({100*unsat_count/num_instances:.1f}%)")
    print(f"  Errors: {error_count}")
    print(f"  Time: {elapsed:.1f}s ({num_instances/elapsed:.1f} instances/s)")
    print(f"  Saved to: {save_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast parallel SAT dataset generation')

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
    gen_parser.add_argument('--max_clauses', type=int, default=426,
                           help='Max clauses in VSM')
    gen_parser.add_argument('--max_vars', type=int, default=100,
                           help='Max variables in VSM')
    gen_parser.add_argument('--k', type=int, default=3,
                           help='k for k-SAT')
    gen_parser.add_argument('--min_vars', type=int, default=100,
                           help='Min variables when generating')
    gen_parser.add_argument('--gen_max_vars', type=int, default=426,
                           help='Max variables when generating')
    gen_parser.add_argument('--cv_ratio', type=float, default=4.26,
                           help='Clause/variable ratio')
    gen_parser.add_argument('--sat_ratio', type=float, default=0.5,
                           help='Ratio of random SAT instances')
    gen_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    gen_parser.add_argument('--num_workers', type=int, default=None,
                           help='Number of parallel workers (default: CPU count - 1)')

    # 生成固定 cv_ratio、变化 var 数量的数据集
    mixed_vars_parser = subparsers.add_parser(
        'generate-mixed-vars',
        help='Generate dataset with fixed cv_ratio and variable num_vars in [min_vars, max_vars_gen]'
    )
    mixed_vars_parser.add_argument('--output_dir', type=str, default='data_mixed_vars',
                                   help='Output directory')
    mixed_vars_parser.add_argument('--num_train', type=int, default=10000,
                                   help='Number of training instances')
    mixed_vars_parser.add_argument('--num_val', type=int, default=1000,
                                   help='Number of validation instances')
    mixed_vars_parser.add_argument('--num_test', type=int, default=1000,
                                   help='Number of test instances')
    mixed_vars_parser.add_argument('--min_vars', type=int, default=40,
                                   help='Minimum number of variables')
    mixed_vars_parser.add_argument('--max_vars_gen', type=int, default=100,
                                   help='Maximum number of variables when generating')
    mixed_vars_parser.add_argument('--cv_ratio', type=float, default=4.26,
                                   help='Fixed clause/variable ratio')
    mixed_vars_parser.add_argument('--max_clauses', type=int, default=550,
                                   help='Max clauses in VSM (for padding)')
    mixed_vars_parser.add_argument('--max_vars', type=int, default=100,
                                   help='Max variables in VSM (for padding), must be >= max_vars_gen')
    mixed_vars_parser.add_argument('--k', type=int, default=3,
                                   help='k for k-SAT')
    mixed_vars_parser.add_argument('--clause_len_mode', type=str, default='fixed',
                                   choices=['fixed', 'uniform', 'neurosat'],
                                   help='Clause-length sampling mode')
    mixed_vars_parser.add_argument('--min_clause_len', type=int, default=2,
                                   help='Minimum clause length for uniform mode')
    mixed_vars_parser.add_argument('--max_clause_len', type=int, default=8,
                                   help='Maximum clause length for uniform mode')
    mixed_vars_parser.add_argument('--clause_len_profile_dir', type=str,
                                   default='neuro_data/train',
                                   help='CNF directory used to build clause-length profile for neurosat mode')
    mixed_vars_parser.add_argument('--seed', type=int, default=42,
                                   help='Random seed')
    mixed_vars_parser.add_argument('--num_workers', type=int, default=None,
                                   help='Number of parallel workers (default: CPU count - 1)')
    mixed_vars_parser.add_argument('--balanced', action='store_true', default=False,
                                   help='Force SAT:UNSAT = 1:1 balanced dataset')

    # 生成固定参数数据集
    fixed_parser = subparsers.add_parser('generate-fixed', help='Generate dataset with fixed var/clause count')
    fixed_parser.add_argument('--output_dir', type=str, default='data_fixed',
                              help='Output directory')
    fixed_parser.add_argument('--num_instances', type=int, default=1000,
                              help='Number of instances to generate')
    fixed_parser.add_argument('--fixed_vars', type=int, required=True,
                              help='Fixed number of variables')
    fixed_parser.add_argument('--fixed_clauses', type=int, default=None,
                              help='Fixed number of clauses (if not set, calculated from cv_ratio)')
    fixed_parser.add_argument('--cv_ratio', type=float, default=4.26,
                              help='Clause/variable ratio (used when fixed_clauses is not set)')
    fixed_parser.add_argument('--max_clauses', type=int, default=500,
                              help='Max clauses in VSM (for padding)')
    fixed_parser.add_argument('--max_vars', type=int, default=250,
                              help='Max variables in VSM (for padding)')
    fixed_parser.add_argument('--k', type=int, default=3,
                              help='k for k-SAT')
    fixed_parser.add_argument('--seed', type=int, default=1022,
                              help='Random seed')
    fixed_parser.add_argument('--num_workers', type=int, default=None,
                              help='Number of parallel workers')
    fixed_parser.add_argument('--output_name', type=str, default='test',
                              help='Output filename (without extension)')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_packed_dataset_parallel(
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
            seed=args.seed,
            num_workers=args.num_workers
        )
    elif args.command == 'generate-mixed-vars':
        generate_mixed_vars_dataset_parallel(
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val,
            num_test=args.num_test,
            min_vars=args.min_vars,
            max_vars_gen=args.max_vars_gen,
            cv_ratio=args.cv_ratio,
            max_clauses=args.max_clauses,
            max_vars=args.max_vars,
            k=args.k,
            seed=args.seed,
            num_workers=args.num_workers,
            balanced=args.balanced,
            clause_len_mode=args.clause_len_mode,
            min_clause_len=args.min_clause_len,
            max_clause_len=args.max_clause_len,
            clause_len_profile_dir=args.clause_len_profile_dir,
        )
    elif args.command == 'generate-fixed':
        generate_fixed_dataset_parallel(
            output_dir=args.output_dir,
            num_instances=args.num_instances,
            fixed_vars=args.fixed_vars,
            fixed_clauses=args.fixed_clauses,
            cv_ratio=args.cv_ratio,
            max_clauses=args.max_clauses,
            max_vars=args.max_vars,
            k=args.k,
            seed=args.seed,
            num_workers=args.num_workers,
            output_name=args.output_name
        )
    else:
        parser.print_help()

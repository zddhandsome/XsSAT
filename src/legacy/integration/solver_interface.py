"""
GeoSATformer Solver Interface Module
SAT求解器接口模块

提供统一的SAT求解器接口，支持：
- VSIDS初始化
- 求解超时控制
- 结果解析
- 与GeoSATformer模型集成
"""

import os
import time
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverResult:
    """求解结果数据类"""
    satisfiable: Optional[bool] = None  # True=SAT, False=UNSAT, None=UNKNOWN
    assignment: Optional[List[int]] = None  # 满足赋值（SAT时）
    solve_time: float = 0.0  # 求解时间（秒）
    num_decisions: int = 0  # 决策数
    num_propagations: int = 0  # 传播数
    num_conflicts: int = 0  # 冲突数
    timed_out: bool = False  # 是否超时
    error: Optional[str] = None  # 错误信息
    
    @property
    def solved(self) -> bool:
        """是否成功求解（非超时且无错误）"""
        return self.satisfiable is not None and not self.timed_out and self.error is None


class SolverInterface(ABC):
    """
    SAT求解器抽象基类
    
    定义统一的求解器接口
    """
    
    def __init__(
        self,
        solver_path: Optional[str] = None,
        timeout: float = 300.0,
        verbose: bool = False
    ):
        """
        Args:
            solver_path: 求解器可执行文件路径
            timeout: 默认超时时间（秒）
            verbose: 是否输出详细信息
        """
        self.solver_path = solver_path
        self.timeout = timeout
        self.verbose = verbose
    
    @abstractmethod
    def solve(
        self,
        cnf_path: str,
        timeout: Optional[float] = None,
        use_vsids_init: bool = False,
        vsids_scores: Optional[np.ndarray] = None,
        **kwargs
    ) -> SolverResult:
        """
        求解CNF实例
        
        Args:
            cnf_path: CNF文件路径
            timeout: 超时时间（秒）
            use_vsids_init: 是否使用VSIDS初始化
            vsids_scores: VSIDS初始分数
            
        Returns:
            求解结果
        """
        pass
    
    @abstractmethod
    def set_initial_activity(self, var: int, score: float):
        """设置变量的初始活动分数"""
        pass
    
    def parse_cnf(self, cnf_path: str) -> Tuple[int, int, List[List[int]]]:
        """
        解析CNF文件
        
        Args:
            cnf_path: CNF文件路径
            
        Returns:
            (变量数, 子句数, 子句列表)
        """
        clauses = []
        num_vars = 0
        num_clauses = 0
        
        with open(cnf_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p cnf'):
                    parts = line.split()
                    num_vars = int(parts[2])
                    num_clauses = int(parts[3])
                else:
                    literals = [int(x) for x in line.split() if x != '0']
                    if literals:
                        clauses.append(literals)
        
        return num_vars, num_clauses, clauses
    
    def cnf_to_vsm(self, cnf_path: str) -> np.ndarray:
        """
        将CNF文件转换为Variable Space Matrix
        
        Args:
            cnf_path: CNF文件路径
            
        Returns:
            VSM矩阵 [num_clauses, num_vars]
        """
        num_vars, num_clauses, clauses = self.parse_cnf(cnf_path)
        
        vsm = np.zeros((len(clauses), num_vars), dtype=np.float32)
        
        for i, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1  # 变量索引从0开始
                vsm[i, var] = 1.0 if lit > 0 else -1.0
        
        return vsm


class SolverIntegration:
    """
    GeoSATformer与SAT求解器的集成类
    
    提供完整的工作流：
    1. 模型推理获取VSIDS分数
    2. 初始化求解器
    3. 执行求解
    4. 收集结果
    """
    
    def __init__(
        self,
        model: nn.Module,
        solver: SolverInterface,
        device: torch.device = None,
        use_amp: bool = True
    ):
        """
        Args:
            model: GeoSATformer模型
            solver: SAT求解器接口
            device: 计算设备
            use_amp: 是否使用混合精度
        """
        self.model = model
        self.solver = solver
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        cnf_path: str,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        对CNF实例进行预测
        
        Args:
            cnf_path: CNF文件路径
            return_embeddings: 是否返回中间嵌入
            
        Returns:
            预测结果字典
        """
        # 转换为VSM
        vsm = self.solver.cnf_to_vsm(cnf_path)
        vsm_tensor = torch.from_numpy(vsm).unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(vsm_tensor)
            else:
                outputs = self.model(vsm_tensor)
        
        # 提取预测结果
        sat_prob = torch.sigmoid(outputs['sat_pred'][0]).item()
        vsids_scores = outputs['vsids_scores'][0].cpu().numpy()
        
        result = {
            'sat_probability': sat_prob,
            'predicted_sat': sat_prob > 0.5,
            'vsids_scores': vsids_scores
        }
        
        # UNSAT Core预测
        if 'clause_scores' in outputs:
            clause_probs = torch.sigmoid(outputs['clause_scores'][0]).cpu().numpy()
            result['clause_muc_probabilities'] = clause_probs
            result['predicted_muc'] = clause_probs > 0.5
        
        # 中间嵌入
        if return_embeddings:
            if 'clause_embeddings' in outputs:
                result['clause_embeddings'] = outputs['clause_embeddings'][0].cpu().numpy()
            if 'var_embeddings' in outputs:
                result['var_embeddings'] = outputs['var_embeddings'][0].cpu().numpy()
        
        return result
    
    def solve_with_vsids_init(
        self,
        cnf_path: str,
        timeout: Optional[float] = None,
        fallback_on_timeout: bool = True
    ) -> Dict[str, Any]:
        """
        使用VSIDS初始化求解CNF实例
        
        实现算法：
        1. vsm = CNF_to_VSM(φ)
        2. outputs = GeoSATformer(vsm)
        3. 初始化求解器的VSIDS分数
        4. 执行求解
        
        Args:
            cnf_path: CNF文件路径
            timeout: 超时时间
            fallback_on_timeout: 超时后是否回退到无初始化求解
            
        Returns:
            求解结果字典
        """
        start_time = time.time()
        
        # 1. 模型预测
        predict_start = time.time()
        predictions = self.predict(cnf_path)
        predict_time = time.time() - predict_start
        
        # 2. 使用VSIDS初始化求解
        solve_start = time.time()
        result = self.solver.solve(
            cnf_path,
            timeout=timeout,
            use_vsids_init=True,
            vsids_scores=predictions['vsids_scores']
        )
        solve_time = time.time() - solve_start
        
        total_time = time.time() - start_time
        
        # 3. 整合结果
        output = {
            'satisfiable': result.satisfiable,
            'assignment': result.assignment,
            'solve_time': solve_time,
            'predict_time': predict_time,
            'total_time': total_time,
            'timed_out': result.timed_out,
            'num_decisions': result.num_decisions,
            'num_conflicts': result.num_conflicts,
            'predictions': predictions
        }
        
        # 4. 超时回退
        if result.timed_out and fallback_on_timeout:
            fallback_result = self.solver.solve(
                cnf_path,
                timeout=timeout,
                use_vsids_init=False
            )
            output['fallback_result'] = {
                'satisfiable': fallback_result.satisfiable,
                'solve_time': fallback_result.solve_time,
                'timed_out': fallback_result.timed_out
            }
        
        return output
    
    def batch_solve(
        self,
        cnf_paths: List[str],
        timeout: Optional[float] = None,
        compare_baseline: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        批量求解多个CNF实例
        
        Args:
            cnf_paths: CNF文件路径列表
            timeout: 每个实例的超时时间
            compare_baseline: 是否与无初始化基线比较
            progress_callback: 进度回调函数
            
        Returns:
            批量求解结果
        """
        enhanced_results = []
        baseline_results = []
        
        for i, cnf_path in enumerate(cnf_paths):
            # 增强求解
            enhanced = self.solve_with_vsids_init(cnf_path, timeout)
            enhanced_results.append(enhanced)
            
            # 基线求解
            if compare_baseline:
                baseline = self.solver.solve(
                    cnf_path,
                    timeout=timeout,
                    use_vsids_init=False
                )
                baseline_results.append({
                    'satisfiable': baseline.satisfiable,
                    'solve_time': baseline.solve_time,
                    'timed_out': baseline.timed_out,
                    'num_decisions': baseline.num_decisions,
                    'num_conflicts': baseline.num_conflicts
                })
            
            if progress_callback:
                progress_callback(i + 1, len(cnf_paths))
        
        # 统计
        stats = self._compute_batch_stats(enhanced_results, baseline_results)
        
        return {
            'enhanced_results': enhanced_results,
            'baseline_results': baseline_results if compare_baseline else None,
            'statistics': stats
        }
    
    def _compute_batch_stats(
        self,
        enhanced: List[Dict],
        baseline: List[Dict]
    ) -> Dict[str, float]:
        """计算批量求解统计信息"""
        n = len(enhanced)
        
        enhanced_times = [r['solve_time'] for r in enhanced]
        enhanced_solved = sum(1 for r in enhanced if not r['timed_out'])
        
        stats = {
            'num_instances': n,
            'enhanced_solved': enhanced_solved,
            'enhanced_solve_rate': enhanced_solved / n,
            'enhanced_avg_time': np.mean(enhanced_times),
            'enhanced_median_time': np.median(enhanced_times)
        }
        
        if baseline:
            baseline_times = [r['solve_time'] for r in baseline]
            baseline_solved = sum(1 for r in baseline if not r['timed_out'])
            
            stats.update({
                'baseline_solved': baseline_solved,
                'baseline_solve_rate': baseline_solved / n,
                'baseline_avg_time': np.mean(baseline_times),
                'baseline_median_time': np.median(baseline_times),
                'speedup_ratio': (np.mean(baseline_times) - np.mean(enhanced_times)) / np.mean(baseline_times) if np.mean(baseline_times) > 0 else 0
            })
        
        return stats


class PythonSATSolver(SolverInterface):
    """
    基于python-sat库的求解器封装
    
    支持多种后端：glucose3, glucose4, lingeling, minisat22等
    """
    
    def __init__(
        self,
        backend: str = 'glucose4',
        timeout: float = 300.0,
        verbose: bool = False
    ):
        super().__init__(timeout=timeout, verbose=verbose)
        self.backend = backend
        self._solver = None
        self._initial_activities = {}
        
        try:
            from pysat.solvers import Solver
            self.Solver = Solver
        except ImportError:
            raise ImportError("python-sat not installed. Run: pip install python-sat")
    
    def solve(
        self,
        cnf_path: str,
        timeout: Optional[float] = None,
        use_vsids_init: bool = False,
        vsids_scores: Optional[np.ndarray] = None,
        **kwargs
    ) -> SolverResult:
        """使用python-sat求解"""
        timeout = timeout or self.timeout
        
        # 解析CNF
        num_vars, num_clauses, clauses = self.parse_cnf(cnf_path)
        
        # 创建求解器
        solver = self.Solver(name=self.backend)
        
        # 添加子句
        for clause in clauses:
            solver.add_clause(clause)
        
        # VSIDS初始化（python-sat不直接支持，这里记录以供子类扩展）
        if use_vsids_init and vsids_scores is not None:
            self._initial_activities = {i + 1: vsids_scores[i] for i in range(len(vsids_scores))}
        
        # 求解
        start_time = time.time()
        
        try:
            if timeout > 0:
                result = solver.solve_limited(expect_interrupt=True)
                # 检查是否超时需要额外处理
            else:
                result = solver.solve()
            
            solve_time = time.time() - start_time
            timed_out = solve_time >= timeout
            
            if result:
                assignment = solver.get_model()
                satisfiable = True
            elif result is False:
                assignment = None
                satisfiable = False
            else:
                assignment = None
                satisfiable = None
            
            return SolverResult(
                satisfiable=satisfiable,
                assignment=assignment,
                solve_time=solve_time,
                timed_out=timed_out
            )
            
        except Exception as e:
            return SolverResult(
                error=str(e),
                solve_time=time.time() - start_time
            )
        finally:
            solver.delete()
    
    def set_initial_activity(self, var: int, score: float):
        """设置初始活动分数（供子类实现）"""
        self._initial_activities[var] = score

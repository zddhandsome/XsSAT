"""
GeoSATformer CaDiCaL Wrapper
CaDiCaL求解器封装

CaDiCaL是一个现代的、高性能的SAT求解器，支持：
- VSIDS初始化
- 增量求解
- 学习子句提取
"""

import os
import time
import subprocess
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .solver_interface import SolverInterface, SolverResult


class CaDiCaLWrapper(SolverInterface):
    """
    CaDiCaL SAT求解器封装
    
    支持两种模式：
    1. 通过pycadical Python绑定
    2. 通过命令行调用
    """
    
    def __init__(
        self,
        solver_path: Optional[str] = None,
        use_python_binding: bool = True,
        timeout: float = 300.0,
        verbose: bool = False
    ):
        """
        Args:
            solver_path: CaDiCaL可执行文件路径（命令行模式）
            use_python_binding: 是否使用Python绑定
            timeout: 默认超时时间
            verbose: 是否输出详细信息
        """
        super().__init__(solver_path, timeout, verbose)
        self.use_python_binding = use_python_binding
        
        # 尝试导入pycadical
        self._cadical = None
        if use_python_binding:
            try:
                import pycadical
                self._has_binding = True
            except ImportError:
                self._has_binding = False
                if self.verbose:
                    print("Warning: pycadical not found, falling back to command line mode")
        else:
            self._has_binding = False
        
        # 检查命令行求解器
        if not self._has_binding:
            if solver_path is None:
                # 尝试从PATH中找到cadical
                import shutil
                self.solver_path = shutil.which('cadical')
                if self.solver_path is None:
                    raise RuntimeError(
                        "CaDiCaL not found. Install pycadical or provide solver_path."
                    )
            else:
                if not os.path.exists(solver_path):
                    raise FileNotFoundError(f"CaDiCaL not found at {solver_path}")
                self.solver_path = solver_path
    
    def solve(
        self,
        cnf_path: str,
        timeout: Optional[float] = None,
        use_vsids_init: bool = False,
        vsids_scores: Optional[np.ndarray] = None,
        **kwargs
    ) -> SolverResult:
        """
        使用CaDiCaL求解CNF实例
        
        Args:
            cnf_path: CNF文件路径
            timeout: 超时时间（秒）
            use_vsids_init: 是否使用VSIDS初始化
            vsids_scores: VSIDS初始分数数组
            
        Returns:
            求解结果
        """
        timeout = timeout or self.timeout
        
        if self._has_binding:
            return self._solve_with_binding(
                cnf_path, timeout, use_vsids_init, vsids_scores
            )
        else:
            return self._solve_with_cli(
                cnf_path, timeout, use_vsids_init, vsids_scores
            )
    
    def _solve_with_binding(
        self,
        cnf_path: str,
        timeout: float,
        use_vsids_init: bool,
        vsids_scores: Optional[np.ndarray]
    ) -> SolverResult:
        """使用Python绑定求解"""
        import pycadical
        
        # 解析CNF
        num_vars, num_clauses, clauses = self.parse_cnf(cnf_path)
        
        # 创建求解器实例
        solver = pycadical.Solver()
        
        # 设置VSIDS初始分数
        if use_vsids_init and vsids_scores is not None:
            for var_idx, score in enumerate(vsids_scores):
                if var_idx < num_vars:
                    self.set_initial_activity_cadical(solver, var_idx + 1, score)
        
        # 添加子句
        for clause in clauses:
            for lit in clause:
                solver.add(lit)
            solver.add(0)  # 结束子句
        
        # 求解
        start_time = time.time()
        
        # 设置超时（通过terminate信号）
        # 注意：pycadical可能需要特殊处理超时
        try:
            result = solver.solve()
            solve_time = time.time() - start_time
            timed_out = solve_time >= timeout
            
            if result == 10:  # SAT
                assignment = [solver.val(i) for i in range(1, num_vars + 1)]
                satisfiable = True
            elif result == 20:  # UNSAT
                assignment = None
                satisfiable = False
            else:  # UNKNOWN
                assignment = None
                satisfiable = None
            
            # 获取统计信息
            stats = self._get_stats_binding(solver)
            
            return SolverResult(
                satisfiable=satisfiable,
                assignment=assignment,
                solve_time=solve_time,
                timed_out=timed_out,
                **stats
            )
            
        except Exception as e:
            return SolverResult(
                error=str(e),
                solve_time=time.time() - start_time
            )
    
    def _solve_with_cli(
        self,
        cnf_path: str,
        timeout: float,
        use_vsids_init: bool,
        vsids_scores: Optional[np.ndarray]
    ) -> SolverResult:
        """使用命令行求解"""
        # 如果需要VSIDS初始化，创建带有phase saving hints的临时文件
        if use_vsids_init and vsids_scores is not None:
            cnf_path = self._create_enhanced_cnf(cnf_path, vsids_scores)
        
        # 构建命令
        cmd = [self.solver_path]
        
        # 添加选项
        if timeout > 0:
            cmd.extend(['--time', str(int(timeout))])
        
        # 输出统计信息
        cmd.append('-s')
        
        # 输入文件
        cmd.append(cnf_path)
        
        # 执行
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10  # 额外缓冲
            )
            
            solve_time = time.time() - start_time
            
            # 解析输出
            return self._parse_cli_output(result, solve_time, timeout)
            
        except subprocess.TimeoutExpired:
            return SolverResult(
                satisfiable=None,
                solve_time=timeout,
                timed_out=True
            )
        except Exception as e:
            return SolverResult(
                error=str(e),
                solve_time=time.time() - start_time
            )
    
    def _create_enhanced_cnf(
        self,
        cnf_path: str,
        vsids_scores: np.ndarray
    ) -> str:
        """创建带有VSIDS hints的增强CNF文件"""
        # 创建临时文件
        fd, enhanced_path = tempfile.mkstemp(suffix='.cnf')
        
        try:
            with os.fdopen(fd, 'w') as f_out:
                with open(cnf_path, 'r') as f_in:
                    for line in f_in:
                        f_out.write(line)
                
                # 添加phase saving hints（按VSIDS分数排序的变量顺序）
                # 注意：这是一种近似的VSIDS初始化方式
                sorted_vars = np.argsort(-vsids_scores)  # 降序
                
                f_out.write('c VSIDS initialization hints\n')
                for rank, var_idx in enumerate(sorted_vars):
                    # 使用注释行存储hints（某些求解器可以读取）
                    f_out.write(f'c vsids {var_idx + 1} {vsids_scores[var_idx]:.6f}\n')
            
            return enhanced_path
            
        except Exception:
            os.unlink(enhanced_path)
            raise
    
    def _parse_cli_output(
        self,
        result: subprocess.CompletedProcess,
        solve_time: float,
        timeout: float
    ) -> SolverResult:
        """解析命令行输出"""
        stdout = result.stdout
        returncode = result.returncode
        
        # CaDiCaL返回码：10=SAT, 20=UNSAT, 0=UNKNOWN
        if returncode == 10:
            satisfiable = True
            assignment = self._parse_assignment(stdout)
        elif returncode == 20:
            satisfiable = False
            assignment = None
        else:
            satisfiable = None
            assignment = None
        
        # 解析统计信息
        stats = self._parse_stats(stdout)
        
        return SolverResult(
            satisfiable=satisfiable,
            assignment=assignment,
            solve_time=solve_time,
            timed_out=solve_time >= timeout,
            **stats
        )
    
    def _parse_assignment(self, output: str) -> List[int]:
        """从输出中解析满足赋值"""
        assignment = []
        
        for line in output.split('\n'):
            if line.startswith('v '):
                literals = line[2:].split()
                for lit in literals:
                    lit = int(lit)
                    if lit != 0:
                        assignment.append(lit)
        
        return assignment if assignment else None
    
    def _parse_stats(self, output: str) -> Dict[str, int]:
        """从输出中解析统计信息"""
        stats = {
            'num_decisions': 0,
            'num_propagations': 0,
            'num_conflicts': 0
        }
        
        for line in output.split('\n'):
            line = line.strip()
            
            if 'decisions:' in line.lower():
                try:
                    stats['num_decisions'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            
            if 'propagations:' in line.lower():
                try:
                    stats['num_propagations'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            
            if 'conflicts:' in line.lower():
                try:
                    stats['num_conflicts'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        
        return stats
    
    def _get_stats_binding(self, solver) -> Dict[str, int]:
        """从Python绑定获取统计信息"""
        # pycadical的统计接口可能因版本而异
        stats = {
            'num_decisions': 0,
            'num_propagations': 0,
            'num_conflicts': 0
        }
        
        try:
            if hasattr(solver, 'decisions'):
                stats['num_decisions'] = solver.decisions()
            if hasattr(solver, 'propagations'):
                stats['num_propagations'] = solver.propagations()
            if hasattr(solver, 'conflicts'):
                stats['num_conflicts'] = solver.conflicts()
        except Exception:
            pass
        
        return stats
    
    def set_initial_activity(self, var: int, score: float):
        """设置变量的初始活动分数（接口方法）"""
        pass  # 在实际求解时使用
    
    def set_initial_activity_cadical(self, solver, var: int, score: float):
        """
        设置CaDiCaL求解器中变量的初始活动分数
        
        注意：CaDiCaL可能不直接支持此功能，使用phase saving作为替代
        """
        try:
            # 尝试使用phase方法设置初始相位
            if score > 0.5:
                solver.phase(var)  # 倾向于True
            else:
                solver.phase(-var)  # 倾向于False
        except AttributeError:
            pass  # 版本不支持
    
    def get_learned_clauses(self, solver) -> List[List[int]]:
        """
        获取学习到的子句（如果支持）
        
        Returns:
            学习子句列表
        """
        learned = []
        
        try:
            # pycadical可能提供的接口
            if hasattr(solver, 'learned_clauses'):
                learned = solver.learned_clauses()
        except Exception:
            pass
        
        return learned
    
    def extract_unsat_core(
        self,
        cnf_path: str,
        assumptions: Optional[List[int]] = None
    ) -> Optional[List[int]]:
        """
        提取UNSAT Core（如果问题是UNSAT）
        
        Args:
            cnf_path: CNF文件路径
            assumptions: 假设文字列表
            
        Returns:
            UNSAT Core中的假设子集，或None
        """
        if not self._has_binding:
            return None
        
        import pycadical
        
        num_vars, num_clauses, clauses = self.parse_cnf(cnf_path)
        
        solver = pycadical.Solver()
        
        for clause in clauses:
            for lit in clause:
                solver.add(lit)
            solver.add(0)
        
        if assumptions:
            for lit in assumptions:
                solver.assume(lit)
        
        result = solver.solve()
        
        if result == 20:  # UNSAT
            core = []
            if assumptions:
                for lit in assumptions:
                    if solver.failed(lit):
                        core.append(lit)
            return core
        
        return None

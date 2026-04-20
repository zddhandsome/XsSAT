"""
GeoSATformer Kissat Wrapper
Kissat求解器封装

Kissat是CaDiCaL的继任者，在SATComp比赛中表现优异
"""

import os
import time
import subprocess
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .solver_interface import SolverInterface, SolverResult


class KissatWrapper(SolverInterface):
    """
    Kissat SAT求解器封装
    
    Kissat特点：
    - 极致优化的CDCL实现
    - 高效的inprocessing技术
    - 出色的工业问题性能
    """
    
    def __init__(
        self,
        solver_path: Optional[str] = None,
        timeout: float = 300.0,
        verbose: bool = False,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            solver_path: Kissat可执行文件路径
            timeout: 默认超时时间
            verbose: 是否输出详细信息
            options: Kissat求解器选项
        """
        super().__init__(solver_path, timeout, verbose)
        self.options = options or {}
        
        # 查找Kissat
        if solver_path is None:
            import shutil
            self.solver_path = shutil.which('kissat')
            if self.solver_path is None:
                raise RuntimeError(
                    "Kissat not found in PATH. Please provide solver_path."
                )
        else:
            if not os.path.exists(solver_path):
                raise FileNotFoundError(f"Kissat not found at {solver_path}")
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
        使用Kissat求解CNF实例
        
        Args:
            cnf_path: CNF文件路径
            timeout: 超时时间（秒）
            use_vsids_init: 是否使用VSIDS初始化
            vsids_scores: VSIDS初始分数
            
        Returns:
            求解结果
        """
        timeout = timeout or self.timeout
        
        # 准备输入文件
        if use_vsids_init and vsids_scores is not None:
            cnf_path = self._create_enhanced_cnf(cnf_path, vsids_scores)
            cleanup_temp = True
        else:
            cleanup_temp = False
        
        # 构建命令
        cmd = self._build_command(cnf_path, timeout)
        
        # 执行求解
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10
            )
            
            solve_time = time.time() - start_time
            
            # 解析结果
            solver_result = self._parse_output(result, solve_time, timeout)
            
            return solver_result
            
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
        finally:
            if cleanup_temp and os.path.exists(cnf_path):
                try:
                    os.unlink(cnf_path)
                except Exception:
                    pass
    
    def _build_command(self, cnf_path: str, timeout: float) -> List[str]:
        """构建Kissat命令"""
        cmd = [self.solver_path]
        
        # 超时设置（Kissat使用--time=<seconds>）
        if timeout > 0:
            cmd.append(f'--time={int(timeout)}')
        
        # 输出统计信息
        cmd.append('--statistics')
        
        # 附加选项
        for key, value in self.options.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
                else:
                    cmd.append(f'--no-{key}')
            else:
                cmd.append(f'--{key}={value}')
        
        # 输入文件
        cmd.append(cnf_path)
        
        return cmd
    
    def _create_enhanced_cnf(
        self,
        cnf_path: str,
        vsids_scores: np.ndarray
    ) -> str:
        """
        创建带有初始化hints的CNF文件
        
        使用phase hints来近似VSIDS初始化
        """
        fd, enhanced_path = tempfile.mkstemp(suffix='.cnf')
        
        try:
            with os.fdopen(fd, 'w') as f_out:
                # 读取原始CNF
                with open(cnf_path, 'r') as f_in:
                    for line in f_in:
                        f_out.write(line)
                
                # 添加phase hints注释
                # Kissat可能会读取某些格式的hints
                f_out.write('\nc Phase initialization hints from GeoSATformer\n')
                
                # 按VSIDS分数排序变量
                sorted_indices = np.argsort(-vsids_scores)
                
                for rank, var_idx in enumerate(sorted_indices[:min(len(sorted_indices), 100)]):
                    score = vsids_scores[var_idx]
                    # 注释格式：c phase <var> <polarity>
                    polarity = 1 if score > 0.5 else -1
                    f_out.write(f'c phase {var_idx + 1} {polarity}\n')
            
            return enhanced_path
            
        except Exception:
            if os.path.exists(enhanced_path):
                os.unlink(enhanced_path)
            raise
    
    def _parse_output(
        self,
        result: subprocess.CompletedProcess,
        solve_time: float,
        timeout: float
    ) -> SolverResult:
        """解析Kissat输出"""
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
        
        # Kissat返回码：10=SAT, 20=UNSAT, 其他=UNKNOWN
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
        stats = self._parse_statistics(stdout)
        
        return SolverResult(
            satisfiable=satisfiable,
            assignment=assignment,
            solve_time=solve_time,
            timed_out=solve_time >= timeout,
            **stats
        )
    
    def _parse_assignment(self, output: str) -> Optional[List[int]]:
        """解析满足赋值"""
        assignment = []
        
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('v '):
                literals = line[2:].split()
                for lit in literals:
                    try:
                        lit_int = int(lit)
                        if lit_int != 0:
                            assignment.append(lit_int)
                    except ValueError:
                        continue
        
        return assignment if assignment else None
    
    def _parse_statistics(self, output: str) -> Dict[str, int]:
        """解析统计信息"""
        stats = {
            'num_decisions': 0,
            'num_propagations': 0,
            'num_conflicts': 0
        }
        
        # Kissat统计输出格式
        stat_patterns = {
            'decisions': 'num_decisions',
            'propagations': 'num_propagations',
            'conflicts': 'num_conflicts'
        }
        
        for line in output.split('\n'):
            line = line.strip().lower()
            
            for pattern, key in stat_patterns.items():
                if pattern in line:
                    # 尝试提取数字
                    parts = line.split()
                    for part in parts:
                        try:
                            value = int(part.replace(',', ''))
                            if value > 0:
                                stats[key] = value
                                break
                        except ValueError:
                            continue
        
        return stats
    
    def set_initial_activity(self, var: int, score: float):
        """设置变量的初始活动分数"""
        # Kissat命令行不直接支持，通过phase hints近似
        pass
    
    def solve_incremental(
        self,
        cnf_path: str,
        assumptions: List[int],
        timeout: Optional[float] = None
    ) -> SolverResult:
        """
        增量求解（带假设）
        
        Args:
            cnf_path: CNF文件路径
            assumptions: 假设列表
            timeout: 超时时间
            
        Returns:
            求解结果
        """
        # Kissat不直接支持增量求解
        # 创建带有单元子句的临时CNF
        timeout = timeout or self.timeout
        
        fd, temp_path = tempfile.mkstemp(suffix='.cnf')
        
        try:
            # 读取原始CNF
            with open(cnf_path, 'r') as f_in:
                content = f_in.read()
            
            # 更新子句数
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.startswith('p cnf'):
                    parts = line.split()
                    num_vars = int(parts[2])
                    num_clauses = int(parts[3]) + len(assumptions)
                    new_lines.append(f'p cnf {num_vars} {num_clauses}')
                else:
                    new_lines.append(line)
            
            # 添加假设作为单元子句
            for lit in assumptions:
                new_lines.append(f'{lit} 0')
            
            with os.fdopen(fd, 'w') as f_out:
                f_out.write('\n'.join(new_lines))
            
            return self.solve(temp_path, timeout)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def get_solver_info(self) -> Dict[str, str]:
        """获取求解器信息"""
        try:
            result = subprocess.run(
                [self.solver_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return {
                'name': 'Kissat',
                'version': result.stdout.strip(),
                'path': self.solver_path
            }
        except Exception as e:
            return {
                'name': 'Kissat',
                'version': 'unknown',
                'path': self.solver_path,
                'error': str(e)
            }


class KissatSCWrapper(KissatWrapper):
    """
    Kissat-SC (Kissat SATComp) 版本封装
    
    专门针对SATComp竞赛优化的版本
    """
    
    def __init__(
        self,
        solver_path: Optional[str] = None,
        timeout: float = 300.0,
        verbose: bool = False
    ):
        # 默认选项，针对竞赛优化
        options = {
            'stable': 1,        # 稳定模式
            'target': 2,        # 目标选择策略
            'tier1': 2,         # Tier-1子句管理
        }
        
        super().__init__(solver_path, timeout, verbose, options)


class KissatMabWrapper(KissatWrapper):
    """
    Kissat-MAB (Multi-Armed Bandit) 版本封装
    
    使用多臂老虎机策略动态选择启发式
    """
    
    def __init__(
        self,
        solver_path: Optional[str] = None,
        timeout: float = 300.0,
        verbose: bool = False
    ):
        # MAB相关选项
        options = {
            'walkinitially': True,  # 使用随机漫步初始化
        }
        
        super().__init__(solver_path, timeout, verbose, options)

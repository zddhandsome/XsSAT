"""
GeoSATformer Curriculum Learning Module
课程学习模块

实现基于问题复杂度的渐进训练策略：
- 复杂度指标：CV ratio (clause/variable), MUC大小, 变量数, 子句数
- 从简单实例逐步过渡到复杂实例
- 支持多种调度策略
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from torch.utils.data import Dataset, Sampler


class ComplexityMetric(Enum):
    """复杂度度量类型"""
    CV_RATIO = 'cv_ratio'           # Clause/Variable ratio
    NUM_VARIABLES = 'num_variables'  # 变量数
    NUM_CLAUSES = 'num_clauses'      # 子句数
    MUC_SIZE = 'muc_size'           # MUC大小（仅UNSAT）
    MUC_RATIO = 'muc_ratio'         # MUC/总子句比例
    AVG_CLAUSE_LENGTH = 'avg_clause_length'  # 平均子句长度
    DENSITY = 'density'             # 稀疏度
    COMBINED = 'combined'           # 综合指标


@dataclass
class ComplexityMetrics:
    """
    SAT实例的复杂度指标
    """
    cv_ratio: float = 0.0           # Clause/Variable ratio
    num_variables: int = 0          # 变量数
    num_clauses: int = 0            # 子句数
    muc_size: int = 0              # MUC大小
    muc_ratio: float = 0.0         # MUC占比
    avg_clause_length: float = 0.0  # 平均子句长度
    density: float = 0.0           # VSM的非零元素比例
    
    @classmethod
    def from_vsm(
        cls,
        vsm: torch.Tensor,
        muc_labels: Optional[torch.Tensor] = None
    ) -> 'ComplexityMetrics':
        """
        从VSM矩阵计算复杂度指标
        
        Args:
            vsm: Variable Space Matrix [num_clauses, num_variables]
            muc_labels: MUC标签 [num_clauses]，可选
        """
        num_clauses, num_variables = vsm.shape
        
        # CV ratio
        cv_ratio = num_clauses / max(num_variables, 1)
        
        # 非零元素（每行的非零元素数 = 子句长度）
        clause_lengths = (vsm != 0).sum(dim=1).float()
        avg_clause_length = clause_lengths.mean().item()
        
        # 稀疏度
        density = (vsm != 0).float().mean().item()
        
        # MUC指标
        muc_size = 0
        muc_ratio = 0.0
        if muc_labels is not None:
            muc_size = muc_labels.sum().item()
            muc_ratio = muc_size / max(num_clauses, 1)
        
        return cls(
            cv_ratio=cv_ratio,
            num_variables=num_variables,
            num_clauses=num_clauses,
            muc_size=int(muc_size),
            muc_ratio=muc_ratio,
            avg_clause_length=avg_clause_length,
            density=density
        )
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'cv_ratio': self.cv_ratio,
            'num_variables': self.num_variables,
            'num_clauses': self.num_clauses,
            'muc_size': self.muc_size,
            'muc_ratio': self.muc_ratio,
            'avg_clause_length': self.avg_clause_length,
            'density': self.density
        }
    
    def compute_combined_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算综合复杂度分数
        
        Args:
            weights: 各指标权重
            
        Returns:
            综合分数（越高越复杂）
        """
        if weights is None:
            weights = {
                'cv_ratio': 0.3,
                'num_variables': 0.2,
                'num_clauses': 0.2,
                'avg_clause_length': 0.15,
                'density': 0.15
            }
        
        # 归一化各指标（假设合理范围）
        normalized = {
            'cv_ratio': min(self.cv_ratio / 10.0, 1.0),
            'num_variables': min(self.num_variables / 1000.0, 1.0),
            'num_clauses': min(self.num_clauses / 5000.0, 1.0),
            'avg_clause_length': min(self.avg_clause_length / 10.0, 1.0),
            'density': self.density
        }
        
        score = sum(weights.get(k, 0) * v for k, v in normalized.items())
        return score


class CurriculumScheduler:
    """
    课程学习调度器
    
    控制训练过程中样本难度的渐进变化
    """
    
    def __init__(
        self,
        dataset: Dataset,
        complexity_key: str = 'complexity',
        total_epochs: int = 100,
        warmup_epochs: int = 10,
        schedule_type: str = 'linear',
        initial_difficulty: float = 0.0,
        final_difficulty: float = 1.0,
        pacing_function: Optional[Callable[[float], float]] = None,
        metric_type: ComplexityMetric = ComplexityMetric.COMBINED
    ):
        """
        Args:
            dataset: 训练数据集
            complexity_key: 数据集中复杂度信息的键名
            total_epochs: 总训练轮数
            warmup_epochs: 课程学习预热轮数（仅使用简单样本）
            schedule_type: 调度类型 ('linear', 'sqrt', 'exp', 'step', 'custom')
            initial_difficulty: 初始难度阈值
            final_difficulty: 最终难度阈值
            pacing_function: 自定义进度函数
            metric_type: 复杂度度量类型
        """
        self.dataset = dataset
        self.complexity_key = complexity_key
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.pacing_function = pacing_function
        self.metric_type = metric_type
        
        self.current_epoch = 0
        self.current_difficulty = initial_difficulty
        
        # 预计算所有样本的复杂度分数
        self.complexity_scores = self._compute_complexity_scores()
        
        # 按复杂度排序的样本索引
        self.sorted_indices = np.argsort(self.complexity_scores)
    
    def _compute_complexity_scores(self) -> np.ndarray:
        """计算数据集中所有样本的复杂度分数"""
        scores = []
        
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            
            if self.complexity_key in sample:
                # 数据集已预计算复杂度
                complexity = sample[self.complexity_key]
                if isinstance(complexity, ComplexityMetrics):
                    score = complexity.compute_combined_score()
                elif isinstance(complexity, dict):
                    score = complexity.get('combined_score', 0.5)
                else:
                    score = float(complexity)
            else:
                # 从VSM计算复杂度
                vsm = sample.get('vsm')
                core_labels = sample.get('core_labels')

                if vsm is not None:
                    metrics = ComplexityMetrics.from_vsm(vsm, core_labels)
                    score = metrics.compute_combined_score()
                else:
                    score = 0.5  # 默认中等难度
            
            scores.append(score)
        
        return np.array(scores)
    
    def step(self, epoch: int):
        """
        更新当前难度
        
        Args:
            epoch: 当前训练轮数
        """
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup期间保持初始难度
            self.current_difficulty = self.initial_difficulty
        else:
            # 计算进度
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            
            if self.pacing_function is not None:
                self.current_difficulty = self.pacing_function(progress)
            else:
                self.current_difficulty = self._compute_difficulty(progress)
    
    def _compute_difficulty(self, progress: float) -> float:
        """根据调度类型计算难度"""
        diff_range = self.final_difficulty - self.initial_difficulty
        
        if self.schedule_type == 'linear':
            return self.initial_difficulty + diff_range * progress
        
        elif self.schedule_type == 'sqrt':
            return self.initial_difficulty + diff_range * np.sqrt(progress)
        
        elif self.schedule_type == 'exp':
            # 指数增长，开始慢后来快
            return self.initial_difficulty + diff_range * (np.exp(2 * progress) - 1) / (np.e ** 2 - 1)
        
        elif self.schedule_type == 'step':
            # 阶梯式增长
            num_steps = 5
            step_progress = int(progress * num_steps) / num_steps
            return self.initial_difficulty + diff_range * step_progress
        
        elif self.schedule_type == 'sigmoid':
            # S型曲线，中间快两端慢
            x = 10 * (progress - 0.5)
            return self.initial_difficulty + diff_range / (1 + np.exp(-x))
        
        else:
            return self.initial_difficulty + diff_range * progress
    
    def get_sample_weights(self) -> np.ndarray:
        """
        获取当前难度下的样本权重
        
        返回每个样本的采样权重，难度低于阈值的样本权重较高
        """
        # 将复杂度分数归一化到[0,1]
        scores_normalized = (self.complexity_scores - self.complexity_scores.min()) / \
                           (self.complexity_scores.max() - self.complexity_scores.min() + 1e-8)
        
        # 计算权重：难度低于当前阈值的样本权重高
        weights = np.exp(-np.maximum(scores_normalized - self.current_difficulty, 0) * 5)
        
        # 归一化
        weights = weights / weights.sum()
        
        return weights
    
    def get_curriculum_indices(self) -> List[int]:
        """
        获取当前难度下应该使用的样本索引
        
        Returns:
            符合当前难度的样本索引列表
        """
        # 选择复杂度低于当前阈值的样本
        threshold = np.percentile(self.complexity_scores, self.current_difficulty * 100)
        valid_mask = self.complexity_scores <= threshold
        
        # 至少保留10%的样本
        min_samples = max(int(len(self.dataset) * 0.1), 100)
        
        if valid_mask.sum() < min_samples:
            # 选择最简单的min_samples个样本
            indices = self.sorted_indices[:min_samples].tolist()
        else:
            indices = np.where(valid_mask)[0].tolist()
        
        return indices
    
    def get_sampler(self) -> 'CurriculumSampler':
        """获取用于DataLoader的采样器"""
        return CurriculumSampler(self)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前状态统计"""
        indices = self.get_curriculum_indices()
        return {
            'current_epoch': self.current_epoch,
            'current_difficulty': self.current_difficulty,
            'num_available_samples': len(indices),
            'total_samples': len(self.dataset),
            'usage_ratio': len(indices) / len(self.dataset),
            'min_complexity': self.complexity_scores.min(),
            'max_complexity': self.complexity_scores.max(),
            'mean_complexity': self.complexity_scores.mean(),
            'threshold_complexity': np.percentile(self.complexity_scores, self.current_difficulty * 100)
        }


class CurriculumSampler(Sampler):
    """
    课程学习采样器
    
    根据CurriculumScheduler的当前难度进行采样
    """
    
    def __init__(
        self,
        scheduler: CurriculumScheduler,
        shuffle: bool = True,
        use_weights: bool = True
    ):
        """
        Args:
            scheduler: 课程学习调度器
            shuffle: 是否打乱顺序
            use_weights: 是否使用加权采样
        """
        self.scheduler = scheduler
        self.shuffle = shuffle
        self.use_weights = use_weights
    
    def __iter__(self):
        if self.use_weights:
            # 加权采样
            weights = self.scheduler.get_sample_weights()
            indices = np.random.choice(
                len(self.scheduler.dataset),
                size=len(self.scheduler.dataset),
                replace=True,
                p=weights
            )
        else:
            # 使用子集
            indices = np.array(self.scheduler.get_curriculum_indices())
            if self.shuffle:
                np.random.shuffle(indices)
        
        return iter(indices.tolist())
    
    def __len__(self):
        if self.use_weights:
            return len(self.scheduler.dataset)
        return len(self.scheduler.get_curriculum_indices())


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    自适应课程学习调度器
    
    根据模型在当前难度样本上的表现自动调整难度
    """
    
    def __init__(
        self,
        dataset: Dataset,
        target_accuracy: float = 0.8,
        increase_threshold: float = 0.85,
        decrease_threshold: float = 0.6,
        adjustment_rate: float = 0.05,
        **kwargs
    ):
        """
        Args:
            dataset: 训练数据集
            target_accuracy: 目标准确率
            increase_threshold: 提升难度的准确率阈值
            decrease_threshold: 降低难度的准确率阈值
            adjustment_rate: 每次调整的幅度
        """
        super().__init__(dataset, **kwargs)
        
        self.target_accuracy = target_accuracy
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.adjustment_rate = adjustment_rate
        
        self.recent_accuracies = []
        self.history = []
    
    def update_with_performance(self, accuracy: float):
        """
        根据当前性能更新难度
        
        Args:
            accuracy: 当前在curriculum样本上的准确率
        """
        self.recent_accuracies.append(accuracy)
        
        # 使用最近几个epoch的平均准确率
        if len(self.recent_accuracies) > 3:
            self.recent_accuracies = self.recent_accuracies[-3:]
        
        avg_accuracy = np.mean(self.recent_accuracies)
        
        # 记录历史
        self.history.append({
            'epoch': self.current_epoch,
            'difficulty': self.current_difficulty,
            'accuracy': accuracy,
            'avg_accuracy': avg_accuracy
        })
        
        # 调整难度
        if avg_accuracy > self.increase_threshold:
            # 表现好，增加难度
            self.current_difficulty = min(
                self.final_difficulty,
                self.current_difficulty + self.adjustment_rate
            )
        elif avg_accuracy < self.decrease_threshold:
            # 表现差，降低难度
            self.current_difficulty = max(
                self.initial_difficulty,
                self.current_difficulty - self.adjustment_rate
            )
    
    def get_history(self) -> List[Dict]:
        """获取调整历史"""
        return self.history


class MultiMetricCurriculumScheduler(CurriculumScheduler):
    """
    多指标课程学习调度器
    
    支持按不同复杂度指标独立调度
    """
    
    def __init__(
        self,
        dataset: Dataset,
        metrics: List[ComplexityMetric],
        metric_weights: Optional[Dict[ComplexityMetric, float]] = None,
        **kwargs
    ):
        """
        Args:
            dataset: 训练数据集
            metrics: 使用的复杂度指标列表
            metric_weights: 各指标权重
        """
        self.metrics = metrics
        self.metric_weights = metric_weights or {m: 1.0 / len(metrics) for m in metrics}
        
        super().__init__(dataset, **kwargs)
    
    def _compute_complexity_scores(self) -> np.ndarray:
        """计算多指标综合复杂度分数"""
        metric_scores = {m: [] for m in self.metrics}

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            vsm = sample.get('vsm')
            core_labels = sample.get('core_labels')

            if vsm is not None:
                metrics = ComplexityMetrics.from_vsm(vsm, core_labels)
                metrics_dict = metrics.to_dict()
                
                for m in self.metrics:
                    metric_scores[m].append(metrics_dict.get(m.value, 0.5))
            else:
                for m in self.metrics:
                    metric_scores[m].append(0.5)
        
        # 归一化每个指标
        normalized_scores = {}
        for m in self.metrics:
            scores = np.array(metric_scores[m])
            normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            normalized_scores[m] = normalized
        
        # 加权综合
        combined = np.zeros(len(self.dataset))
        for m in self.metrics:
            combined += self.metric_weights[m] * normalized_scores[m]
        
        return combined

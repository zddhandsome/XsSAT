"""
GeoSATformer Evaluator Module
评估器模块

实现模型评估功能：
1. 分类准确率：SAT/UNSAT二分类准确率
2. UNSAT Core F1：MUC预测的精确率和召回率
3. 求解加速比：(原始时间 - 加速后时间) / 原始时间
4. PAR-2 Score：考虑超时的加权平均时间
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    # SAT分类指标
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # SAT/UNSAT分别的准确率
    sat_accuracy: float = 0.0
    unsat_accuracy: float = 0.0
    
    # UNSAT Core预测指标
    muc_precision: float = 0.0
    muc_recall: float = 0.0
    muc_f1: float = 0.0
    muc_iou: float = 0.0  # Intersection over Union
    
    # 求解器集成指标
    speedup_ratio: float = 0.0
    par2_score: float = 0.0
    solve_rate: float = 0.0
    
    # VSIDS预测指标
    vsids_mse: float = 0.0
    vsids_ranking_accuracy: float = 0.0
    
    # 其他统计
    num_samples: int = 0
    num_sat: int = 0
    num_unsat: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'sat_accuracy': self.sat_accuracy,
            'unsat_accuracy': self.unsat_accuracy,
            'muc_precision': self.muc_precision,
            'muc_recall': self.muc_recall,
            'muc_f1': self.muc_f1,
            'muc_iou': self.muc_iou,
            'speedup_ratio': self.speedup_ratio,
            'par2_score': self.par2_score,
            'solve_rate': self.solve_rate,
            'vsids_mse': self.vsids_mse,
            'vsids_ranking_accuracy': self.vsids_ranking_accuracy,
            'num_samples': self.num_samples,
            'num_sat': self.num_sat,
            'num_unsat': self.num_unsat
        }


class Evaluator:
    """
    GeoSATformer 评估器
    
    提供全面的模型评估功能
    """
    
    def __init__(
        self,
        threshold_sat: float = 0.5,
        threshold_muc: float = 0.5,
        use_amp: bool = True,
        solver_interface: Optional[Any] = None,
        timeout: float = 300.0
    ):
        """
        Args:
            threshold_sat: SAT分类阈值
            threshold_muc: MUC预测阈值
            use_amp: 是否使用混合精度
            solver_interface: SAT求解器接口（用于加速比评估）
            timeout: 求解超时时间（秒）
        """
        self.threshold_sat = threshold_sat
        self.threshold_muc = threshold_muc
        self.use_amp = use_amp
        self.solver_interface = solver_interface
        self.timeout = timeout
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        全面评估模型
        
        Args:
            model: 要评估的模型
            dataloader: 数据加载器
            device: 计算设备
            verbose: 是否显示进度
            
        Returns:
            评估指标字典
        """
        model.eval()
        
        # 收集预测结果
        all_sat_preds = []
        all_sat_labels = []
        all_muc_preds = []
        all_muc_labels = []
        all_muc_masks = []
        all_vsids_preds = []
        all_vsids_labels = []
        all_var_masks = []
        
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                with autocast(enabled=self.use_amp):
                    outputs = model(
                        batch['vsm'],
                        batch.get('clause_mask'),
                        batch.get('var_mask')
                    )
                
                # SAT预测
                sat_probs = torch.sigmoid(outputs['sat_pred'])
                all_sat_preds.append(sat_probs.cpu())
                all_sat_labels.append(batch['sat_label'].cpu())
                
                # MUC预测（仅UNSAT样本）
                if 'clause_scores' in outputs and 'core_labels' in batch:
                    all_muc_preds.append(torch.sigmoid(outputs['clause_scores']).cpu())
                    all_muc_labels.append(batch['core_labels'].cpu())
                    if 'clause_mask' in batch:
                        all_muc_masks.append(batch['clause_mask'].cpu())

                # VSIDS预测
                if 'vsids_scores' in outputs and 'vsids_labels' in batch:
                    all_vsids_preds.append(outputs['vsids_scores'].cpu())
                    all_vsids_labels.append(batch['vsids_labels'].cpu())
                    if 'var_mask' in batch:
                        all_var_masks.append(batch['var_mask'].cpu())
        
        # 合并结果
        sat_preds = torch.cat(all_sat_preds, dim=0)
        sat_labels = torch.cat(all_sat_labels, dim=0)
        
        # 计算SAT分类指标
        sat_metrics = self._compute_sat_metrics(sat_preds, sat_labels)
        
        # 计算MUC指标
        muc_metrics = {}
        if all_muc_preds:
            muc_preds = torch.cat(all_muc_preds, dim=0)
            muc_labels = torch.cat(all_muc_labels, dim=0)
            muc_masks = torch.cat(all_muc_masks, dim=0) if all_muc_masks else None
            
            # 仅评估UNSAT样本
            unsat_mask = sat_labels == 0
            if unsat_mask.any():
                muc_metrics = self._compute_muc_metrics(
                    muc_preds[unsat_mask],
                    muc_labels[unsat_mask],
                    muc_masks[unsat_mask] if muc_masks is not None else None
                )
        
        # 计算VSIDS指标
        vsids_metrics = {}
        if all_vsids_preds:
            vsids_preds = torch.cat(all_vsids_preds, dim=0)
            vsids_labels = torch.cat(all_vsids_labels, dim=0)
            var_masks = torch.cat(all_var_masks, dim=0) if all_var_masks else None
            vsids_metrics = self._compute_vsids_metrics(vsids_preds, vsids_labels, var_masks)
        
        # 合并所有指标
        metrics = EvaluationMetrics(
            num_samples=len(sat_labels),
            num_sat=(sat_labels == 1).sum().item(),
            num_unsat=(sat_labels == 0).sum().item(),
            **sat_metrics,
            **muc_metrics,
            **vsids_metrics
        )
        
        return metrics.to_dict()
    
    def _compute_sat_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """计算SAT分类指标"""
        binary_preds = (preds > self.threshold_sat).long()
        
        # 基础准确率
        accuracy = (binary_preds == labels).float().mean().item()
        
        # 分类别准确率
        sat_mask = labels == 1
        unsat_mask = labels == 0
        
        sat_accuracy = (binary_preds[sat_mask] == 1).float().mean().item() if sat_mask.any() else 0.0
        unsat_accuracy = (binary_preds[unsat_mask] == 0).float().mean().item() if unsat_mask.any() else 0.0
        
        # Precision, Recall, F1（以SAT为正类）
        tp = ((binary_preds == 1) & (labels == 1)).sum().float()
        fp = ((binary_preds == 1) & (labels == 0)).sum().float()
        fn = ((binary_preds == 0) & (labels == 1)).sum().float()
        
        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sat_accuracy': sat_accuracy,
            'unsat_accuracy': unsat_accuracy
        }
    
    def _compute_muc_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """计算MUC预测指标"""
        binary_preds = (preds > self.threshold_muc).long()
        
        if masks is not None:
            # 只计算有效位置
            valid = masks.bool()
        else:
            valid = torch.ones_like(labels).bool()
        
        # 计算每个样本的指标，然后平均
        precisions = []
        recalls = []
        ious = []
        
        for i in range(preds.shape[0]):
            pred_i = binary_preds[i][valid[i]]
            label_i = labels[i][valid[i]]
            
            tp = ((pred_i == 1) & (label_i == 1)).sum().float()
            fp = ((pred_i == 1) & (label_i == 0)).sum().float()
            fn = ((pred_i == 0) & (label_i == 1)).sum().float()
            
            precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
            
            # IoU
            intersection = tp
            union = tp + fp + fn
            iou = (intersection / union).item() if union > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            ious.append(iou)
        
        muc_precision = np.mean(precisions)
        muc_recall = np.mean(recalls)
        muc_f1 = 2 * muc_precision * muc_recall / (muc_precision + muc_recall) if (muc_precision + muc_recall) > 0 else 0.0
        muc_iou = np.mean(ious)
        
        return {
            'muc_precision': muc_precision,
            'muc_recall': muc_recall,
            'muc_f1': muc_f1,
            'muc_iou': muc_iou
        }
    
    def _compute_vsids_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """计算VSIDS预测指标"""
        if masks is not None:
            valid = masks.bool()
            mse = ((preds - labels) ** 2)[valid].mean().item()
        else:
            mse = ((preds - labels) ** 2).mean().item()
        
        # 排序准确率（采样pairs）
        ranking_accuracies = []
        num_samples = min(100, preds.shape[0])
        
        for i in range(num_samples):
            if masks is not None:
                valid_idx = torch.where(masks[i] > 0)[0]
            else:
                valid_idx = torch.arange(preds.shape[1])
            
            if len(valid_idx) < 2:
                continue
            
            # 采样pairs
            n = min(100, len(valid_idx) * (len(valid_idx) - 1) // 2)
            idx1 = valid_idx[torch.randint(len(valid_idx), (n,))]
            idx2 = valid_idx[torch.randint(len(valid_idx), (n,))]
            
            # 移除相同索引
            diff = idx1 != idx2
            idx1, idx2 = idx1[diff], idx2[diff]
            
            if len(idx1) == 0:
                continue
            
            pred_diff = preds[i, idx1] - preds[i, idx2]
            label_diff = labels[i, idx1] - labels[i, idx2]
            
            # 符号一致性
            correct = ((pred_diff > 0) == (label_diff > 0)).float().mean().item()
            ranking_accuracies.append(correct)
        
        ranking_accuracy = np.mean(ranking_accuracies) if ranking_accuracies else 0.0
        
        return {
            'vsids_mse': mse,
            'vsids_ranking_accuracy': ranking_accuracy
        }
    
    def evaluate_with_solver(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        cnf_paths: List[str],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        评估与SAT求解器集成的性能
        
        Args:
            model: 模型
            dataloader: 数据加载器
            device: 计算设备
            cnf_paths: CNF文件路径列表
            verbose: 是否显示进度
            
        Returns:
            包含求解加速比等指标的字典
        """
        if self.solver_interface is None:
            raise ValueError("Solver interface not provided")
        
        model.eval()
        
        baseline_times = []
        enhanced_times = []
        baseline_solved = 0
        enhanced_solved = 0
        
        iterator = tqdm(zip(dataloader, cnf_paths), desc="Solving", total=len(cnf_paths)) if verbose else zip(dataloader, cnf_paths)
        
        for batch, cnf_path in iterator:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # 获取模型预测的VSIDS分数
            with torch.no_grad():
                with autocast(enabled=self.use_amp):
                    outputs = model(
                        batch['vsm'],
                        batch.get('clause_mask'),
                        batch.get('var_mask')
                    )
                vsids_scores = outputs['vsids_scores'][0].cpu().numpy()
            
            # 基线求解（不使用VSIDS初始化）
            start_time = time.time()
            result_baseline = self.solver_interface.solve(
                cnf_path,
                timeout=self.timeout,
                use_vsids_init=False
            )
            baseline_time = time.time() - start_time
            
            if result_baseline['solved']:
                baseline_solved += 1
                baseline_times.append(baseline_time)
            else:
                baseline_times.append(self.timeout * 2)  # PAR-2
            
            # 增强求解（使用VSIDS初始化）
            start_time = time.time()
            result_enhanced = self.solver_interface.solve(
                cnf_path,
                timeout=self.timeout,
                use_vsids_init=True,
                vsids_scores=vsids_scores
            )
            enhanced_time = time.time() - start_time
            
            if result_enhanced['solved']:
                enhanced_solved += 1
                enhanced_times.append(enhanced_time)
            else:
                enhanced_times.append(self.timeout * 2)  # PAR-2
        
        # 计算指标
        n = len(baseline_times)
        
        speedup_ratio = 0.0
        if sum(baseline_times) > 0:
            speedup_ratio = (sum(baseline_times) - sum(enhanced_times)) / sum(baseline_times)
        
        par2_baseline = np.mean(baseline_times)
        par2_enhanced = np.mean(enhanced_times)
        
        return {
            'speedup_ratio': speedup_ratio,
            'par2_baseline': par2_baseline,
            'par2_enhanced': par2_enhanced,
            'solve_rate_baseline': baseline_solved / n,
            'solve_rate_enhanced': enhanced_solved / n,
            'num_evaluated': n
        }
    
    def compute_calibration(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        计算模型校准度（calibration）
        
        Args:
            preds: 预测概率
            labels: 真实标签
            num_bins: 分箱数量
            
        Returns:
            校准度指标，包括ECE、MCE等
        """
        preds = preds.numpy()
        labels = labels.numpy()
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (preds > bin_lower) & (preds <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = preds[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        return {
            'ece': ece,  # Expected Calibration Error
            'mce': mce,  # Maximum Calibration Error
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }


class BenchmarkEvaluator:
    """
    基准测试评估器
    
    用于在标准基准（如SATComp）上评估模型
    """
    
    def __init__(
        self,
        evaluator: Evaluator,
        benchmark_dir: str,
        categories: Optional[List[str]] = None
    ):
        """
        Args:
            evaluator: 基础评估器
            benchmark_dir: 基准测试目录
            categories: 要评估的类别列表
        """
        self.evaluator = evaluator
        self.benchmark_dir = benchmark_dir
        self.categories = categories or ['random', 'crafted', 'industrial']
    
    def run_benchmark(
        self,
        model: nn.Module,
        device: torch.device,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        运行基准测试
        
        Args:
            model: 模型
            device: 设备
            verbose: 是否显示进度
            
        Returns:
            每个类别的评估结果
        """
        results = {}
        
        for category in self.categories:
            if verbose:
                print(f"\n=== Evaluating {category} ===")
            
            # 这里需要根据实际数据集结构加载数据
            # 示例：results[category] = self.evaluator.evaluate(...)
            results[category] = {}
        
        # 汇总
        results['overall'] = self._aggregate_results(results)
        
        return results
    
    def _aggregate_results(
        self,
        category_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """汇总各类别结果"""
        if not category_results:
            return {}
        
        # 计算加权平均（可以根据样本数加权）
        all_metrics = defaultdict(list)
        
        for category, metrics in category_results.items():
            if category == 'overall':
                continue
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k].append(v)
        
        return {k: np.mean(v) for k, v in all_metrics.items()}

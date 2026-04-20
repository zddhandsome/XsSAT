"""
Adaptive Head Slicing Module

基于 Self-Satisfied 的 Head Slicing 技术，
实现自适应序列压缩以降低计算复杂度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HeadSlicingLayer(nn.Module):
    """
    Head Slicing Layer
    
    根据重要性分数对序列进行裁剪，保留高分数的 tokens。
    
    Args:
        embed_dim: 嵌入维度
        slicing_ratio: 保留比例 (0, 1]
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        slicing_ratio: float = 0.5
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.slicing_ratio = slicing_ratio
        
        # 分数预测网络
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_indices: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 [batch, seq_len, embed_dim]
            mask: 有效位置掩码 [batch, seq_len]
            return_indices: 是否返回保留的索引
            
        Returns:
            裁剪后的张量和分数
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算重要性分数
        scores = self.score_net(x).squeeze(-1)  # [batch, seq_len]
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # 计算要保留的数量
        if mask is not None:
            num_valid = mask.sum(dim=1)
            num_keep = (num_valid.float() * self.slicing_ratio).long().clamp(min=1)
        else:
            num_keep = torch.full(
                (batch_size,), 
                int(seq_len * self.slicing_ratio),
                device=x.device, dtype=torch.long
            ).clamp(min=1)
        
        # 获取 top-k 索引（使用固定大小以支持批处理）
        max_keep = num_keep.max().item()
        _, top_indices = torch.topk(scores, k=max_keep, dim=1)
        
        # 排序索引以保持原始顺序
        top_indices_sorted, _ = torch.sort(top_indices, dim=1)
        
        # 收集保留的 tokens
        batch_indices = torch.arange(
            batch_size, device=x.device
        ).unsqueeze(1).expand(-1, max_keep)
        
        sliced_x = x[batch_indices, top_indices_sorted]
        sliced_scores = scores[batch_indices, top_indices_sorted]
        
        if return_indices:
            return sliced_x, sliced_scores, top_indices_sorted
            
        return sliced_x, sliced_scores


class AdaptiveHeadSlicing(nn.Module):
    """
    Adaptive Head Slicing
    
    基于 UNSAT Core 预测分数的自适应序列压缩。
    高分数的 tokens（更可能属于 UNSAT Core）被优先保留。
    
    Args:
        embed_dim: 嵌入维度
        slicing_ratio: 基础保留比例
        min_tokens: 最小保留 token 数
        use_temperature: 是否使用温度参数
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        slicing_ratio: float = 0.5,
        min_tokens: int = 4,
        use_temperature: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.slicing_ratio = slicing_ratio
        self.min_tokens = min_tokens
        self.use_temperature = use_temperature
        
        # 重要性评分网络
        self.importance_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # 温度参数（可学习）
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.temperature = 1.0
            
        # 软选择的投影
        self.selection_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hard_selection: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 [batch, seq_len, embed_dim]
            mask: 有效位置掩码 [batch, seq_len]
            hard_selection: 是否使用硬选择（推理时）

        Returns:
            选择后的张量和重要性分数
        """
        batch_size, seq_len, _ = x.shape

        # 计算重要性分数
        scores = self.importance_scorer(x).squeeze(-1)  # [batch, seq_len]

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # 统一使用软选择，保证训练和验证行为一致
        # 这样可以避免 train/eval 模式下输出维度不同导致的不稳定
        return self._soft_selection(x, scores, mask)
    
    def _hard_selection(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """硬选择"""
        batch_size, seq_len, _ = x.shape
        
        # 计算保留数量
        if mask is not None:
            num_valid = mask.sum(dim=1)
            num_keep = (num_valid.float() * self.slicing_ratio).long()
        else:
            num_keep = torch.full(
                (batch_size,), 
                int(seq_len * self.slicing_ratio),
                device=x.device, dtype=torch.long
            )
        num_keep = num_keep.clamp(min=self.min_tokens, max=seq_len)
        
        max_keep = num_keep.max().item()
        
        # Top-k 选择
        _, indices = torch.topk(scores, k=max_keep, dim=1)
        indices_sorted, _ = torch.sort(indices, dim=1)
        
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1)
        selected = x[batch_idx.expand(-1, max_keep), indices_sorted]
        selected_scores = torch.sigmoid(scores[batch_idx.expand(-1, max_keep), indices_sorted])
        
        return selected, selected_scores
    
    def _soft_selection(
        self,
        x: torch.Tensor,
        scores: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        软选择（可微）

        使用注意力权重实现软选择，保留所有 tokens 但加权。
        """
        # 计算软选择权重 - 使用固定温度避免数值不稳定
        temp = 1.0  # 使用固定温度

        # 对 scores 进行裁剪避免极端值
        scores_clamped = scores.clamp(-50, 50)
        selection_weights = F.softmax(scores_clamped / temp, dim=-1)

        if mask is not None:
            selection_weights = selection_weights * mask.float()
            # 重新归一化
            weight_sum = selection_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            selection_weights = selection_weights / weight_sum

        # 加权表示
        # 不改变维度，而是调整每个 token 的贡献
        weighted_x = x * selection_weights.unsqueeze(-1)
        weighted_x = self.selection_proj(weighted_x)

        return weighted_x, selection_weights


class PyramidHeadSlicing(nn.Module):
    """
    金字塔 Head Slicing
    
    形成金字塔结构，逐层降低序列长度。
    
    Args:
        embed_dim: 嵌入维度
        num_levels: 金字塔层数
        slicing_ratios: 每层的保留比例
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_levels: int = 3,
        slicing_ratios: Optional[Tuple[float, ...]] = None
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        
        # 默认金字塔比例
        if slicing_ratios is None:
            slicing_ratios = tuple(0.5 ** (i + 1) for i in range(num_levels))
        self.slicing_ratios = slicing_ratios
        
        # 每层的 slicing 模块
        self.slicing_layers = nn.ModuleList([
            AdaptiveHeadSlicing(embed_dim, ratio)
            for ratio in slicing_ratios
        ])
        
        # 层间 Transformer
        self.level_transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_levels)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: [batch, seq_len]
            
        Returns:
            最终表示和每层的输出
        """
        level_outputs = []
        level_scores = []
        
        for level in range(self.num_levels):
            # Transformer 处理
            x = self.level_transformers[level](
                x, src_key_padding_mask=~mask if mask is not None else None
            )
            
            # Head Slicing
            x, scores = self.slicing_layers[level](x, mask)
            
            level_outputs.append(x)
            level_scores.append(scores)
            
            # 更新掩码（所有保留的位置都有效）
            if mask is not None:
                new_len = x.shape[1]
                mask = torch.ones(
                    x.shape[0], new_len,
                    dtype=torch.bool, device=x.device
                )
        
        return x, level_outputs, level_scores


class DifferentiableTopK(torch.autograd.Function):
    """
    可微分的 Top-K 选择
    
    前向传播使用硬 top-k，反向传播使用软梯度。
    """
    
    @staticmethod
    def forward(ctx, scores, k):
        # 硬 top-k 选择
        _, indices = torch.topk(scores, k, dim=-1)
        
        # 创建 one-hot 选择掩码
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, indices, 1.0)
        
        ctx.save_for_backward(scores)
        ctx.k = k
        
        return mask
    
    @staticmethod
    def backward(ctx, grad_output):
        scores, = ctx.saved_tensors
        
        # 使用 softmax 计算软梯度
        soft_selection = F.softmax(scores, dim=-1)
        grad_scores = grad_output * soft_selection
        
        return grad_scores, None


class LearnableSlicingRatio(nn.Module):
    """
    可学习的 Slicing 比例
    
    根据输入动态预测最优的保留比例。
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        min_ratio: float = 0.1,
        max_ratio: float = 0.9
    ):
        super().__init__()
        
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
        # 比例预测网络
        self.ratio_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            
        Returns:
            预测的比例 [batch]
        """
        # 使用全局平均作为输入
        global_repr = x.mean(dim=1)
        
        # 预测比例
        ratio = self.ratio_predictor(global_repr).squeeze(-1)
        
        # 缩放到有效范围
        ratio = self.min_ratio + ratio * (self.max_ratio - self.min_ratio)
        
        return ratio

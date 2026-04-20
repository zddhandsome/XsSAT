"""
Hierarchical Aggregation Module

实现来自 SATformer 的层次化子句聚合，
逐层合并子句组，捕获多子句交互。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class AttentionUnit(nn.Module):
    """
    Attention Unit
    
    单个注意力聚合单元，用于将一组子句嵌入聚合为单个表示。
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: Dropout 率
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 查询向量（可学习）
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-Forward
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch, group_size, embed_dim]
            mask: 有效位置掩码 [batch, group_size]
            
        Returns:
            聚合后的表示 [batch, embed_dim]
        """
        batch_size = x.shape[0]
        
        # 扩展查询向量
        query = self.query.expand(batch_size, -1, -1)
        
        # 转换掩码格式
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # MHA 需要 True 表示被掩盖的位置
        
        # 注意力聚合
        out, _ = self.attention(
            query, x, x,
            key_padding_mask=key_padding_mask
        )
        
        # 移除序列维度
        out = out.squeeze(1)
        
        # Feed-Forward
        out = out + self.ff(out)
        out = self.norm(out)
        
        return out


class HierarchicalAggregation(nn.Module):
    """
    Hierarchical Aggregation
    
    层次化子句聚合模块，逐层合并子句组。
    
    结构：
    Level 0: 所有子句
    Level 1: 将子句分组（每组 group_size 个），每组聚合为一个表示
    Level 2: 继续分组和聚合
    ...
    最终: 得到全局表示
    
    Args:
        embed_dim: 嵌入维度
        num_levels: 层次数
        group_size: 每组的元素数
        num_heads: 注意力头数
        dropout: Dropout 率
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_levels: int = 3,
        group_size: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.group_size = group_size
        
        # 每一层的注意力单元
        self.attention_units = nn.ModuleList([
            AttentionUnit(embed_dim, num_heads, dropout)
            for _ in range(num_levels)
        ])
        
        # 层间投影
        self.level_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(num_levels - 1)
        ])
        
        # 最终聚合
        self.final_aggregation = AttentionUnit(embed_dim, num_heads, dropout)
        
    def forward(
        self,
        clause_embeddings: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            clause_embeddings: 子句嵌入 [batch, num_clauses, embed_dim]
            clause_mask: 子句掩码 [batch, num_clauses]
            
        Returns:
            全局表示 [batch, embed_dim]
        """
        x = clause_embeddings
        mask = clause_mask
        
        for level in range(self.num_levels):
            x, mask = self._aggregate_level(x, mask, level)
            
            # 层间投影（除了最后一层）
            if level < self.num_levels - 1:
                x = self.level_projections[level](x)
        
        # 最终聚合到单个向量
        if x.shape[1] > 1:
            global_repr = self.final_aggregation(x, mask)
        else:
            global_repr = x.squeeze(1)
            
        return global_repr
    
    def _aggregate_level(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        level: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        单层聚合
        
        Args:
            x: [batch, seq_len, embed_dim]
            mask: [batch, seq_len]
            level: 当前层级
            
        Returns:
            聚合后的张量和掩码
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算分组数
        num_groups = (seq_len + self.group_size - 1) // self.group_size
        
        # 填充到 group_size 的倍数
        padded_len = num_groups * self.group_size
        if seq_len < padded_len:
            padding = torch.zeros(
                batch_size, padded_len - seq_len, self.embed_dim,
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=1)
            
            if mask is not None:
                mask_padding = torch.zeros(
                    batch_size, padded_len - seq_len,
                    device=mask.device, dtype=mask.dtype
                )
                mask = torch.cat([mask, mask_padding], dim=1)
        
        # 重塑为组 [batch * num_groups, group_size, embed_dim]
        x = x.view(batch_size * num_groups, self.group_size, self.embed_dim)
        
        if mask is not None:
            mask = mask.view(batch_size * num_groups, self.group_size)
        
        # 聚合每组
        attention_unit = self.attention_units[level]
        aggregated = attention_unit(x, mask)
        
        # 恢复批次维度 [batch, num_groups, embed_dim]
        aggregated = aggregated.view(batch_size, num_groups, self.embed_dim)
        
        # 创建新的掩码（每组如果有至少一个有效元素则有效）
        new_mask = None
        if mask is not None:
            mask = mask.view(batch_size, num_groups, self.group_size)
            new_mask = mask.any(dim=2)
            
        return aggregated, new_mask
    
    def forward_with_intermediates(
        self,
        clause_embeddings: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        带中间层输出的前向传播（用于分析）
        
        Returns:
            全局表示和各层中间输出
        """
        x = clause_embeddings
        mask = clause_mask
        intermediates = [x]
        
        for level in range(self.num_levels):
            x, mask = self._aggregate_level(x, mask, level)
            intermediates.append(x)
            
            if level < self.num_levels - 1:
                x = self.level_projections[level](x)
        
        if x.shape[1] > 1:
            global_repr = self.final_aggregation(x, mask)
        else:
            global_repr = x.squeeze(1)
            
        return global_repr, intermediates


class AdaptiveHierarchicalAggregation(HierarchicalAggregation):
    """
    自适应层次化聚合
    
    根据输入大小动态调整层次数和组大小。
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        max_levels: int = 5,
        base_group_size: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_levels=max_levels,
            group_size=base_group_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.max_levels = max_levels
        self.base_group_size = base_group_size
        
    def forward(
        self,
        clause_embeddings: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """自适应前向传播"""
        num_clauses = clause_embeddings.shape[1]
        
        # 计算需要的层数
        actual_levels = 1
        current_size = num_clauses
        while current_size > 1 and actual_levels < self.max_levels:
            current_size = (current_size + self.base_group_size - 1) // self.base_group_size
            actual_levels += 1
            
        # 只使用需要的层
        x = clause_embeddings
        mask = clause_mask
        
        for level in range(actual_levels):
            x, mask = self._aggregate_level(x, mask, level)
            
            if level < actual_levels - 1 and level < len(self.level_projections):
                x = self.level_projections[level](x)
        
        if x.shape[1] > 1:
            global_repr = self.final_aggregation(x, mask)
        else:
            global_repr = x.squeeze(1)
            
        return global_repr


class PoolingAggregation(nn.Module):
    """
    池化聚合（简化版本）
    
    使用简单的池化操作进行聚合，作为层次化聚合的替代方案。
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        pooling: str = 'attention'
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pooling = pooling
        
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1)
            )
            
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: [batch, seq_len]
            
        Returns:
            聚合表示 [batch, embed_dim]
        """
        if self.pooling == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1)
                pooled = pooled / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
                
        elif self.pooling == 'max':
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled, _ = x.max(dim=1)
            
        elif self.pooling == 'attention':
            scores = self.attention_pool(x).squeeze(-1)
            
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
                
            weights = F.softmax(scores, dim=-1)
            pooled = torch.einsum('bs,bse->be', weights, x)
            
        else:
            pooled = x.mean(dim=1)
            
        pooled = self.output_proj(pooled)
        pooled = self.norm(pooled)
        
        return pooled

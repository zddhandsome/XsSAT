"""
Variable Tower Module

Variable Tower 从列视角处理 VSM，学习变量级别的表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class VariableTransformerBlock(nn.Module):
    """
    Variable Transformer Block
    
    与 ClauseTransformerBlock 结构相同，但针对变量维度优化。
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        ff_dim: 前馈网络隐藏维度
        dropout: Dropout 率
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.scale = self.head_dim ** -0.5
        
        # Multi-Head Self-Attention
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: 注意力掩码
            
        Returns:
            [batch, seq_len, embed_dim]
        """
        # Pre-Norm Self-Attention
        residual = x
        x = self.norm1(x)
        x = self._attention(x, mask)
        x = residual + x
        
        # Pre-Norm Feed-Forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x
    
    def _attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Efficient Multi-Head Attention"""
        batch_size, seq_len, _ = x.shape
        
        # 一次性投影 Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out)


class VariableTower(nn.Module):
    """
    Variable Tower
    
    处理 VSM 的列视角，学习变量级别的表示。
    设计上与 Clause Tower 对称，但可以有不同的配置。
    
    Args:
        embed_dim: 嵌入维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        ff_dim: 前馈网络隐藏维度
        dropout: Dropout 率
        use_relative_pos: 是否使用相对位置编码
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_relative_pos: bool = False
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_relative_pos = use_relative_pos
        
        # Transformer 层
        self.layers = nn.ModuleList([
            VariableTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 最终 Layer Norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 相对位置编码（可选）
        if use_relative_pos:
            self.rel_pos_bias = RelativePositionalBias(num_heads, 128)
            
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: 注意力掩码
            
        Returns:
            [batch, seq_len, embed_dim]
        """
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.final_norm(x)
        
        return x
    
    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        带中间层输出的前向传播（用于分析）
        
        Returns:
            最终输出和中间层列表
        """
        intermediates = []
        
        for layer in self.layers:
            x = layer(x, mask)
            intermediates.append(x)
            
        x = self.final_norm(x)
        
        return x, intermediates


class RelativePositionalBias(nn.Module):
    """
    相对位置偏置
    
    为注意力分数添加基于相对位置的偏置。
    """
    
    def __init__(
        self,
        num_heads: int,
        max_distance: int = 128
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # 相对位置嵌入表
        self.bias_table = nn.Parameter(
            torch.randn(2 * max_distance + 1, num_heads) * 0.02
        )
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        计算相对位置偏置
        
        Args:
            seq_len: 序列长度
            
        Returns:
            偏置矩阵 [seq_len, seq_len, num_heads]
        """
        positions = torch.arange(seq_len, device=self.bias_table.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # 裁剪到有效范围
        relative_positions = relative_positions.clamp(
            -self.max_distance, self.max_distance
        )
        
        # 转换为索引
        indices = relative_positions + self.max_distance
        
        return self.bias_table[indices]


class VariableEmbedding(nn.Module):
    """
    变量嵌入模块
    
    将变量在各子句中的出现情况嵌入。
    """
    
    def __init__(
        self,
        max_clauses: int,
        embed_dim: int,
        aggregation: str = 'attention'
    ):
        super().__init__()
        
        self.max_clauses = max_clauses
        self.embed_dim = embed_dim
        self.aggregation = aggregation
        
        # 出现状态嵌入
        self.occurrence_embed = nn.Embedding(3, embed_dim, padding_idx=1)
        
        # 子句位置嵌入
        self.clause_position_embed = nn.Embedding(max_clauses, embed_dim)
        
        # 注意力聚合
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1)
            )
            
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, vsm_col: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vsm_col: VSM 的一列 [batch, max_clauses]
            
        Returns:
            变量嵌入 [batch, embed_dim]
        """
        batch_size = vsm_col.shape[0]
        
        # 转换为 embedding 索引
        indices = (vsm_col + 1).long()
        
        # 状态嵌入
        occ_embed = self.occurrence_embed(indices)
        
        # 添加位置嵌入
        positions = torch.arange(self.max_clauses, device=vsm_col.device)
        pos_embed = self.clause_position_embed(positions)
        combined = occ_embed + pos_embed
        
        # 聚合
        if self.aggregation == 'attention':
            # 注意力加权聚合
            attn_scores = self.attention(combined).squeeze(-1)
            
            # 掩盖未出现的位置
            mask = (vsm_col != 0)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            var_embed = torch.einsum('bc,bce->be', attn_weights, combined)
        elif self.aggregation == 'mean':
            mask = (vsm_col != 0).unsqueeze(-1).float()
            var_embed = (combined * mask).sum(dim=1)
            var_embed = var_embed / mask.sum(dim=1).clamp(min=1)
        else:
            var_embed = combined.mean(dim=1)
            
        return self.output_proj(var_embed)


class VariableInteractionLayer(nn.Module):
    """
    变量交互层
    
    学习变量之间的共现关系。
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 共现图注意力
        self.cooccurrence_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        var_embeddings: torch.Tensor,
        cooccurrence_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            var_embeddings: 变量嵌入 [batch, num_vars, embed_dim]
            cooccurrence_matrix: 变量共现矩阵（可选）
            
        Returns:
            更新后的变量嵌入
        """
        # 自注意力捕获变量交互
        attn_output, _ = self.cooccurrence_attn(
            var_embeddings, var_embeddings, var_embeddings,
            attn_mask=cooccurrence_matrix
        )
        
        # 门控融合
        gate = self.gate(torch.cat([var_embeddings, attn_output], dim=-1))
        output = gate * attn_output + (1 - gate) * var_embeddings
        
        return self.norm(output)

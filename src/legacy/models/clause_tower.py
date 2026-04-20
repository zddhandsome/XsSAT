"""
Clause Tower Module

Clause Tower 从行视角处理 VSM，学习子句级别的表示。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ClauseTransformerBlock(nn.Module):
    """
    Clause Transformer Block
    
    标准 Transformer 块，包含：
    - Multi-Head Self-Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connection
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        ff_dim: 前馈网络隐藏维度
        dropout: Dropout 率
        activation: 激活函数类型
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        
        # Multi-Head Self-Attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Feed-Forward Network
        self.ff_linear1 = nn.Linear(embed_dim, self.ff_dim)
        self.ff_linear2 = nn.Linear(self.ff_dim, embed_dim)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        
        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
            
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: 注意力掩码 [batch, seq_len] 或 [batch, seq_len, seq_len]
            
        Returns:
            [batch, seq_len, embed_dim]
        """
        # Self-Attention with Pre-Norm
        residual = x
        x = self.norm1(x)
        x = self._attention(x, mask)
        x = residual + x
        
        # Feed-Forward with Pre-Norm
        residual = x
        x = self.norm2(x)
        x = self._feed_forward(x)
        x = residual + x
        
        return x
    
    def _attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Multi-Head Self-Attention"""
        batch_size, seq_len, _ = x.shape
        
        # 投影 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置 [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                mask = mask.unsqueeze(1)
                
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权求和
        out = torch.matmul(attn_weights, v)
        
        # 恢复形状
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(out)
    
    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-Forward Network"""
        x = self.ff_linear1(x)
        x = self.activation(x)
        x = self.ff_dropout(x)
        x = self.ff_linear2(x)
        return x


class ClauseTower(nn.Module):
    """
    Clause Tower
    
    处理 VSM 的行视角，学习子句级别的表示。
    
    Args:
        embed_dim: 嵌入维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        ff_dim: 前馈网络隐藏维度
        dropout: Dropout 率
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Transformer 层
        self.layers = nn.ModuleList([
            ClauseTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 最终 Layer Norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
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
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        获取指定层的注意力权重（用于可视化）
        
        Args:
            x: 输入张量
            mask: 掩码
            layer_idx: 层索引
            
        Returns:
            注意力权重
        """
        # 前向传播到指定层
        for i, layer in enumerate(self.layers):
            if i == layer_idx or (layer_idx == -1 and i == len(self.layers) - 1):
                # 计算注意力权重
                batch_size, seq_len, _ = x.shape
                
                x_normed = layer.norm1(x)
                q = layer.q_proj(x_normed)
                k = layer.k_proj(x_normed)
                
                q = q.view(batch_size, seq_len, layer.num_heads, layer.head_dim)
                k = k.view(batch_size, seq_len, layer.num_heads, layer.head_dim)
                
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) * layer.scale
                
                if mask is not None:
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(~mask, float('-inf'))
                    
                return F.softmax(scores, dim=-1)
            else:
                x = layer(x, mask)
                
        return None


class ClauseEmbedding(nn.Module):
    """
    子句嵌入模块
    
    将子句中的 literal 嵌入并聚合。
    """
    
    def __init__(
        self,
        max_vars: int,
        embed_dim: int,
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        self.max_vars = max_vars
        self.embed_dim = embed_dim
        self.aggregation = aggregation
        
        # Literal 嵌入（+1, -1, 0 三种状态）
        self.literal_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        
        # 变量位置嵌入
        self.var_position_embed = nn.Embedding(max_vars, embed_dim)
        
        # 聚合投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, vsm_row: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vsm_row: VSM 的一行 [batch, max_vars]，值为 -1, 0, 1
            
        Returns:
            子句嵌入 [batch, embed_dim]
        """
        batch_size = vsm_row.shape[0]
        
        # 转换为 embedding 索引 (0: padding, 1: negative, 2: positive)
        indices = (vsm_row + 1).long()  # -1 -> 0, 0 -> 1, 1 -> 2
        
        # Literal 嵌入
        lit_embed = self.literal_embed(indices)  # [batch, max_vars, embed_dim]
        
        # 添加位置嵌入
        positions = torch.arange(self.max_vars, device=vsm_row.device)
        pos_embed = self.var_position_embed(positions)
        lit_embed = lit_embed + pos_embed
        
        # 聚合
        if self.aggregation == 'mean':
            mask = (vsm_row != 0).unsqueeze(-1).float()
            clause_embed = (lit_embed * mask).sum(dim=1)
            clause_embed = clause_embed / mask.sum(dim=1).clamp(min=1)
        elif self.aggregation == 'max':
            clause_embed, _ = lit_embed.max(dim=1)
        elif self.aggregation == 'sum':
            clause_embed = lit_embed.sum(dim=1)
        else:
            clause_embed = lit_embed.mean(dim=1)
            
        return self.output_proj(clause_embed)

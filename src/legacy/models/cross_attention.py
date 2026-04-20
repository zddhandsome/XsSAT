"""
Cross-Attention Module

实现 Clause Tower 和 Variable Tower 之间的跨塔注意力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer

    单向跨塔注意力：query 来自一个塔，key-value 来自另一个塔。

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
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer Norm 和 Dropout
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Query 张量 [batch, q_len, embed_dim]
            key_value: Key-Value 张量 [batch, kv_len, embed_dim]
            query_mask: Query 掩码 [batch, q_len]
            kv_mask: Key-Value 掩码 [batch, kv_len]
            sparse_mask: 稀疏连接掩码 [batch, q_len, kv_len] — True 表示有连接

        Returns:
            注意力输出 [batch, q_len, embed_dim]
        """
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key_value.shape

        # Pre-Norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        # 投影
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # 重塑为多头格式
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)  # [batch, heads, q_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用掩码
        if kv_mask is not None:
            # [batch, kv_len] -> [batch, 1, 1, kv_len]
            kv_mask = kv_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~kv_mask, float('-inf'))

        # 应用稀疏连接掩码
        if sparse_mask is not None:
            # sparse_mask: [B, q_len, kv_len] -> [B, 1, q_len, kv_len]
            scores = scores.masked_fill(~sparse_mask.unsqueeze(1), float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)  # padding 行全 -inf 时防 NaN
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        out = torch.matmul(attn_weights, v)

        # 恢复形状
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, q_len, self.embed_dim)

        return self.out_proj(out)


class ClauseSelfAttentionBlock(nn.Module):
    """
    Single Transformer Encoder block for clause-to-clause self-attention.

    Pre-LayerNorm, Multi-Head Self-Attention, FFN with GELU.
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        ff_dim = embed_dim * 4

        # Pre-LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, num_clauses, embed_dim]
            mask: [B, num_clauses] bool — True for valid clauses
        Returns:
            [B, num_clauses, embed_dim]
        """
        # Self-Attention with Pre-Norm
        residual = x
        x_norm = self.norm1(x)
        x = residual + self._attention(x_norm, mask)

        # FFN with Pre-Norm
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ff(x_norm)

        return x

    def _attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: [B, S] -> [B, 1, 1, S]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)


class BidirectionalCrossAttention(nn.Module):
    """
    双向跨塔注意力

    同时计算 Clause -> Variable 和 Variable -> Clause 的注意力，
    实现两个塔之间的信息融合。

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        num_layers: 交叉注意力层数
        dropout: Dropout 率
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers

        # Clause -> Variable 注意力层
        self.clause_to_var_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Variable -> Clause 注意力层
        self.var_to_clause_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Feed-Forward 层
        self.clause_ff = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.var_ff = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        clause_embeddings: torch.Tensor,
        var_embeddings: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None,
        var_mask: Optional[torch.Tensor] = None,
        vsm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clause_embeddings: 子句嵌入 [batch, num_clauses, embed_dim]
            var_embeddings: 变量嵌入 [batch, num_vars, embed_dim]
            clause_mask: 子句掩码 [batch, num_clauses]
            var_mask: 变量掩码 [batch, num_vars]
            vsm: Variable Space Matrix [batch, num_clauses, num_vars] — 用于构建稀疏掩码

        Returns:
            更新后的 (clause_embeddings, var_embeddings)
        """
        # 从 VSM 构建稀疏连接掩码
        clause_to_var_mask = None
        var_to_clause_mask = None
        if vsm is not None:
            vsm_mask = (vsm != 0)                        # [B, num_clauses, num_vars]
            clause_to_var_mask = vsm_mask                 # clause 查询 → var 键值
            var_to_clause_mask = vsm_mask.transpose(1, 2) # var 查询 → clause 键值

        for i in range(self.num_layers):
            # Clause -> Variable: Variable 作为 query，Clause 作为 kv
            var_update = self.clause_to_var_layers[i](
                var_embeddings, clause_embeddings,
                query_mask=var_mask, kv_mask=clause_mask,
                sparse_mask=var_to_clause_mask
            )
            var_embeddings = var_embeddings + var_update
            var_embeddings = var_embeddings + self.var_ff[i](var_embeddings)

            # Variable -> Clause: Clause 作为 query，Variable 作为 kv
            clause_update = self.var_to_clause_layers[i](
                clause_embeddings, var_embeddings,
                query_mask=clause_mask, kv_mask=var_mask,
                sparse_mask=clause_to_var_mask
            )
            clause_embeddings = clause_embeddings + clause_update
            clause_embeddings = clause_embeddings + self.clause_ff[i](clause_embeddings)

        return clause_embeddings, var_embeddings


class IterativeMessagePassing(nn.Module):
    """
    NeuroSAT 风格迭代消息传递模块。

    使用共享权重的 cross-attention + self-attention 循环 T 次，
    实现类似布尔约束传播的迭代推理行为。通过 LSTM 门控保证
    长迭代下的梯度稳定性。

    不增加参数量（相比多层独立参数），但大幅增加推理深度。

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        num_iterations: 迭代次数 T（可在推理时覆盖）
        dropout: Dropout 率
        use_lstm_gate: 是否使用 LSTM 门控（关闭则退化为简单残差）
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_iterations: int = 8,
        dropout: float = 0.1,
        use_lstm_gate: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_iterations = num_iterations
        self.use_lstm_gate = use_lstm_gate

        # 共享权重：各 1 份
        self.clause_to_var_attn = CrossAttentionLayer(embed_dim, num_heads, dropout)
        self.var_to_clause_attn = CrossAttentionLayer(embed_dim, num_heads, dropout)

        # Feed-Forward（共享权重）
        self.clause_ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.var_ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # Clause self-attention（共享权重）
        self.clause_self_attn = ClauseSelfAttentionBlock(embed_dim, num_heads, dropout)

        # LSTM 门控
        if use_lstm_gate:
            self.clause_lstm = nn.LSTMCell(embed_dim, embed_dim)
            self.var_lstm = nn.LSTMCell(embed_dim, embed_dim)

        # 最终 LayerNorm
        self.clause_final_norm = nn.LayerNorm(embed_dim)
        self.var_final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        clause_emb: torch.Tensor,
        var_emb: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None,
        var_mask: Optional[torch.Tensor] = None,
        vsm: Optional[torch.Tensor] = None,
        num_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clause_emb: 子句嵌入 [B, C, D]
            var_emb: 变量嵌入 [B, V, D]
            clause_mask: 子句掩码 [B, C]
            var_mask: 变量掩码 [B, V]
            vsm: Variable Space Matrix [B, C, V]
            num_iterations: 覆盖默认迭代次数（推理时外推用）

        Returns:
            (clause_state, var_state) 各 [B, *, D]
        """
        T = num_iterations if num_iterations is not None else self.num_iterations
        B, C, D = clause_emb.shape
        _, V, _ = var_emb.shape

        # 从 VSM 构建稀疏掩码
        clause_to_var_mask = None
        var_to_clause_mask = None
        if vsm is not None:
            vsm_mask = (vsm != 0)                         # [B, C, V]
            clause_to_var_mask = vsm_mask                  # clause→var
            var_to_clause_mask = vsm_mask.transpose(1, 2)  # var→clause

        # 初始化工作状态
        clause_state = clause_emb
        var_state = var_emb

        # 初始化 LSTM 隐状态
        if self.use_lstm_gate:
            clause_h = clause_emb.reshape(B * C, D)
            clause_c = torch.zeros_like(clause_h)
            var_h = var_emb.reshape(B * V, D)
            var_c = torch.zeros_like(var_h)

        for t in range(T):
            # 1. clause→var cross-attention: var 作为 query，clause 作为 kv
            var_update = self.clause_to_var_attn(
                var_state, clause_state,
                query_mask=var_mask, kv_mask=clause_mask,
                sparse_mask=var_to_clause_mask
            )
            if self.use_lstm_gate:
                var_h, var_c = self.var_lstm(
                    var_update.reshape(B * V, D), (var_h, var_c)
                )
                var_state = var_h.reshape(B, V, D)
            else:
                var_state = var_state + var_update
            var_state = var_state + self.var_ff(var_state)

            # 2. var→clause cross-attention: clause 作为 query，var 作为 kv
            clause_update = self.var_to_clause_attn(
                clause_state, var_state,
                query_mask=clause_mask, kv_mask=var_mask,
                sparse_mask=clause_to_var_mask
            )
            if self.use_lstm_gate:
                clause_h, clause_c = self.clause_lstm(
                    clause_update.reshape(B * C, D), (clause_h, clause_c)
                )
                clause_state = clause_h.reshape(B, C, D)
            else:
                clause_state = clause_state + clause_update
            clause_state = clause_state + self.clause_ff(clause_state)

            # 3. clause self-attention
            clause_state = self.clause_self_attn(clause_state, clause_mask)

        return self.clause_final_norm(clause_state), self.var_final_norm(var_state)

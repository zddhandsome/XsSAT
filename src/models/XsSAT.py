"""
XsSAT: 基于多通道 VSM、共享步进 axial 推理和 hard-clause 读出的 SAT 求解架构。

1. 结构增强的 CellEmbedding
2. 对齐式 NegationOperator
3. 共享参数的 Recurrent Axial Backbone
4. ClauseGlobalTokenLayer + PolarityPairMixer
5. HardClauseReadout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional


# ============================================================
# SDPA 注意力
# ============================================================

class SDPAttention(nn.Module):
    """
    Parameter names match nn.MultiheadAttention exactly:
      in_proj_weight [3D, D], in_proj_bias [3D], out_proj Linear(D, D)
    so existing checkpoints load without any key remapping.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 batch_first: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Fused QKV projection (single matmul instead of 3)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]  (batch_first)
            key_padding_mask: [B, S] bool, True = ignore this position
        Returns:
            out: [B, S, D]
        """
        B, S, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Fused QKV: [B, S, 3D]
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.unflatten(-1, (3, H, d)).permute(2, 0, 3, 1, 4).unbind(0)
        # q, k, v: [B, H, S, d]

        # Convert key_padding_mask to SDPA-compatible float attn_mask
        # Use finfo.min (finite) instead of -inf to avoid NaN when
        # fp16 attention scores overflow to +inf  (+inf + -inf = nan)
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, S], True = masked (ignored)
            # SDPA float mask: added to scores, so masked positions get large negative
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S] broadcasts to [B,H,S,S]
            attn_mask = torch.zeros(B, 1, 1, S, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), torch.finfo(q.dtype).min)

        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_p)
        # out: [B, H, S, d]

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# ============================================================
# 组件 A: 可学习否定算子
# ============================================================

class NegationOperator(nn.Module):
    """
    可学习否定算子，满足对合约束 N(N(x)) ≈ x

    将布尔逻辑的否定 ¬ 嵌入为连续空间中的线性变换。
    通过训练损失约束 N² ≈ I，保证逻辑语义正确性。

    Args:
        embed_dim: 嵌入维度
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(embed_dim))
        self.b = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., D] 任意前缀形状的嵌入
        Returns:
            N(x): [..., D]
        """
        return x @ self.W + self.b

    def involution_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        对合约束损失: ||N(N(x)) - x||²

        Args:
            x: [..., D]
        Returns:
            标量损失
        """
        nx = self.forward(x)
        nnx = self.forward(nx)
        return F.mse_loss(nnx, x)

    def consistency_loss(
        self, emb_pos: torch.Tensor, emb_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        否定一致性损失: ||N(emb_pos) - emb_neg||²

        Args:
            emb_pos: 正literal嵌入 [..., D]
            emb_neg: 负literal嵌入 [..., D]
        Returns:
            标量损失
        """
        n_pos = self.forward(emb_pos)
        return F.mse_loss(n_pos, emb_neg)


# ============================================================
# 组件 B: Cell级嵌入 + 对偶位置编码
# ============================================================

class CellEmbedding(nn.Module):
    """
    Cell级嵌入模块

    对VSM中每个 (clause_i, var_j) 位置:
      raw = [ch0[i,j], ch1[i,j]]       # 2维
      cell_emb = Linear(raw)            # → D维
      cell_emb += base_embed(j)         # 变量身份
      cell_emb += polarity_sign * offset # 极性方向

    其中 polarity_sign 根据该位置是正literal还是负literal决定。

    Args:
        max_vars: 最大变量数
        embed_dim: 嵌入维度
    """

    def __init__(
        self,
        max_vars: int,
        embed_dim: int,
        clause_short_threshold: int = 4,
    ):
        super().__init__()
        self.max_vars = max_vars
        self.embed_dim = embed_dim
        self.clause_short_threshold = max(int(clause_short_threshold), 1)

        self.cell_proj = nn.Linear(2, embed_dim)

        # 变量身份嵌入 (可学习)
        self.var_identity = nn.Embedding(max_vars, embed_dim)

        self.polarity_offset = nn.Parameter(torch.randn(embed_dim) * 0.02)

        hidden_dim = max(embed_dim // 2, 16)
        self.row_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.col_proj = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.structural_scale = nn.Parameter(torch.tensor(1.0))

        self.norm = nn.LayerNorm(embed_dim)

    def _build_structural_features(
        self,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        clause_mask: torch.Tensor,
        var_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clause_mask_f = clause_mask.float()
        var_mask_f = var_mask.float()

        clause_count = clause_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        var_count = var_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

        literal_mask = pos_mask | neg_mask
        clause_len = literal_mask.sum(dim=2).float() * clause_mask_f
        pos_count = pos_mask.sum(dim=2).float() * clause_mask_f
        neg_count = neg_mask.sum(dim=2).float() * clause_mask_f
        clause_den = clause_len.clamp(min=1.0)

        row_features = torch.stack(
            [
                clause_len / var_count,
                pos_count / clause_den,
                neg_count / clause_den,
                ((clause_len > 0) & (clause_len <= self.clause_short_threshold)).float(),
            ],
            dim=-1,
        )
        row_features = row_features * clause_mask_f.unsqueeze(-1)

        pos_freq = pos_mask.sum(dim=1).float() * var_mask_f
        neg_freq = neg_mask.sum(dim=1).float() * var_mask_f
        total_freq = pos_freq + neg_freq
        total_den = total_freq.clamp(min=1.0)

        col_features = torch.stack(
            [
                total_freq / clause_count,
                pos_freq / clause_count,
                neg_freq / clause_count,
                (pos_freq - neg_freq).abs() / total_den,
                (((pos_freq == 0) ^ (neg_freq == 0)) & (total_freq > 0)).float(),
            ],
            dim=-1,
        )
        col_features = col_features * var_mask_f.unsqueeze(-1)

        return row_features, col_features

    def forward(
        self,
        vsm: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None,
        var_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            vsm: [B, 2, C, V] 多通道VSM

        Returns:
            cell_emb: [B, C, V, D] 嵌入网格
            cell_mask: [B, C, V] bool, 非零位置为True
            pos_mask: [B, C, V] 正literal位置
            neg_mask: [B, C, V] 负literal位置
        """
        B, _, C, V = vsm.shape
        if clause_mask is None:
            clause_mask = torch.ones(B, C, dtype=torch.bool, device=vsm.device)
        if var_mask is None:
            var_mask = torch.ones(B, V, dtype=torch.bool, device=vsm.device)

        valid_grid = clause_mask.unsqueeze(-1) & var_mask.unsqueeze(1)
        valid_grid_f = valid_grid.to(vsm.dtype)

        ch0 = vsm[:, 0] * valid_grid_f  # [B, C, V] 正literal
        ch1 = vsm[:, 1] * valid_grid_f  # [B, C, V] 负literal

        # 正/负literal位置 mask
        pos_mask = (ch0 > 0) & valid_grid
        neg_mask = (ch1 > 0) & valid_grid
        cell_mask = pos_mask | neg_mask

        raw = torch.stack([ch0, ch1], dim=-1)  # [B, C, V, 2]

        # This small-input projection can hit cublasLt fp16 heuristic failures on some GPUs.
        # Keep it in fp32 and let later layers use autocast as usual.
        with autocast(enabled=False):
            cell_emb = self.cell_proj(raw.float())  # [B, C, V, D]

        # 变量身份嵌入
        var_idx = torch.arange(V, device=vsm.device)  # [V]
        var_emb = self.var_identity(var_idx)  # [V, D]
        cell_emb = cell_emb + var_emb.unsqueeze(0).unsqueeze(0)  # broadcast

        row_features, col_features = self._build_structural_features(
            pos_mask,
            neg_mask,
            clause_mask,
            var_mask,
        )
        row_emb = self.row_proj(row_features).unsqueeze(2)
        col_emb = self.col_proj(col_features).unsqueeze(1)
        cell_emb = cell_emb + self.structural_scale * (row_emb + col_emb)

        # 对偶位置编码: 正literal位置 +offset, 负literal位置 -offset
        polarity_sign = ch0 - ch1  # [B, C, V], 值为 +1, -1, 0
        cell_emb = cell_emb + polarity_sign.unsqueeze(-1) * self.polarity_offset

        cell_emb = self.norm(cell_emb)
        cell_emb = cell_emb * valid_grid.unsqueeze(-1).float()

        return cell_emb, cell_mask, pos_mask, neg_mask


# ============================================================
# 组件 C: 稀疏轴向注意力
# ============================================================

class AxialAttentionBlock(nn.Module):
    """
    单个轴向注意力块: 多头自注意力 + 残差 + LN + FFN + 残差 + LN

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        ffn_ratio: FFN扩展比
        dropout: Dropout率
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SDPAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, D]
            key_padding_mask: [batch, seq_len] bool, True=忽略该位置
        Returns:
            [batch, seq_len, D]
        """
        # 处理全mask的行: 如果某行所有位置都被mask, 取消mask以避免NaN
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)  # [B, 1]
            key_padding_mask = key_padding_mask & ~all_masked  # 全mask行→全不mask

        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))

        return x


class AxialAttentionLayer(nn.Module):
    """
    一层轴向注意力 = Row Attention + Column Attention

    Row: 对每行(clause)内的变量位置做注意力 → clause内literal交互
    Column: 对每列(variable)跨clause做注意力 → variable跨clause交互

    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        ffn_ratio: FFN扩展比
        dropout: Dropout率
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.row_attn = AxialAttentionBlock(embed_dim, num_heads, ffn_ratio, dropout)
        self.col_attn = AxialAttentionBlock(embed_dim, num_heads, ffn_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, V, D] 嵌入网格 (已截断到有效范围)
            cell_mask: [B, C, V] bool, True=有效位置

        Returns:
            [B, C, V, D] 更新后的嵌入
        """
        B, C, V, D = x.shape

        # === Row Attention: 对每行做注意力 ===
        x_row = x.reshape(B * C, V, D)
        row_mask = ~cell_mask.reshape(B * C, V)  # True=忽略
        x_row = self.row_attn(x_row, key_padding_mask=row_mask)
        x = x_row.reshape(B, C, V, D)

        # === Column Attention: 对每列做注意力 ===
        x_col = x.permute(0, 2, 1, 3).reshape(B * V, C, D)
        col_mask = ~cell_mask.permute(0, 2, 1).reshape(B * V, C)
        x_col = self.col_attn(x_col, key_padding_mask=col_mask)
        x = x_col.reshape(B, V, C, D).permute(0, 2, 1, 3)

        return x


class RecurrentCellUpdate(nn.Module):
    """使用门控残差在多步 rollout 之间更新 cell 状态。"""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(embed_dim // 2, 16)
        self.prev_norm = nn.LayerNorm(embed_dim)
        self.new_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        prev_state: torch.Tensor,
        new_state: torch.Tensor,
        cell_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prev_norm = self.prev_norm(prev_state)
        new_norm = self.new_norm(new_state)
        delta = new_norm - prev_norm

        gate = torch.sigmoid(self.gate(torch.cat([prev_norm, new_norm, delta], dim=-1)))
        updated = prev_state + gate * (new_state - prev_state)
        updated = self.out_norm(updated)

        if cell_mask is not None:
            updated = torch.where(cell_mask.unsqueeze(-1), updated, prev_state)

        return updated


class ClauseGlobalTokenLayer(nn.Module):
    """在 clause 维度上插入一个全局 token，与所有 clause 做交互后再写回 cell 网格。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        writeback_scale: float = 0.1,
    ):
        super().__init__()
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.block = AxialAttentionBlock(embed_dim, num_heads, ffn_ratio, dropout)
        self.writeback_scale = writeback_scale

    def forward(
        self,
        x: torch.Tensor,
        cell_mask: torch.Tensor,
        clause_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, V, D]
            cell_mask: [B, C, V] bool
            clause_mask: [B, C] bool
        Returns:
            [B, C, V, D]
        """
        cell_mask_f = cell_mask.float()
        clause_den = cell_mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        clause_emb = (x * cell_mask_f.unsqueeze(-1)).sum(dim=2) / clause_den

        clause_mask_f = clause_mask.float().unsqueeze(-1)
        clause_count = clause_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled_clause = (clause_emb * clause_mask_f).sum(dim=1, keepdim=True) / clause_count

        token = self.global_token.expand(x.shape[0], -1, -1) + pooled_clause
        seq = torch.cat([token, clause_emb], dim=1)

        token_mask = torch.zeros(
            x.shape[0], 1, dtype=torch.bool, device=x.device
        )
        seq_mask = torch.cat([token_mask, ~clause_mask], dim=1)
        seq = self.block(seq, key_padding_mask=seq_mask)

        clause_delta = (seq[:, 1:, :] - clause_emb) * clause_mask_f
        writeback = clause_delta.unsqueeze(2) * cell_mask_f.unsqueeze(-1)
        return x + self.writeback_scale * writeback


# ============================================================
# 组件 D: Hard-clause 读出
# ============================================================


class TokenAttentionPool(nn.Module):
    """对一组 token 做轻量注意力池化。"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            mask: [B, T] bool
        Returns:
            pooled: [B, D]
        """
        x_norm = self.norm(x)
        logits = self.score(x_norm).squeeze(-1)

        safe_mask = mask.clone()
        all_masked = ~safe_mask.any(dim=1)
        if all_masked.any():
            safe_mask[all_masked, 0] = True

        logits = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=1)
        attn = attn * safe_mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (x * attn.unsqueeze(-1)).sum(dim=1)


class PolarityPairMixer(nn.Module):
    """
    显式建模同一变量正/负 literal 的列级关系，并写回 cell 网格。
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        writeback_scale: float = 0.1,
    ):
        super().__init__()
        self.writeback_scale = writeback_scale
        self.pair_norm = nn.LayerNorm(embed_dim * 4)
        self.delta_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        cell_emb: torch.Tensor,
        vsm: torch.Tensor,
        var_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            cell_emb: [B, C, V, D]
            vsm: [B, 2, C, V]
            var_mask: [B, V] bool
        Returns:
            updated cell_emb: [B, C, V, D]
        """
        pos_mask = vsm[:, 0] > 0  # [B, C, V]
        neg_mask = vsm[:, 1] > 0  # [B, C, V]

        pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)  # [B, V]
        neg_count = neg_mask.float().sum(dim=1).clamp(min=1.0)  # [B, V]

        pos_emb = (
            cell_emb * pos_mask.unsqueeze(-1).float()
        ).sum(dim=1) / pos_count.unsqueeze(-1)  # [B, V, D]
        neg_emb = (
            cell_emb * neg_mask.unsqueeze(-1).float()
        ).sum(dim=1) / neg_count.unsqueeze(-1)  # [B, V, D]

        pair_feat = torch.cat(
            [pos_emb, neg_emb, pos_emb - neg_emb, pos_emb * neg_emb],
            dim=-1,
        )  # [B, V, 4D]
        pair_feat = self.pair_norm(pair_feat)

        pair_delta = self.delta_mlp(pair_feat)  # [B, V, D]
        pair_gate = torch.sigmoid(self.gate_mlp(pair_feat))  # [B, V, 1]
        pair_delta = pair_delta * pair_gate

        if var_mask is not None:
            pair_delta = pair_delta * var_mask.unsqueeze(-1).float()

        pair_delta = pair_delta.unsqueeze(1)  # [B, 1, V, D]
        update = (
            pos_mask.unsqueeze(-1).float() - neg_mask.unsqueeze(-1).float()
        ) * pair_delta

        cell_emb = cell_emb + self.writeback_scale * update
        return self.out_norm(cell_emb)


class HardClauseReadout(nn.Module):
    """面向 hardest clauses 的 clause-centric 读出。"""

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        topk_ratio: float = 0.1,
        min_topk: int = 8,
        detach_core_backbone: bool = False,
    ):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk
        self.detach_core_backbone = detach_core_backbone

        self.clause_token_pool = TokenAttentionPool(embed_dim)
        self.var_token_pool = TokenAttentionPool(embed_dim)
        self.clause_global_pool = TokenAttentionPool(embed_dim)
        self.var_global_pool = TokenAttentionPool(embed_dim)

        self.clause_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.core_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        self.clause_norm = nn.LayerNorm(embed_dim)
        self.var_norm = nn.LayerNorm(embed_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + 4, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        cell_emb: torch.Tensor,
        cell_mask: torch.Tensor,
        clause_mask: torch.Tensor,
        var_mask: torch.Tensor,
        vsm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del vsm

        B, C, V, D = cell_emb.shape

        clause_tokens = self.clause_token_pool(
            cell_emb.reshape(B * C, V, D),
            cell_mask.reshape(B * C, V),
        ).reshape(B, C, D)

        var_tokens = self.var_token_pool(
            cell_emb.permute(0, 2, 1, 3).reshape(B * V, C, D),
            cell_mask.permute(0, 2, 1).reshape(B * V, C),
        ).reshape(B, V, D)

        clause_scores_raw = self.clause_head(clause_tokens).squeeze(-1)  # [B, C]
        masked_clause_scores = clause_scores_raw.masked_fill(~clause_mask, float('-inf'))

        valid_clause_count = clause_mask.sum(dim=1).clamp(min=1)
        desired_topk = torch.ceil(
            valid_clause_count.float() * self.topk_ratio
        ).long().clamp(min=1)
        desired_topk = torch.maximum(
            desired_topk,
            torch.full_like(desired_topk, self.min_topk),
        )
        desired_topk = torch.minimum(desired_topk, valid_clause_count)
        k_max = int(desired_topk.max().item())

        topk_scores, topk_idx = torch.topk(masked_clause_scores, k=k_max, dim=1)
        topk_tokens = torch.gather(
            clause_tokens,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # [B, k_max, D]

        topk_valid = (
            torch.arange(k_max, device=cell_emb.device).unsqueeze(0)
            < desired_topk.unsqueeze(1)
        )
        z_clause = self.clause_global_pool(topk_tokens, topk_valid)
        z_var = self.var_global_pool(var_tokens, var_mask)

        topk_scores_zero = topk_scores.masked_fill(~topk_valid, 0.0)
        topk_count = topk_valid.float().sum(dim=1).clamp(min=1.0)
        clause_score_mean = topk_scores_zero.sum(dim=1) / topk_count
        clause_score_min = topk_scores.masked_fill(~topk_valid, float('inf')).min(dim=1).values
        clause_score_max = topk_scores.masked_fill(~topk_valid, float('-inf')).max(dim=1).values
        clause_score_std = torch.sqrt(
            (
                ((topk_scores_zero - clause_score_mean.unsqueeze(1)) ** 2)
                * topk_valid.float()
            ).sum(dim=1) / topk_count
        )
        clause_score_gap = clause_score_max - clause_score_min

        z_clause_norm = self.clause_norm(z_clause)
        z_var_norm = self.var_norm(z_var)
        diff = torch.abs(z_clause_norm - z_var_norm)
        stats = torch.stack(
            [clause_score_min, clause_score_mean, clause_score_std, clause_score_gap],
            dim=-1,
        )
        fused = torch.cat([z_clause_norm, diff, z_clause_norm * diff, stats], dim=-1)
        sat_logit = self.final_mlp(fused)

        core_input = clause_tokens.detach() if self.detach_core_backbone else clause_tokens
        core_scores = self.core_head(core_input).squeeze(-1)
        clause_vote = clause_score_mean.unsqueeze(-1)

        return {
            'sat_logit': sat_logit,
            'clause_scores': core_scores,
            'clause_vote': clause_vote,
        }


# ============================================================
# 主模型
# ============================================================

class XsSAT(nn.Module):
    """
    XsSAT 主模型

    数据流:
    MC-VSM [B,2,C,V] → Cell嵌入 [B,C,V,D] → 否定算子
    → recurrent axial backbone → HardClauseReadout

    Args:
        max_clauses: 最大子句数
        max_vars: 最大变量数
        embed_dim: 嵌入维度 D
        num_layers: 轴向注意力层数 L
        num_heads: 注意力头数
        ffn_ratio: FFN扩展比
        dropout: Dropout率
        tau_init: 旧版 readout 的兼容字段, 当前实现不会使用
    """

    def __init__(
        self,
        max_clauses: int = 50,
        max_vars: int = 10,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        tau_init: float = 1.0,
        neg_samples: int = 256,
        use_gradient_checkpoint: bool = False,
        use_multichannel_vsm: bool = True,
        use_negation: bool = True,
        attention_type: str = 'axial',
        readout_type: str = 'hard_clause',
        use_polarity_offset: bool = True,
        use_periodic_global_token: bool = True,
        global_token_every_n_layers: int = 2,
        global_token_writeback_scale: float = 0.1,
        use_clause_literal_fusion: bool = False,
        use_multiscale_clause_context: bool = False,
        clause_hierarchy_levels: int = 2,
        clause_hierarchy_window: int = 4,
        clause_context_prototypes: int = 4,
        detach_core_backbone: bool = False,
        use_polarity_pair_mixer: bool = True,
        pair_mixer_every_n_layers: int = 2,
        pair_mixer_writeback_scale: float = 0.1,
        hard_clause_topk_ratio: float = 0.1,
        hard_clause_min_topk: int = 8,
        use_structural_features: bool = True,
        clause_short_threshold: int = 4,
        use_recurrent_axial: bool = True,
        recurrent_steps: int = 4,
        recurrent_base_layers: int = 2,
    ):
        super().__init__()

        # `tau_init` 只服务旧版 soft-min readout，保留该参数仅用于兼容旧 yaml。
        del tau_init, clause_hierarchy_levels, clause_hierarchy_window, clause_context_prototypes
        self._validate_supported_configuration(
            use_multichannel_vsm=use_multichannel_vsm,
            use_negation=use_negation,
            attention_type=attention_type,
            readout_type=readout_type,
            use_polarity_offset=use_polarity_offset,
            use_periodic_global_token=use_periodic_global_token,
            use_clause_literal_fusion=use_clause_literal_fusion,
            use_multiscale_clause_context=use_multiscale_clause_context,
            use_polarity_pair_mixer=use_polarity_pair_mixer,
            use_structural_features=use_structural_features,
            use_recurrent_axial=use_recurrent_axial,
        )

        self.max_clauses = max_clauses
        self.max_vars = max_vars
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.neg_samples = neg_samples
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.global_token_every_n_layers = max(int(global_token_every_n_layers), 1)
        self.pair_mixer_every_n_layers = max(int(pair_mixer_every_n_layers), 1)
        self.recurrent_steps = max(int(recurrent_steps), 1)

        self.cell_embedding = CellEmbedding(
            max_vars, embed_dim,
            clause_short_threshold=clause_short_threshold,
        )

        self.pair_mixer = PolarityPairMixer(
            embed_dim=embed_dim,
            dropout=dropout,
            writeback_scale=pair_mixer_writeback_scale,
        )

        self.negation_op = NegationOperator(embed_dim)

        base_layer_count = max(int(recurrent_base_layers), 1)
        base_layer_count = min(base_layer_count, num_layers)
        self.axial_layers = nn.ModuleList([
            AxialAttentionLayer(embed_dim, num_heads, ffn_ratio, dropout)
            for _ in range(base_layer_count)
        ])
        self.recurrent_update = RecurrentCellUpdate(embed_dim, dropout)

        self.periodic_global_layer_ids = []
        self.global_clause_layers = nn.ModuleList()
        for layer_idx in range(len(self.axial_layers)):
            if (layer_idx + 1) % self.global_token_every_n_layers == 0:
                self.periodic_global_layer_ids.append(layer_idx)
                self.global_clause_layers.append(
                    ClauseGlobalTokenLayer(
                        embed_dim,
                        num_heads,
                        ffn_ratio,
                        dropout,
                        writeback_scale=global_token_writeback_scale,
                    )
                )

        self.final_norm = nn.LayerNorm(embed_dim)

        self.readout = HardClauseReadout(
            embed_dim,
            dropout,
            topk_ratio=hard_clause_topk_ratio,
            min_topk=hard_clause_min_topk,
            detach_core_backbone=detach_core_backbone,
        )

        self._init_weights()

    @staticmethod
    def _validate_supported_configuration(
        *,
        use_multichannel_vsm: bool,
        use_negation: bool,
        attention_type: str,
        readout_type: str,
        use_polarity_offset: bool,
        use_periodic_global_token: bool,
        use_clause_literal_fusion: bool,
        use_multiscale_clause_context: bool,
        use_polarity_pair_mixer: bool,
        use_structural_features: bool,
        use_recurrent_axial: bool,
    ) -> None:
        if not use_multichannel_vsm:
            raise ValueError("XsSAT.py now only supports the multichannel VSM path used by sr10_40_vsm_plus.yaml.")
        if not use_negation:
            raise ValueError("XsSAT.py now requires `use_negation=True` to match sr10_40_vsm_plus.yaml.")
        if attention_type != 'axial':
            raise ValueError("XsSAT.py now only supports `attention_type='axial'`.")
        if readout_type != 'hard_clause':
            raise ValueError("XsSAT.py now only supports `readout_type='hard_clause'`.")
        if not use_polarity_offset:
            raise ValueError("XsSAT.py now requires `use_polarity_offset=True`.")
        if not use_periodic_global_token:
            raise ValueError("XsSAT.py now requires `use_periodic_global_token=True`.")
        if use_clause_literal_fusion:
            raise ValueError("Clause-literal fusion was removed with the old semantic readout path.")
        if use_multiscale_clause_context:
            raise ValueError("Multiscale clause context was removed with the old semantic readout path.")
        if not use_polarity_pair_mixer:
            raise ValueError("XsSAT.py now requires `use_polarity_pair_mixer=True`.")
        if not use_structural_features:
            raise ValueError("XsSAT.py now requires `use_structural_features=True`.")
        if not use_recurrent_axial:
            raise ValueError("XsSAT.py now requires `use_recurrent_axial=True`.")

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        vsm: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None,
        var_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            vsm: 多通道VSM [B, 2, max_clauses, max_vars]
            clause_mask: [B, max_clauses] bool
            var_mask: [B, max_vars] bool

        Returns:
            dict with:
                sat_pred: [B, 1] SAT/UNSAT logit
                clause_scores: [B, C] clause投票分数
                negation_loss: 标量, 否定算子约束损失
        """
        B = vsm.shape[0]

        # 默认mask
        if clause_mask is None:
            clause_mask = torch.ones(B, self.max_clauses, dtype=torch.bool, device=vsm.device)
        if var_mask is None:
            var_mask = torch.ones(B, self.max_vars, dtype=torch.bool, device=vsm.device)

        # === Cell嵌入 ===
        cell_emb, cell_mask, pos_mask, neg_mask = self.cell_embedding(
            vsm,
            clause_mask=clause_mask,
            var_mask=var_mask,
        )
        # cell_emb: [B, C, V, D]
        # cell_mask: [B, C, V]

        # === 否定算子: 仅对负literal位置施加 N(·) ===
        if neg_mask.any():
            negation_loss = self._compute_negation_loss(cell_emb, cell_mask, pos_mask, neg_mask, var_mask)
            neg_embs = cell_emb[neg_mask]  # [N_neg, D]
            cell_emb[neg_mask] = self.negation_op(neg_embs)
        else:
            negation_loss = torch.tensor(0.0, device=vsm.device)

        # === 稀疏轴向注意力 ===
        B, C_full, V_full, D = cell_emb.shape

        # 动态截断: 只计算一次，避免每层重复 .item() 导致 CPU-GPU 同步
        clause_any = cell_mask.any(dim=2)  # [B, C]
        var_any = cell_mask.any(dim=1)     # [B, V]
        C_eff = max(int(clause_any.any(dim=0).sum().item()), 1)
        V_eff = max(int(var_any.any(dim=0).sum().item()), 1)

        # 截断到有效范围
        cell_emb = cell_emb[:, :C_eff, :V_eff, :].contiguous()
        cell_mask_trimmed = cell_mask[:, :C_eff, :V_eff].contiguous()
        clause_mask_trimmed = (clause_mask[:, :C_eff] & clause_any[:, :C_eff]).contiguous()
        var_mask_trimmed = var_mask[:, :V_eff].contiguous()
        vsm_trimmed = vsm[:, :, :C_eff, :V_eff].contiguous()

        for _ in range(self.recurrent_steps):
            prev_state = cell_emb
            cell_emb = self._run_axial_stack(
                cell_emb,
                cell_mask_trimmed,
                clause_mask_trimmed,
                var_mask_trimmed,
                vsm_trimmed,
            )
            if self.use_gradient_checkpoint and self.training:
                cell_emb = checkpoint(
                    self.recurrent_update,
                    prev_state,
                    cell_emb,
                    cell_mask_trimmed,
                    use_reentrant=False,
                )
            else:
                cell_emb = self.recurrent_update(prev_state, cell_emb, cell_mask_trimmed)

        # 填充回完整尺寸 (readout 需要完整的 clause/var 维度)
        if C_eff < C_full or V_eff < V_full:
            padded = cell_emb.new_zeros(B, C_full, V_full, D)
            padded[:, :C_eff, :V_eff, :] = cell_emb
            cell_emb = padded

        cell_emb = self.final_norm(cell_emb)

        # === 读出 ===
        readout = self.readout(cell_emb, cell_mask, clause_mask, var_mask, vsm=vsm)

        outputs = {
            'sat_pred': readout['sat_logit'],
            'clause_scores': readout['clause_scores'],
            'negation_loss': negation_loss,
            'clause_vote': readout['clause_vote'],
        }
        return outputs

    def _run_axial_stack(
        self,
        cell_emb: torch.Tensor,
        cell_mask_trimmed: torch.Tensor,
        clause_mask_trimmed: torch.Tensor,
        var_mask_trimmed: torch.Tensor,
        vsm_trimmed: torch.Tensor,
    ) -> torch.Tensor:
        global_layer_ptr = 0
        for layer_idx, layer in enumerate(self.axial_layers):
            if self.use_gradient_checkpoint and self.training:
                cell_emb = checkpoint(layer, cell_emb, cell_mask_trimmed, use_reentrant=False)
            else:
                cell_emb = layer(cell_emb, cell_mask_trimmed)

            if (
                global_layer_ptr < len(self.periodic_global_layer_ids)
                and layer_idx == self.periodic_global_layer_ids[global_layer_ptr]
            ):
                global_layer = self.global_clause_layers[global_layer_ptr]
                if self.use_gradient_checkpoint and self.training:
                    cell_emb = checkpoint(
                        global_layer,
                        cell_emb,
                        cell_mask_trimmed,
                        clause_mask_trimmed,
                        use_reentrant=False,
                    )
                else:
                    cell_emb = global_layer(cell_emb, cell_mask_trimmed, clause_mask_trimmed)
                global_layer_ptr += 1

            if (layer_idx + 1) % self.pair_mixer_every_n_layers == 0:
                if self.use_gradient_checkpoint and self.training:
                    cell_emb = checkpoint(
                        self.pair_mixer,
                        cell_emb,
                        vsm_trimmed,
                        var_mask_trimmed,
                        use_reentrant=False,
                    )
                else:
                    cell_emb = self.pair_mixer(cell_emb, vsm_trimmed, var_mask_trimmed)

        return cell_emb

    @staticmethod
    def _aggregate_variable_embeddings(
        cell_emb: torch.Tensor,
        polarity_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = polarity_mask.float().unsqueeze(-1)
        return (cell_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _compute_negation_loss(
        self,
        cell_emb: torch.Tensor,
        cell_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        var_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算否定算子的训练约束损失

        L_neg = λ₁·||N(emb_pos) - emb_neg||² + λ₂·||N(N(x)) - x||²

        使用同一变量列上的正/负聚合表示构造一致性约束，
        避免随机配对引入语义噪声。
        """
        device = cell_emb.device

        active_emb = cell_emb[cell_mask].detach()
        if active_emb.shape[0] > self.neg_samples:
            idx = torch.randperm(active_emb.shape[0], device=device)[:self.neg_samples]
            active_emb = active_emb[idx]
        involution_loss = (
            self.negation_op.involution_loss(active_emb)
            if active_emb.numel() > 0
            else torch.tensor(0.0, device=device)
        )

        pos_var = self._aggregate_variable_embeddings(cell_emb, pos_mask).detach()
        neg_var = self._aggregate_variable_embeddings(cell_emb, neg_mask).detach()
        pair_mask = (
            pos_mask.any(dim=1)
            & neg_mask.any(dim=1)
            & var_mask
        )

        if pair_mask.any():
            pos_pairs = pos_var[pair_mask]
            neg_pairs = neg_var[pair_mask]
            if pos_pairs.shape[0] > self.neg_samples:
                idx = torch.randperm(pos_pairs.shape[0], device=device)[:self.neg_samples]
                pos_pairs = pos_pairs[idx]
                neg_pairs = neg_pairs[idx]
            consistency_loss = self.negation_op.consistency_loss(pos_pairs, neg_pairs)
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        return involution_loss + consistency_loss

    def predict(
        self,
        vsm: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None,
        var_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """推理预测"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(vsm, clause_mask, var_mask)

            sat_prob = torch.sigmoid(outputs['sat_pred'])
            sat_pred = (sat_prob > 0.5).long()

            core_prob = torch.sigmoid(outputs['clause_scores'])  # 分数越低越可能是Core
            core_pred = (core_prob > threshold).long()

        return {
            'sat_pred': sat_pred,
            'sat_prob': sat_prob,
            'core_pred': core_pred,
            'core_prob': core_prob,
        }

    def get_unsat_core(
        self,
        vsm: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """获取预测的 UNSAT Core"""
        predictions = self.predict(vsm, clause_mask)
        core_prob = predictions['core_prob']

        if top_k is not None:
            _, indices = torch.topk(core_prob, k=top_k, dim=-1)
            return indices
        else:
            return (core_prob > threshold).nonzero(as_tuple=True)

"""
XsSAT Multi-Task Loss Module

损失函数:
  L_total = L_sat + α_core · L_core + α_neg · L_negation

1. SAT分类损失: BCE (二分类)
2. UNSAT Core预测损失: BCE + pos_weight (clause级, 仅对UNSAT样本)
3. 否定算子约束损失: 由模型前向传播直接产生
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SATClassificationLoss(nn.Module):
    """
    SAT/UNSAT 二分类损失

    使用带类别权重的交叉熵损失。
    """

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.label_smoothing = label_smoothing

        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size] or [batch_size, 1]
            targets: [batch_size], 0=UNSAT, 1=SAT
        """
        logits = logits.squeeze(-1)
        targets = targets.float().squeeze(-1)

        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )


class UNSATCoreLoss(nn.Module):
    """
    UNSAT Core 预测损失
    BCEWithLogitsLoss + pos_weight 处理类别不平衡。
    """

    def __init__(
        self,
        pos_weight: float = 6.0,
        **kwargs,
    ):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, max_clauses]
            targets: [batch_size, max_clauses], 1=Core
            mask: [batch_size, max_clauses] bool
        """
        pos_weight = self.pos_weight.to(logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pos_weight, reduction='none'
        )

        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / (mask.float().sum() + 1e-8)
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    XsSAT 多任务联合损失

    L_total = L_sat + α_core · L_core + α_neg · L_negation

    Args:
        alpha_core: UNSAT Core损失权重
        alpha_neg: 否定算子损失权重
        sat_loss_config: SAT损失配置
        clause_loss_config: Clause损失配置
    """

    def __init__(
        self,
        alpha_core: float = 0.2,
        alpha_neg: float = 0.1,
        sat_loss_config: Optional[Dict] = None,
        clause_loss_config: Optional[Dict] = None,
    ):
        super().__init__()

        self.alpha_core = alpha_core
        self.alpha_neg = alpha_neg

        sat_config = sat_loss_config or {}
        clause_config = clause_loss_config or {}

        self.sat_loss = SATClassificationLoss(**sat_config)
        self.clause_loss = UNSATCoreLoss(**clause_config)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: 模型输出 (sat_pred, clause_scores, negation_loss)
            targets: 标签 (sat_label, core_labels, clause_mask)

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}
        device = outputs['sat_pred'].device

        # SAT分类损失
        sat_label = targets.get('sat_label')
        if sat_label is not None:
            l_sat = self.sat_loss(outputs['sat_pred'], sat_label)
        else:
            l_sat = torch.tensor(0.0, device=device)
        loss_dict['sat_loss'] = l_sat

        # UNSAT Core损失 (仅对UNSAT样本计算)
        core_labels = targets.get('core_labels')
        if sat_label is not None and core_labels is not None:
            unsat_mask = (sat_label.squeeze(-1) == 0)
            if unsat_mask.any():
                clause_mask = targets.get('clause_mask')
                if clause_mask is not None:
                    clause_mask = clause_mask[unsat_mask]
                l_core = self.clause_loss(
                    outputs['clause_scores'][unsat_mask],
                    core_labels[unsat_mask],
                    clause_mask
                )
            else:
                # Keep the clause branch in the autograd graph so DDP does not
                # mark core-head parameters as unused on SAT-only ranks.
                l_core = outputs['clause_scores'].sum() * 0.0
        else:
            l_core = outputs['clause_scores'].sum() * 0.0
        loss_dict['core_loss'] = l_core

        # 否定算子约束损失 (由模型前向传播产生)
        l_neg = outputs.get('negation_loss', torch.tensor(0.0, device=device))
        loss_dict['negation_loss'] = l_neg

        # 总损失
        total_loss = l_sat + self.alpha_core * l_core + self.alpha_neg * l_neg
        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict

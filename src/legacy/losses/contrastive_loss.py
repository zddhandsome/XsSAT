"""
GeoSATformer Contrastive Learning Loss Module
对比学习损失模块

实现三重对比学习目标：
1. Intra-MUC Cohesion: 同一MUC内的clauses在embedding空间中聚集
2. Inter-class Separation: MUC clauses与非MUC clauses分离
3. Cross-Instance Similarity: 不同实例中结构相似的MUC有相似表示

L_contrast = L_intra + L_inter + L_cross
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class IntraMUCLoss(nn.Module):
    """
    MUC内部聚合损失 (Intra-MUC Cohesion)
    
    确保同一MUC内的clause embeddings在特征空间中靠近
    使用InfoNCE风格的对比损失
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True
    ):
        """
        Args:
            temperature: softmax温度参数
            normalize: 是否对embeddings进行L2归一化
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        clause_embeddings: torch.Tensor,
        muc_labels: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算MUC内部聚合损失
        
        Args:
            clause_embeddings: clause嵌入 [batch_size, num_clauses, embed_dim]
            muc_labels: MUC标签 [batch_size, num_clauses], 1=属于MUC
            clause_mask: 有效clause掩码 [batch_size, num_clauses]
            
        Returns:
            loss: 标量损失值
        """
        batch_size, num_clauses, embed_dim = clause_embeddings.shape
        
        if self.normalize:
            clause_embeddings = F.normalize(clause_embeddings, p=2, dim=-1)
        
        total_loss = 0.0
        num_valid_samples = 0
        
        for b in range(batch_size):
            # 获取MUC和非MUC的clause indices
            if clause_mask is not None:
                valid_mask = clause_mask[b].bool()
                muc_mask = (muc_labels[b] == 1) & valid_mask
            else:
                muc_mask = muc_labels[b] == 1
            
            muc_indices = torch.where(muc_mask)[0]
            
            if len(muc_indices) < 2:
                continue
            
            # 获取MUC clause embeddings
            muc_embeds = clause_embeddings[b, muc_indices]  # [num_muc, embed_dim]
            
            # 计算相似度矩阵
            sim_matrix = torch.mm(muc_embeds, muc_embeds.t()) / self.temperature
            
            # 对角线为自己，需要mask掉
            mask_diag = torch.eye(len(muc_indices), device=sim_matrix.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask_diag, float('-inf'))
            
            # InfoNCE loss: 每个MUC clause应该与其他MUC clauses相似
            # 这里使用所有其他MUC clauses作为正样本
            labels = torch.arange(len(muc_indices), device=sim_matrix.device)
            
            # 简化版：使用mean of log softmax over positive pairs
            log_softmax = F.log_softmax(sim_matrix, dim=-1)
            
            # 排除对角线后的平均
            mask_not_diag = ~mask_diag
            loss = -log_softmax[mask_not_diag].mean()
            
            total_loss += loss
            num_valid_samples += 1
        
        if num_valid_samples > 0:
            return total_loss / num_valid_samples
        return torch.tensor(0.0, device=clause_embeddings.device)


class InterClassLoss(nn.Module):
    """
    MUC与非MUC分离损失 (Inter-class Separation)
    
    确保MUC clauses与非MUC clauses在特征空间中分离
    使用triplet loss或contrastive loss
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        loss_type: str = 'triplet',
        temperature: float = 0.07,
        hard_negative_mining: bool = True
    ):
        """
        Args:
            margin: triplet loss的margin
            loss_type: 'triplet' 或 'contrastive'
            temperature: contrastive loss的温度
            hard_negative_mining: 是否使用hard negative mining
        """
        super().__init__()
        self.margin = margin
        self.loss_type = loss_type
        self.temperature = temperature
        self.hard_negative_mining = hard_negative_mining
    
    def triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """三元组损失"""
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()
    
    def forward(
        self,
        clause_embeddings: torch.Tensor,
        muc_labels: torch.Tensor,
        clause_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算类间分离损失
        
        Args:
            clause_embeddings: clause嵌入 [batch_size, num_clauses, embed_dim]
            muc_labels: MUC标签 [batch_size, num_clauses]
            clause_mask: 有效clause掩码 [batch_size, num_clauses]
            
        Returns:
            loss: 标量损失值
        """
        batch_size, num_clauses, embed_dim = clause_embeddings.shape
        
        # L2归一化
        clause_embeddings = F.normalize(clause_embeddings, p=2, dim=-1)
        
        total_loss = 0.0
        num_valid_triplets = 0
        
        for b in range(batch_size):
            if clause_mask is not None:
                valid_mask = clause_mask[b].bool()
                muc_mask = (muc_labels[b] == 1) & valid_mask
                non_muc_mask = (muc_labels[b] == 0) & valid_mask
            else:
                muc_mask = muc_labels[b] == 1
                non_muc_mask = muc_labels[b] == 0
            
            muc_indices = torch.where(muc_mask)[0]
            non_muc_indices = torch.where(non_muc_mask)[0]
            
            if len(muc_indices) < 1 or len(non_muc_indices) < 1:
                continue
            
            muc_embeds = clause_embeddings[b, muc_indices]
            non_muc_embeds = clause_embeddings[b, non_muc_indices]
            
            if self.loss_type == 'triplet':
                # 采样triplets
                num_triplets = min(len(muc_indices) * len(non_muc_indices), 100)
                
                if len(muc_indices) >= 2:
                    # Anchor和Positive来自MUC，Negative来自非MUC
                    anchor_idx = torch.randint(len(muc_indices), (num_triplets,), device=muc_embeds.device)
                    
                    if self.hard_negative_mining:
                        # Hard negative: 选择与anchor最相似的非MUC clause
                        anchors = muc_embeds[anchor_idx]
                        sim_to_neg = torch.mm(anchors, non_muc_embeds.t())
                        neg_idx = sim_to_neg.argmax(dim=-1)
                    else:
                        neg_idx = torch.randint(len(non_muc_indices), (num_triplets,), device=muc_embeds.device)
                    
                    # Positive: 随机选择其他MUC clause
                    pos_idx = torch.randint(len(muc_indices), (num_triplets,), device=muc_embeds.device)
                    # 确保positive不等于anchor
                    same_mask = pos_idx == anchor_idx
                    pos_idx[same_mask] = (pos_idx[same_mask] + 1) % len(muc_indices)
                    
                    anchors = muc_embeds[anchor_idx]
                    positives = muc_embeds[pos_idx]
                    negatives = non_muc_embeds[neg_idx]
                    
                    loss = self.triplet_loss(anchors, positives, negatives)
                else:
                    # 只有一个MUC clause，使用简化的contrastive loss
                    anchor = muc_embeds[0:1].expand(len(non_muc_indices), -1)
                    loss = -F.cosine_similarity(anchor, -non_muc_embeds).mean()
                
                total_loss += loss
                num_valid_triplets += 1
            
            else:  # contrastive loss
                # MUC clause应该与其他MUC相似，与非MUC不相似
                # 使用InfoNCE风格
                all_embeds = torch.cat([muc_embeds, non_muc_embeds], dim=0)
                num_muc = len(muc_indices)
                
                for i in range(num_muc):
                    anchor = muc_embeds[i:i+1]
                    sim = torch.mm(anchor, all_embeds.t()).squeeze(0) / self.temperature
                    
                    # 正样本是其他MUC clauses
                    pos_mask = torch.zeros(len(all_embeds), device=sim.device)
                    pos_mask[:num_muc] = 1
                    pos_mask[i] = 0  # 排除自己
                    
                    if pos_mask.sum() > 0:
                        # Supervised contrastive loss
                        exp_sim = torch.exp(sim)
                        pos_exp = (exp_sim * pos_mask).sum()
                        neg_exp = (exp_sim * (1 - pos_mask)).sum()
                        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
                        total_loss += loss
                        num_valid_triplets += 1
        
        if num_valid_triplets > 0:
            return total_loss / num_valid_triplets
        return torch.tensor(0.0, device=clause_embeddings.device)


class CrossInstanceLoss(nn.Module):
    """
    跨实例MUC相似性损失 (Cross-Instance Similarity)
    
    不同SAT实例中结构相似的MUC应该有相似的表示
    基于MUC的结构特征（如clause长度分布、变量重叠度）定义相似性
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        similarity_threshold: float = 0.7
    ):
        """
        Args:
            temperature: softmax温度
            similarity_threshold: 结构相似性阈值
        """
        super().__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
    
    def compute_structural_similarity(
        self,
        muc_features1: Dict[str, torch.Tensor],
        muc_features2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算两个MUC的结构相似性
        
        基于：
        - MUC大小比例
        - 平均clause长度
        - 变量出现频率分布
        """
        size_sim = 1 - torch.abs(muc_features1['size_ratio'] - muc_features2['size_ratio'])
        length_sim = 1 - torch.abs(muc_features1['avg_clause_length'] - muc_features2['avg_clause_length'])
        
        # 综合相似度
        similarity = (size_sim + length_sim) / 2
        return similarity
    
    def forward(
        self,
        muc_representations: torch.Tensor,
        muc_features: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        计算跨实例相似性损失
        
        Args:
            muc_representations: 批次内每个实例的MUC全局表示 [batch_size, embed_dim]
            muc_features: 每个实例的MUC结构特征列表
            
        Returns:
            loss: 标量损失值
        """
        batch_size = muc_representations.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=muc_representations.device)
        
        # L2归一化
        muc_representations = F.normalize(muc_representations, p=2, dim=-1)
        
        # 计算embedding相似度矩阵
        embed_sim = torch.mm(muc_representations, muc_representations.t()) / self.temperature
        
        # 如果有结构特征，计算结构相似性作为监督信号
        if muc_features is not None and len(muc_features) == batch_size:
            struct_sim = torch.zeros(batch_size, batch_size, device=muc_representations.device)
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        struct_sim[i, j] = self.compute_structural_similarity(
                            muc_features[i], muc_features[j]
                        )
            
            # 结构相似的实例应该在embedding空间也相似
            # 使用MSE loss或者ranking loss
            mask = ~torch.eye(batch_size, device=embed_sim.device).bool()
            
            # 归一化embed_sim到[0,1]范围
            embed_sim_norm = torch.sigmoid(embed_sim)
            
            loss = F.mse_loss(embed_sim_norm[mask], struct_sim[mask])
        else:
            # 没有结构特征时，使用自监督方法
            # 假设同一batch内的MUC有一定相似性（来自相同分布）
            # 使用InfoNCE风格损失促进batch内表示的一致性
            
            mask_diag = torch.eye(batch_size, device=embed_sim.device).bool()
            embed_sim = embed_sim.masked_fill(mask_diag, float('-inf'))
            
            # 使用softmax得到概率分布
            log_prob = F.log_softmax(embed_sim, dim=-1)
            
            # 自监督：每个样本应该与batch中的某些样本相似
            # 简化实现：最大化与最相似样本的相似度
            loss = -log_prob.max(dim=-1)[0].mean()
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    完整的对比学习损失
    
    L_contrast = λ_intra * L_intra + λ_inter * L_inter + λ_cross * L_cross
    """
    
    def __init__(
        self,
        lambda_intra: float = 1.0,
        lambda_inter: float = 1.0,
        lambda_cross: float = 0.5,
        temperature: float = 0.07,
        triplet_margin: float = 1.0,
        intra_config: Optional[Dict] = None,
        inter_config: Optional[Dict] = None,
        cross_config: Optional[Dict] = None
    ):
        """
        Args:
            lambda_intra: MUC内部损失权重
            lambda_inter: 类间分离损失权重
            lambda_cross: 跨实例损失权重
            temperature: 默认温度参数
            triplet_margin: triplet loss的margin
            intra_config: IntraMUCLoss配置
            inter_config: InterClassLoss配置
            cross_config: CrossInstanceLoss配置
        """
        super().__init__()
        
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.lambda_cross = lambda_cross
        
        intra_cfg = intra_config or {'temperature': temperature}
        inter_cfg = inter_config or {'margin': triplet_margin, 'temperature': temperature}
        cross_cfg = cross_config or {'temperature': temperature}
        
        self.intra_loss = IntraMUCLoss(**intra_cfg)
        self.inter_loss = InterClassLoss(**inter_cfg)
        self.cross_loss = CrossInstanceLoss(**cross_cfg)
    
    def forward(
        self,
        clause_embeddings: torch.Tensor,
        muc_labels: torch.Tensor,
        muc_representations: Optional[torch.Tensor] = None,
        clause_mask: Optional[torch.Tensor] = None,
        muc_features: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        计算完整的对比学习损失
        
        Args:
            clause_embeddings: clause嵌入 [batch_size, num_clauses, embed_dim]
            muc_labels: MUC标签 [batch_size, num_clauses]
            muc_representations: MUC全局表示 [batch_size, embed_dim] (可选)
            clause_mask: 有效clause掩码 [batch_size, num_clauses]
            muc_features: MUC结构特征 (可选)
            return_components: 是否返回各组成部分
            
        Returns:
            loss: 总对比损失，或(总损失, 损失字典)
        """
        # 仅对UNSAT样本计算对比损失
        # 假设所有样本都有MUC标签（UNSAT样本）
        
        l_intra = self.intra_loss(clause_embeddings, muc_labels, clause_mask)
        l_inter = self.inter_loss(clause_embeddings, muc_labels, clause_mask)
        
        if muc_representations is not None:
            l_cross = self.cross_loss(muc_representations, muc_features)
        else:
            l_cross = torch.tensor(0.0, device=clause_embeddings.device)
        
        total_loss = (
            self.lambda_intra * l_intra +
            self.lambda_inter * l_inter +
            self.lambda_cross * l_cross
        )
        
        if return_components:
            return total_loss, {
                'intra_loss': l_intra,
                'inter_loss': l_inter,
                'cross_loss': l_cross
            }
        
        return total_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss)
    
    用于SAT/UNSAT分类的表示学习
    同类样本靠近，异类样本远离
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算监督对比损失
        
        Args:
            features: 样本特征 [batch_size, embed_dim]
            labels: 样本标签 [batch_size]
            mask: 有效样本掩码 [batch_size]
            
        Returns:
            loss: 标量损失值
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 构建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        
        # 计算相似度
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 移除对角线（自己与自己）
        mask_self = torch.eye(batch_size, device=device)
        mask_positive = mask_positive - mask_self
        
        # 对数概率
        exp_sim = torch.exp(similarity) * (1 - mask_self)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 计算正样本对的平均对数概率
        mean_log_prob_pos = (mask_positive * log_prob).sum(dim=1) / (mask_positive.sum(dim=1) + 1e-8)
        
        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss

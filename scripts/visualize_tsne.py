"""
t-SNE Visualization for Clause Embeddings

可视化 clause embeddings 的聚类效果，验证对比学习是否有效。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse


def load_model_and_data(checkpoint_path, data_path, device='cuda'):
    """加载模型和数据"""
    from src.models.geosatformer import GeoSATformer
    from src.data.dataset import PackedSATDataset

    # 加载数据集
    dataset = PackedSATDataset(data_path)

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint['model_state_dict']

    # 从 state_dict 推断模型配置
    embed_dim = sd['clause_patch_embed.projection.bias'].shape[0]
    num_layers = max([int(k.split('.')[2]) for k in sd.keys() if 'clause_tower.layers.' in k]) + 1
    num_hierarchical_levels = max([int(k.split('.')[2]) for k in sd.keys() if 'hierarchical.attention_units.' in k]) + 1
    patch_size = sd['clause_patch_embed.projection.weight'].shape[1]

    model_config = {
        'max_clauses': dataset.max_clauses,
        'max_vars': dataset.max_vars,
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'num_heads': 8,
        'patch_size': patch_size,
        'num_hierarchical_levels': num_hierarchical_levels
    }

    print(f"Inferred model config: {model_config}")

    model = GeoSATformer(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, dataset


def extract_embeddings(model, dataset, num_samples=100, device='cuda'):
    """
    提取 clause embeddings

    Args:
        model: GeoSATformer 模型
        dataset: 数据集
        num_samples: 采样数量
        device: 设备

    Returns:
        embeddings: 所有 clause embeddings
        labels: MUC 标签 (0=非MUC, 1=MUC)
        sat_labels: SAT 标签 (0=UNSAT, 1=SAT)
        instance_ids: 样本 ID
    """
    all_embeddings = []
    all_labels = []
    all_sat_labels = []
    all_instance_ids = []

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Extracting embeddings"):
            sample = dataset[idx]

            vsm = sample['vsm'].unsqueeze(0).to(device)
            clause_mask = sample['clause_mask'].unsqueeze(0).to(device)
            var_mask = sample['var_mask'].unsqueeze(0).to(device)

            outputs = model(vsm, clause_mask, var_mask, return_embeddings=True)

            # 获取 clause embeddings [1, max_clauses, embed_dim]
            clause_emb = outputs['clause_embeddings'].squeeze(0).cpu().numpy()

            # 获取有效 clause 的 mask
            valid_mask = clause_mask.squeeze(0).cpu().numpy()
            num_valid = valid_mask.sum()

            # 获取 MUC 标签
            core_labels = sample['core_labels'].numpy()
            sat_label = sample['sat_label'].item()

            # 只取有效的 clauses
            valid_embeddings = clause_emb[:num_valid]
            valid_labels = core_labels[:num_valid]

            all_embeddings.append(valid_embeddings)
            all_labels.append(valid_labels)
            all_sat_labels.extend([sat_label] * num_valid)
            all_instance_ids.extend([idx] * num_valid)

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    sat_labels = np.array(all_sat_labels)
    instance_ids = np.array(all_instance_ids)

    return embeddings, labels, sat_labels, instance_ids


def visualize_tsne(embeddings, labels, sat_labels, instance_ids, output_path,
                   perplexity=30, n_iter=1000, random_state=42):
    """
    使用 t-SNE 可视化 clause embeddings

    创建多个子图：
    1. 按 MUC/非MUC 标签着色
    2. 按 SAT/UNSAT 实例着色
    3. 只显示 UNSAT 实例的 MUC/非MUC 分布
    """
    print(f"Running t-SNE on {len(embeddings)} clause embeddings...")

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        max_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )

    embeddings_2d = tsne.fit_transform(embeddings)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ======== 子图1: MUC vs 非MUC (所有实例) ========
    ax1 = axes[0, 0]

    muc_mask = labels == 1
    non_muc_mask = labels == 0

    ax1.scatter(embeddings_2d[non_muc_mask, 0], embeddings_2d[non_muc_mask, 1],
                c='blue', alpha=0.3, s=10, label=f'Non-MUC ({non_muc_mask.sum()})')
    ax1.scatter(embeddings_2d[muc_mask, 0], embeddings_2d[muc_mask, 1],
                c='red', alpha=0.6, s=20, label=f'MUC ({muc_mask.sum()})')

    ax1.set_title('Clause Embeddings: MUC vs Non-MUC (All Instances)', fontsize=12)
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend()

    # ======== 子图2: SAT vs UNSAT 实例 ========
    ax2 = axes[0, 1]

    sat_mask = sat_labels == 1
    unsat_mask = sat_labels == 0

    ax2.scatter(embeddings_2d[sat_mask, 0], embeddings_2d[sat_mask, 1],
                c='green', alpha=0.4, s=10, label=f'SAT instances ({sat_mask.sum()})')
    ax2.scatter(embeddings_2d[unsat_mask, 0], embeddings_2d[unsat_mask, 1],
                c='orange', alpha=0.4, s=10, label=f'UNSAT instances ({unsat_mask.sum()})')

    ax2.set_title('Clause Embeddings: SAT vs UNSAT Instances', fontsize=12)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend()

    # ======== 子图3: 仅 UNSAT 实例的 MUC/非MUC ========
    ax3 = axes[1, 0]

    unsat_non_muc = unsat_mask & non_muc_mask
    unsat_muc = unsat_mask & muc_mask

    ax3.scatter(embeddings_2d[unsat_non_muc, 0], embeddings_2d[unsat_non_muc, 1],
                c='lightblue', alpha=0.4, s=10, label=f'Non-MUC ({unsat_non_muc.sum()})')
    ax3.scatter(embeddings_2d[unsat_muc, 0], embeddings_2d[unsat_muc, 1],
                c='darkred', alpha=0.7, s=25, label=f'MUC ({unsat_muc.sum()})')

    ax3.set_title('UNSAT Instances Only: MUC vs Non-MUC', fontsize=12)
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.legend()

    # ======== 子图4: 按实例 ID 着色 (显示聚类) ========
    ax4 = axes[1, 1]

    # 选择部分实例展示
    unique_instances = np.unique(instance_ids)
    num_show = min(10, len(unique_instances))
    show_instances = unique_instances[:num_show]

    colors = plt.cm.tab10(np.linspace(0, 1, num_show))

    for i, inst_id in enumerate(show_instances):
        inst_mask = instance_ids == inst_id
        sat_status = "SAT" if sat_labels[inst_mask][0] == 1 else "UNSAT"
        ax4.scatter(embeddings_2d[inst_mask, 0], embeddings_2d[inst_mask, 1],
                    c=[colors[i]], alpha=0.6, s=15, label=f'Inst {inst_id} ({sat_status})')

    ax4.set_title('Clause Embeddings by Instance (First 10)', fontsize=12)
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")

    # 打印统计信息
    print("\n===== Statistics =====")
    print(f"Total clauses: {len(embeddings)}")
    print(f"MUC clauses: {muc_mask.sum()} ({100*muc_mask.mean():.1f}%)")
    print(f"Non-MUC clauses: {non_muc_mask.sum()} ({100*non_muc_mask.mean():.1f}%)")
    print(f"From SAT instances: {sat_mask.sum()}")
    print(f"From UNSAT instances: {unsat_mask.sum()}")

    return embeddings_2d


def compute_clustering_metrics(embeddings, labels):
    """
    计算聚类质量指标

    Returns:
        metrics: 包含各种聚类指标的字典
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    metrics = {}

    # 只在有两个类别时计算
    if len(np.unique(labels)) >= 2:
        # Silhouette Score: [-1, 1], 越大越好
        metrics['silhouette'] = silhouette_score(embeddings, labels)

        # Davies-Bouldin Index: 越小越好
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)

    # 计算类内/类间距离比
    muc_mask = labels == 1
    non_muc_mask = labels == 0

    if muc_mask.sum() > 1 and non_muc_mask.sum() > 1:
        muc_embeddings = embeddings[muc_mask]
        non_muc_embeddings = embeddings[non_muc_mask]

        # MUC 类内平均距离
        muc_center = muc_embeddings.mean(axis=0)
        intra_muc = np.mean(np.linalg.norm(muc_embeddings - muc_center, axis=1))

        # 非MUC 类内平均距离
        non_muc_center = non_muc_embeddings.mean(axis=0)
        intra_non_muc = np.mean(np.linalg.norm(non_muc_embeddings - non_muc_center, axis=1))

        # 类间距离
        inter_class = np.linalg.norm(muc_center - non_muc_center)

        metrics['intra_muc_dist'] = intra_muc
        metrics['intra_non_muc_dist'] = intra_non_muc
        metrics['inter_class_dist'] = inter_class
        metrics['separation_ratio'] = inter_class / (intra_muc + intra_non_muc + 1e-8)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='t-SNE Visualization of Clause Embeddings')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/large/checkpoint_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                        default='data_packed/val.pt',
                        help='Path to packed dataset')
    parser.add_argument('--output', type=str,
                        default='tsne_clause_embeddings.png',
                        help='Output image path')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of SAT instances to sample')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # 转换为绝对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, args.checkpoint)
    data_path = os.path.join(base_dir, args.data)
    output_path = os.path.join(base_dir, args.output)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型和数据
    print("Loading model and data...")
    model, dataset = load_model_and_data(checkpoint_path, data_path, device)

    # 提取 embeddings
    embeddings, labels, sat_labels, instance_ids = extract_embeddings(
        model, dataset, num_samples=args.num_samples, device=device
    )

    # 计算聚类指标
    print("\n===== Clustering Metrics =====")
    metrics = compute_clustering_metrics(embeddings, labels)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 可视化
    visualize_tsne(
        embeddings, labels, sat_labels, instance_ids,
        output_path, perplexity=args.perplexity
    )


if __name__ == '__main__':
    main()

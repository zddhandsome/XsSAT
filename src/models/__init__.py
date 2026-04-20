"""
XsSAT Models Module

核心组件:
- XsSAT: 主模型 (多通道VSM + recurrent axial backbone)
- NegationOperator: 可学习否定算子
- CellEmbedding: Cell级嵌入 + 对偶位置编码
- AxialAttentionLayer: 稀疏轴向注意力层
- ClauseGlobalTokenLayer: clause级全局上下文写回
- PolarityPairMixer: 正负极性显式交互
- HardClauseReadout: hardest-clause 读出
"""

from .XsSAT import (
    XsSAT,
    NegationOperator,
    CellEmbedding,
    AxialAttentionLayer,
    AxialAttentionBlock,
    RecurrentCellUpdate,
    ClauseGlobalTokenLayer,
    TokenAttentionPool,
    PolarityPairMixer,
    HardClauseReadout,
)

__all__ = [
    'XsSAT',
    'NegationOperator',
    'CellEmbedding',
    'AxialAttentionLayer',
    'AxialAttentionBlock',
    'RecurrentCellUpdate',
    'ClauseGlobalTokenLayer',
    'TokenAttentionPool',
    'PolarityPairMixer',
    'HardClauseReadout',
]

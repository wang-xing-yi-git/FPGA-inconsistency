"""
简化的隐性不一致检测模型 - GAT + Bi-GRU（适配细粒度语义节点）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class SimpleGATLayer(nn.Module):
    """简化的图注意力层（适配变长细粒度节点）"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # 新增：注意力得分计算
        self.attention = nn.Linear(out_features * 2, 1, bias=False)

    def forward(
        self, 
        features: torch.Tensor,  # [batch, N, in_features]
        adj_matrix: torch.Tensor,  # [batch, N, N]
        mask: Optional[torch.Tensor] = None  # [batch, N] 屏蔽填充节点
    ) -> torch.Tensor:
        """
        增强版：加入注意力得分，更贴合细粒度节点的图注意力逻辑
        """
        # 1. 线性投影
        h = self.linear(features)  # [batch, N, out_features]
        
        # 2. 计算注意力得分
        N = h.size(1)
        h_repeat = h.unsqueeze(2).repeat(1, 1, N, 1)  # [batch, N, N, out_features]
        h_repeat_t = h.unsqueeze(1).repeat(1, N, 1, 1)  # [batch, N, N, out_features]
        concat = torch.cat([h_repeat, h_repeat_t], dim=-1)  # [batch, N, N, 2*out_features]
        e = F.leaky_relu(self.attention(concat).squeeze(-1))  # [batch, N, N]
        
        # 3. 结合邻接矩阵的注意力掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)  # 仅邻接节点有注意力
        
        # 4. 注意力归一化
        attention = F.softmax(attention, dim=-1)  # [batch, N, N]
        
        # 5. 邻接矩阵加权聚合
        aggregated = torch.bmm(attention, h)  # [batch, N, out_features]
        
        # 6. 应用节点掩码（屏蔽填充节点）
        if mask is not None:
            aggregated = aggregated * mask.unsqueeze(-1)
        
        return aggregated


class SimpleGAT(nn.Module):
    """简化的图注意力网络（适配变长细粒度节点）"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # 第一层
        self.layers.append(SimpleGATLayer(in_features, hidden_features))

        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(SimpleGATLayer(hidden_features, hidden_features))

        # 最后一层
        if num_layers > 1:
            self.layers.append(SimpleGATLayer(hidden_features, out_features))

    def forward(
        self, 
        features: torch.Tensor,  # [batch, N, in_features]
        adj_matrix: torch.Tensor,  # [batch, N, N]
        mask: Optional[torch.Tensor] = None  # [batch, N]
    ) -> torch.Tensor:
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_matrix, mask)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if mask is not None:
                    x = x * mask.unsqueeze(-1)
        return x


class ImplicitInconsistencyModel(nn.Module):
    """
    隐性不一致检测模型 (简化版GAT + Bi-GRU) - 细粒度语义节点版
    增强：优化细粒度节点的掩码和序列处理逻辑
    """

    def __init__(
        self,
        feature_dim: int = 768,          # 细粒度节点的特征维度
        hidden_dim: int = 256,           # 投影层隐藏维度
        gat_dim: int = 128,              # GAT输出维度
        gru_dim: int = 64,               # GRU隐藏维度
        gat_layers: int = 2,             # GAT层数
        gru_layers: int = 1,             # GRU层数
        dropout_rate: float = 0.1,       # Dropout率
    ):
        super().__init__()

        # 1. 细粒度节点特征投影（共享投影层）
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 2. GAT网络（处理细粒度节点的图结构）
        self.gat = SimpleGAT(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=gat_dim,
            num_layers=gat_layers,
        )

        # 3. Bi-GRU（处理细粒度节点序列）
        self.gru = nn.GRU(
            input_size=gat_dim,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if gru_layers > 1 else 0.0
        )

        # 4. GRU输出融合（双向）+ 序列池化
        self.gru_fusion = nn.Linear(gru_dim * 2, gru_dim)
        self.seq_pool = nn.AdaptiveAvgPool1d(1)  # 自适应池化（适配变长序列）

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        req_nodes: torch.Tensor,          # [batch, n, 768] 需求细粒度节点
        code_nodes: torch.Tensor,         # [batch, m, 768] 代码细粒度节点
        alignment_matrix: torch.Tensor,   # [batch, N, N] N = n+m（含填充）
        total_nodes: torch.Tensor         # [batch] 真实总节点数（n+m，不含填充）
    ) -> Tuple[torch.Tensor, Dict]:
        """
        增强版：优化掩码逻辑，确保细粒度节点的填充部分完全屏蔽
        """
        batch_size = req_nodes.shape[0]
        max_total_nodes = alignment_matrix.shape[1]
        device = req_nodes.device

        # 1. 拼接需求+代码细粒度节点 [batch, max_n+max_m, 768]
        combined_nodes = torch.cat([req_nodes, code_nodes], dim=1)
        
        # 2. 特征投影（对每个细粒度节点）
        projected = self.feature_projection(combined_nodes)  # [batch, L, hidden_dim]

        # 3. 构建高精度节点掩码（核心增强）
        mask = torch.zeros(batch_size, max_total_nodes, device=device)
        for i in range(batch_size):
            # 仅真实节点数范围内为1，填充部分为0
            mask[i, :total_nodes[i]] = 1.0
        # 确保mask是float类型（避免类型错误）
        mask = mask.float()

        # 4. GAT处理（使用细粒度节点的对齐矩阵，屏蔽填充）
        gat_out = self.gat(projected, alignment_matrix, mask)
        gat_out = gat_out * mask.unsqueeze(-1)  # 二次屏蔽，确保填充部分为0

        # 5. Bi-GRU处理细粒度节点序列
        # 增强：GRU输入前再次掩码，避免填充节点干扰
        gru_input = gat_out * mask.unsqueeze(-1)
        gru_out, gru_hidden = self.gru(gru_input)
        gru_out = gru_out * mask.unsqueeze(-1)

        # 6. 融合双向GRU输出 + 序列池化（适配变长节点）
        gru_out_fused = self.gru_fusion(gru_out)  # [batch, L, gru_dim]
        # 自适应平均池化：适配任意长度的细粒度序列
        gru_pooled = self.seq_pool(gru_out_fused.transpose(1, 2)).squeeze(-1)

        # 7. 分类预测
        output = self.classifier(gru_pooled)  # [batch, 1]

        # 诊断信息
        diagnostics = {
            "gat_output": gat_out.detach(),
            "gru_output": gru_out.detach(),
            "pooled_features": gru_pooled.detach(),
            "mask": mask.detach(),
            "attention_mask": mask.unsqueeze(-1).repeat(1, 1, gat_out.size(-1)).detach()
        }

        return output, diagnostics
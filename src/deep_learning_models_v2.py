"""
简化的隐性不一致检测模型 - GAT + Bi-GRU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class SimpleGATLayer(nn.Module):
    """简化的图注意力层"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, in_features]
            adj_matrix: [N, N] (相似度矩阵)
        Returns:
            [N, out_features]
        """
        # 应用邻接矩阵的加权聚合
        aggregated = torch.matmul(adj_matrix, features)  # [N, in_features]
        output = self.linear(aggregated)  # [N, out_features]
        return output


class SimpleGAT(nn.Module):
    """简化的图注意力网络"""

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

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, in_features]
            adj_matrix: [N, N]
        Returns:
            [N, out_features]
        """
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_matrix)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class ImplicitInconsistencyModel(nn.Module):
    """
    隐性不一致检测模型 (简化版GAT + Bi-GRU)

    管道：
    1. 特征投影
    2. GAT处理图结构
    3. Bi-GRU处理序列
    4. 分类预测
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        gat_dim: int = 128,
        gru_dim: int = 64,
        gat_layers: int = 2,
        gru_layers: int = 1,
    ):
        super().__init__()

        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        # GAT网络
        self.gat = SimpleGAT(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=gat_dim,
            num_layers=gat_layers,
        )

        # Bi-GRU
        self.gru = nn.GRU(
            input_size=gat_dim,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # GRU 输出融合层（双向）
        self.gru_fusion = nn.Linear(gru_dim * 2, gru_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        req_vector: torch.Tensor,
        code_vector: torch.Tensor,
        alignment_pairs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            req_vector: [1, 768]
            code_vector: [1, 768]
            alignment_pairs: [2, 2] (邻接矩阵)

        Returns:
            不一致分数 [1] 和诊断信息
        """
        # 1. 特征融合与投影
        combined = torch.cat([req_vector, code_vector], dim=0)  # [2, 768]
        projected = self.feature_projection(combined)  # [2, hidden_dim]

        # 2. GAT处理（使用对齐对作为图结构）
        gat_out = self.gat(projected, alignment_pairs)  # [2, gat_dim]

        # 3. 构建序列用于 Bi-GRU
        sequence = gat_out.unsqueeze(0)  # [1, 2, gat_dim]

        # 4. Bi-GRU处理
        gru_out, gru_hidden = self.gru(sequence)  # hidden: [2, 1, gru_dim]

        # 融合双向隐藏状态
        # gru_hidden: [num_layers*2, batch, gru_dim]
        # 对于 num_layers=1, bidirectional=True: [2, 1, gru_dim]
        forward_hidden = gru_hidden[0]  # [1, gru_dim]
        backward_hidden = gru_hidden[1]  # [1, gru_dim]
        combined_hidden = torch.cat(
            [forward_hidden, backward_hidden], dim=-1
        )  # [1, 2*gru_dim]
        fused_hidden = self.gru_fusion(combined_hidden)  # [1, gru_dim]

        # 5. 分类预测
        output = self.classifier(fused_hidden)  # [1, 1]
        output = output.squeeze(1)  # [1]

        diagnostics = {
            "gat_output": gat_out.detach(),
            "gru_output": gru_out.detach(),
        }

        return output, diagnostics

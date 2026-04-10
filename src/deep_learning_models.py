"""
深度学习模型模块
实现 GAT (图注意力网络) + Bi-GRU (双向门控循环单元) 用于隐性不一致检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional


class GATLayer(nn.Module):
    """
    图注意力网络层 (Graph Attention Layer)

    用途：学习图中节点之间的注意力权重
    - 节点 = 需求术语 + 代码元素
    - 边 = 向量相似关系

    原理：
    1. 对每个节点的特征进行线性变换
    2. 计算节点对之间的注意力系数 (softmax)
    3. 根据注意力权重聚合邻居节点信息
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout

        # 确保输出维度能被头数整除
        assert (
            out_features % num_heads == 0
        ), "out_features must be divisible by num_heads"
        self.head_dim = out_features // num_heads

        # 线性变换：特征投影
        self.W = nn.Linear(in_features, out_features, bias=False)

        # 注意力系数的学习参数 (用于计算注意力权重)
        self.a = nn.Parameter(torch.Tensor(1, num_heads, 2 * self.head_dim))

        # 偏置项
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 节点特征矩阵 [N, in_features]
            adj_matrix: 邻接矩阵 (相似度矩阵) [N, N]

        Returns:
            增强后的特征 [N, out_features]
        """
        # 1. 特征线性变换
        h = self.W(features)  # [N, out_features]
        h = h.view(-1, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]

        N = h.size(0)

        # 2. 计算注意力系数
        # 对每条边 (i, j) 计算注意力权重
        # concat(h_i, h_j) 经过线性层 + LeakyReLU

        # 膨胀h用于配对比较 [N, 1, num_heads, head_dim]
        h_i = h.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, num_heads, head_dim]
        h_j = h.unsqueeze(0).expand(N, -1, -1, -1)  # [N, N, num_heads, head_dim]

        # 连接 [N, N, num_heads, 2*head_dim]
        h_concat = torch.cat([h_i, h_j], dim=-1)

        # 计算注意力分数 [N, N, num_heads]
        e = torch.einsum("ijhd,hd->ijh", h_concat, self.a.squeeze(0))
        e = F.leaky_relu(e, negative_slope=0.2)

        # 3. 应用邻接矩阵 (只在有边的地方计算注意力)
        # adj_matrix [N, N] -> [N, N, 1] 扩展到多头
        mask = adj_matrix.unsqueeze(-1) > 0.1  # 相似度 > 0.1 认为有边

        # 将没有边的位置设为 -inf
        e = e.masked_fill(~mask, float("-inf"))

        # 4. Softmax 获得注意力权重 [N, N, num_heads]
        attention = F.softmax(e, dim=1)
        attention = self.dropout_layer(attention)

        # 处理 NaN (来自 -inf 的 softmax)
        attention = torch.nan_to_num(attention, 0.0)

        # 5. 应用注意力权重，聚合邻居信息
        # attention: [N, N, num_heads]
        # h: [N, num_heads, head_dim]
        # 对每个头进行注意力加权求和
        attention_mean = attention.mean(dim=2)  # [N, N]
        # 通过矩阵乘法进行加权聚合: [N, N] x [N, d] -> [N, d]
        h_aggregated = torch.matmul(attention_mean, h.view(N, -1))  # [N, in_features]
        h_out = h_aggregated

        # 6. 多头连接
        h_out = h_out.view(N, -1)
        h_out = h_out + self.bias

        return h_out


class GraphAttentionNetwork(nn.Module):
    """
    完整的图注意力网络 (GAT)

    用途：
    - 输入：需求/代码的节点特征 + 相似度矩阵 (邻接矩阵)
    - 输出：增强的节点表示 (考虑了图结构信息)
    - 作用：学习节点间的依赖关系和相互影响
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度
            out_features: 输出特征维度
            num_layers: GAT层数
            num_heads: 多头注意力头数
            dropout: Dropout概率
        """
        super(GraphAttentionNetwork, self).__init__()
        self.num_layers = num_layers

        # 构建多层GAT
        self.gat_layers = nn.ModuleList()

        # 第一层
        self.gat_layers.append(
            GATLayer(in_features, hidden_features, num_heads, dropout)
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATLayer(hidden_features, hidden_features, num_heads, dropout)
            )

        # 最后一层
        if num_layers > 1:
            self.gat_layers.append(
                GATLayer(hidden_features, out_features, num_heads, dropout)
            )

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 节点特征 [N, in_features]
            adj_matrix: 邻接矩阵 (相似度矩阵) [N, N]

        Returns:
            输出特征 [N, out_features]
        """
        x = features
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, adj_matrix)
            if i < len(self.gat_layers) - 1:  # 除了最后一层，都加ReLU
                x = F.relu(x)

        return x


class BiGRUEncoder(nn.Module):
    """
    双向GRU编码器 (Bidirectional GRU)

    用途：
    - 输入：序列数据 (需求-代码对齐序列)
    - 前向GRU：从左到右处理依赖关系
    - 后向GRU：从右到左处理影响关系
    - 融合：双向信息聚合用于隐性不一致检测

    作用：
    1. 捕捉对齐序列中的上下文信息
    2. 前向看"依赖"，后向看"被依赖"
    3. 融合双向信息判断是否存在隐性冲突
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: GRU隐藏状态维度
            num_layers: GRU层数
            dropout: Dropout概率
            bidirectional: 是否双向
        """
        super(BiGRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # 构建双向GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 输出层 (融合双向信息)
        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fusion_layer = nn.Linear(gru_output_dim, hidden_size)

    def forward(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            sequences: 序列数据 [batch_size, seq_len, input_size]

        Returns:
            (融合后的输出, 最后隐藏状态)
            - 融合输出: [batch_size, seq_len, hidden_size]
            - 最后隐藏状态: [batch_size, hidden_size]
        """
        # GRU前向传播
        output, hidden = self.gru(sequences)

        # 融合双向信息
        if self.bidirectional:
            # output shape: [batch_size, seq_len, hidden_size*2]
            fused_output = self.fusion_layer(output)
            # hidden shape: [num_layers*2, batch_size, hidden_size]
            # 取最后一层的双向隐藏状态
            fused_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            fused_hidden = self.fusion_layer(fused_hidden)
        else:
            fused_output = output
            fused_hidden = hidden[-1]

        return fused_output, fused_hidden


class ImplicitInconsistencyModel(nn.Module):
    """
    隐性不一致检测模型 (GAT + Bi-GRU)

    完整的深度学习管道：
    1. 使用 GAT 分析需求/代码节点间的关系图
    2. 使用 Bi-GRU 处理对齐序列中的上下文
    3. 融合两种信息进行不一致预测

    输入：
    - 需求特征向量 (768维)
    - 代码特征向量 (768维)
    - 对齐结果 (哪些需求与代码对应)

    输出：
    - 隐性不一致分数 [0, 1]
    - 不一致位置和类型
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        gat_dim: int = 128,
        gru_dim: int = 64,
        gat_layers: int = 2,
        gat_heads: int = 4,
        gru_layers: int = 1,
    ):
        """
        Args:
            feature_dim: 输入特征维度 (768 来自BERT/CNN)
            hidden_dim: 中间隐藏层维度
            gat_dim: GAT输出维度
            gru_dim: GRU隐藏层维度
            gat_layers: GAT层数
            gat_heads: GAT多头注意力头数
            gru_layers: GRU层数
        """
        super(ImplicitInconsistencyModel, self).__init__()

        # 1. 输入投影层 (将768维投影到中间维度)
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )

        # 2. GAT 部分 (分析图结构)
        self.gat = GraphAttentionNetwork(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=gat_dim,
            num_layers=gat_layers,
            num_heads=gat_heads,
            dropout=0.1,
        )

        # 3. Bi-GRU 部分 (分析序列)
        self.bigru = BiGRUEncoder(
            input_size=gat_dim,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            dropout=0.1,
            bidirectional=True,
        )

        # 4. 分类头 (预测不一致)
        # 注意：BiGRUEncoder 输出的隐藏状态经过融合层后是 gru_dim 维
        self.classifier = nn.Sequential(
            nn.Linear(gru_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 输出 [0, 1] 的不一致概率
        )

        # 5. 损失函数
        self.criterion = nn.BCELoss()

    def forward(
        self,
        req_vector: torch.Tensor,
        code_vector: torch.Tensor,
        alignment_pairs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            req_vector: 需求特征向量 [1, 768]
            code_vector: 代码特征向量 [1, 768]
            alignment_pairs: 对齐矩阵 [N, N] (0-1，表示相似度)

        Returns:
            (不一致分数, 诊断信息)
            - 不一致分数: [0, 1] 标量
            - 诊断信息: 包含中间计算结果的字典
        """
        batch_size = 1
        device = req_vector.device

        # 1. 融合特征 (需求 + 代码)
        # 这里创建一个节点集合：需求节点 + 代码节点
        combined_features = torch.cat([req_vector, code_vector], dim=0)  # [2, 768]

        # 2. 投影到中间维度
        projected = self.feature_projection(combined_features)  # [2, hidden_dim]

        # 3. 构建邻接矩阵 (相似度矩阵)
        # 计算需求和代码的相似度
        similarity = F.cosine_similarity(req_vector, code_vector, dim=-1)  # 标量

        # 创建对齐矩阵 [2, 2]
        adj_matrix = torch.zeros(2, 2, device=device)
        adj_matrix[0, 1] = similarity  # 需求到代码
        adj_matrix[1, 0] = similarity  # 代码到需求
        adj_matrix[0, 0] = 1.0  # 自环
        adj_matrix[1, 1] = 1.0

        # 4. GAT 处理 (分析节点关系)
        gat_output = self.gat(projected, adj_matrix)  # [2, gat_dim]

        # 5. 构建序列用于 Bi-GRU
        # 序列 = [req节点, code节点] -> [1, 2, gat_dim]
        sequence = gat_output.unsqueeze(0)  # [1, 2, gat_dim]

        # 6. Bi-GRU 处理序列
        gru_output, gru_hidden = self.bigru(sequence)  # hidden: [1, gru_dim]

        # 7. 分类预测
        inconsistency_score = self.classifier(gru_hidden)  # [1, 1]
        # 保持一维张量，只移除维度为1的维度
        inconsistency_score = inconsistency_score.squeeze(1)  # [1]

        # 诊断信息
        diagnostics = {
            "similarity": similarity.item(),
            "gat_output": gat_output.detach(),
            "gru_output": gru_output.detach(),
            "gru_hidden": gru_hidden.detach(),
        }

        return inconsistency_score, diagnostics

    def compute_loss(
        self, pred_scores: torch.Tensor, true_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失函数 (用于训练)

        Args:
            pred_scores: 预测的不一致分数 [batch_size]
            true_labels: 真实标签 (0/1) [batch_size]

        Returns:
            损失值 (标量)
        """
        return self.criterion(pred_scores, true_labels.float())


# ===== 辅助函数 =====


def build_alignment_matrix(
    req_elements: Dict, code_elements: Dict, alignment_pairs: List[Dict]
) -> torch.Tensor:
    """
    从对齐结果构建邻接矩阵

    Args:
        req_elements: 需求语义要素
        code_elements: 代码语义要素
        alignment_pairs: 对齐对列表 [{'req': ..., 'code': ..., 'confidence': ...}]

    Returns:
        邻接矩阵 [N, N]
    """
    # 提取所有节点
    req_keywords = req_elements.get("keywords", [])
    code_modules = code_elements.get("modules", [])

    N = len(req_keywords) + len(code_modules)
    adj_matrix = torch.zeros(N, N)

    # 填充对齐关系
    for pair in alignment_pairs:
        req_idx = None
        code_idx = None

        for i, kw in enumerate(req_keywords):
            if kw in pair.get("req", ""):
                req_idx = i
                break

        for j, mod in enumerate(code_modules):
            if mod in pair.get("code", ""):
                code_idx = len(req_keywords) + j
                break

        if req_idx is not None and code_idx is not None:
            confidence = pair.get("confidence", 0.5)
            adj_matrix[req_idx, code_idx] = confidence
            adj_matrix[code_idx, req_idx] = confidence

    return adj_matrix

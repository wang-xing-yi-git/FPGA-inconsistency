"""
GAT + Bi-GRU 模型训练脚本

本脚本用于训练隐性不一致检测模型

训练数据格式（必须提供）：
[
    {
        "id": 1,
        "req_text": "需求文本...",
        "code_text": "代码文本...",
        "req_vector": [768维浮点数],
        "code_vector": [768维浮点数],
        "alignment_pairs": [
            {"req": "术语1", "code": "元素1", "confidence": 0.9},
            ...
        ],
        "label": 0 或 1,  # 0=一致, 1=存在隐性不一致
        "inconsistency_details": "不一致类型描述..."  # 当label=1时
    },
    ...
]

训练步骤：
1. 准备数据集（至少500个样本）
2. 运行此脚本
3. 模型将保存到 models/implicit_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Dict
import os
from pathlib import Path

from src.deep_learning_models import ImplicitInconsistencyModel


class InconsistencyDataset(Dataset):
    """隐性不一致检测数据集"""

    def __init__(self, data_file: str):
        """
        Args:
            data_file: JSON数据文件路径
        """
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 转换为张量
        req_vector = torch.tensor(sample["req_vector"], dtype=torch.float32)
        code_vector = torch.tensor(sample["code_vector"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.float32)

        # 构建对齐矩阵
        alignment_pairs = sample.get("alignment_pairs", [])
        alignment_matrix = torch.zeros(2, 2, dtype=torch.float32)

        if alignment_pairs:
            avg_confidence = np.mean(
                [p.get("confidence", 0.5) for p in alignment_pairs]
            )
            alignment_matrix[0, 1] = avg_confidence
            alignment_matrix[1, 0] = avg_confidence

        alignment_matrix[0, 0] = 1.0
        alignment_matrix[1, 1] = 1.0

        return {
            "req_vector": req_vector,
            "code_vector": code_vector,
            "alignment_matrix": alignment_matrix,
            "label": label,
            "id": sample["id"],
        }


def train_model(
    data_file: str,
    model_save_path: str = "models/implicit_model.pth",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """
    训练隐性不一致检测模型

    Args:
        data_file: 训练数据文件路径 (JSON)
        model_save_path: 模型保存路径
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
    """

    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("\n请准备训练数据，格式示例：")
        print(
            """
[
    {
        "id": 1,
        "req_text": "需求文本...",
        "code_text": "代码文本...",
        "req_vector": [0.1, -0.2, ...],  // 768维向量
        "code_vector": [0.15, -0.18, ...],  // 768维向量
        "alignment_pairs": [
            {"req": "计数器", "code": "counter", "confidence": 0.92}
        ],
        "label": 0,  // 0=一致, 1=不一致
        "inconsistency_details": "具体问题描述"
    }
]
        """
        )
        return

    print("=" * 60)
    print("GAT + Bi-GRU 隐性不一致检测模型 - 训练")
    print("=" * 60)

    # 加载数据集
    print(f"\n📊 加载训练数据: {data_file}")
    dataset = InconsistencyDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"   样本数: {len(dataset)}")
    print(f"   批大小: {batch_size}")
    print(f"   批数: {len(dataloader)}")

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  设备: {device}")

    # 初始化模型
    print("\n🧠 初始化模型...")
    model = ImplicitInconsistencyModel(
        feature_dim=768,
        hidden_dim=256,
        gat_dim=128,
        gru_dim=64,
        gat_layers=2,
        gat_heads=4,
        gru_layers=1,
    )
    model = model.to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    print(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    print("\n🚀 开始训练...")
    print("-" * 60)

    best_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            req_vectors = batch["req_vector"].to(device)
            code_vectors = batch["code_vector"].to(device)
            alignment_matrices = batch["alignment_matrix"].to(device)
            labels = batch["label"].to(device)

            # 前向传播
            optimizer.zero_grad()

            predictions = []
            for i in range(req_vectors.size(0)):
                pred, _ = model(
                    req_vectors[i : i + 1],
                    code_vectors[i : i + 1],
                    alignment_matrices[i],
                )
                predictions.append(pred)

            predictions = torch.cat(predictions)

            # 计算损失
            loss = criterion(predictions, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # 计算准确率
            preds_binary = (predictions > 0.5).float()
            correct += (preds_binary == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}"
            )

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  早停触发 (连续 {patience} 个epoch无改进)")
                break

    print("-" * 60)
    print(f"\n✅ 训练完成！")
    print(f"   最佳模型已保存到: {model_save_path}")
    print(f"   最佳损失: {best_loss:.4f}")

    return model


def evaluate_model(
    model: ImplicitInconsistencyModel, data_file: str, device: str = "cpu"
):
    """
    评估模型性能

    Args:
        model: 已训练的模型
        data_file: 测试数据文件
        device: 计算设备
    """
    print("\n📈 模型评估")
    print("-" * 60)

    dataset = InconsistencyDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            req_vectors = batch["req_vector"].to(device)
            code_vectors = batch["code_vector"].to(device)
            alignment_matrices = batch["alignment_matrix"].to(device)
            labels = batch["label"].numpy()

            predictions = []
            for i in range(req_vectors.size(0)):
                pred, _ = model(
                    req_vectors[i : i + 1],
                    code_vectors[i : i + 1],
                    alignment_matrices[i],
                )
                predictions.append(pred.cpu().numpy())

            all_preds.extend(predictions)
            all_labels.extend(labels)

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # 计算指标
    tp = ((all_preds > 0.5) & (all_labels == 1)).sum()
    tn = ((all_preds <= 0.5) & (all_labels == 0)).sum()
    fp = ((all_preds > 0.5) & (all_labels == 0)).sum()
    fn = ((all_preds <= 0.5) & (all_labels == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    # 训练数据路径
    TRAIN_DATA = "data/implicit_inconsistency_training_data.json"
    TEST_DATA = "data/implicit_inconsistency_test_data.json"
    MODEL_PATH = "models/implicit_model.pth"

    # 训练模型
    model = train_model(
        data_file=TRAIN_DATA,
        model_save_path=MODEL_PATH,
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
    )

    # 评估模型 (如果有测试数据)
    if os.path.exists(TEST_DATA):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        evaluate_model(model, TEST_DATA, device=device)

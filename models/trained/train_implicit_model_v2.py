"""
使用简化 GAT+Bi-GRU 模型的训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict
import os

import sys
from pathlib import Path
# 获取当前脚本所在的目录（data文件夹）
current_dir = Path(__file__).parent
# 获取项目根目录（FPGA-inconsistency文件夹），即data的父目录
root_dir = current_dir.parent.parent
# 将项目根目录添加到Python搜索路径中
sys.path.insert(0, str(root_dir))
from src.deep_learning_models_v2 import ImplicitInconsistencyModel


class InconsistencyDataset(Dataset):
    """隐性不一致检测数据集"""

    def __init__(self, data_file: str):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

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
        }


def train_model(
    data_file: str = "data/implicit_inconsistency_training_data.json",
    model_save_path: str = "models/implicit_model.pth",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """训练隐性不一致检测模型"""

    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在 {data_file}")
        return None

    print("=" * 60)
    print("GAT + Bi-GRU 隐性不一致检测模型 - 训练")
    print("=" * 60)

    # 加载数据
    print(f"\n加载训练数据: {data_file}")
    dataset = InconsistencyDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  样本数: {len(dataset)}")
    print(f"  批大小: {batch_size}")
    print(f"  批数: {len(dataloader)}")

    # 设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    model = ImplicitInconsistencyModel()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    print("\n开始训练...")
    print("-" * 60)

    best_loss = float("inf")
    patience_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            req_vectors = batch["req_vector"].to(device)
            code_vectors = batch["code_vector"].to(device)
            alignment_matrices = batch["alignment_matrix"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # 处理每个样本
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            correct += ((predictions > 0.5) == labels).sum().item()
            total += len(labels)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
        )

        # 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: 在 {patience} 个 epoch 内没有改进")
                break

    print("-" * 60)
    print("训练完成！")
    print(f"模型已保存到: {model_save_path}")

    return model


def evaluate_model(
    model,
    test_file: str = "data/implicit_inconsistency_test_data.json",
    device=torch.device("cpu"),
):
    """评估模型"""

    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return

    print("\n评估模型...")
    print("-" * 60)

    # 加载测试数据
    dataset = InconsistencyDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            req_vectors = batch["req_vector"].to(device)
            code_vectors = batch["code_vector"].to(device)
            alignment_matrices = batch["alignment_matrix"].to(device)
            labels = batch["label"].to(device)

            for i in range(req_vectors.size(0)):
                pred, _ = model(
                    req_vectors[i : i + 1],
                    code_vectors[i : i + 1],
                    alignment_matrices[i],
                )
                all_preds.append((pred > 0.5).item())
                all_labels.append(labels[i].item())

    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = np.sum((all_preds == 1) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    import sys

    TRAIN_DATA = "data/implicit_inconsistency_training_data_v3.json"
    TEST_DATA = "data/implicit_inconsistency_test_data_v3.json"
    MODEL_PATH = "models/implicit_model_v3.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练
    model = train_model(
        data_file=TRAIN_DATA,
        model_save_path=MODEL_PATH,
        num_epochs=50,
        batch_size=16,
        learning_rate=0.0005,
    )

    if model is not None:
        # 加载最佳模型
        model.load_state_dict(torch.load(MODEL_PATH))
        model = model.to(device)

        # 评估
        evaluate_model(model, TEST_DATA, device)

    print()

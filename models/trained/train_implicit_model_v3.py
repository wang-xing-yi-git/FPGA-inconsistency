"""
使用简化 GAT+Bi-GRU 模型的训练脚本（适配细粒度语义节点）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict, List
import os

import sys
from pathlib import Path
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.insert(0, str(root_dir))
from src.deep_learning_models_v3 import ImplicitInconsistencyModel


class InconsistencyDataset(Dataset):
    """隐性不一致检测数据集（细粒度语义节点版）"""

    def __init__(self, data_file: str):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # 数据校验（新增）
        self._validate_data()

    def _validate_data(self):
        """校验细粒度数据格式"""
        for idx, sample in enumerate(self.data):
            # 校验req_nodes/code_nodes是二维列表
            assert isinstance(sample["req_nodes"], list), f"样本{idx}: req_nodes必须是列表"
            assert isinstance(sample["code_nodes"], list), f"样本{idx}: code_nodes必须是列表"
            for vec in sample["req_nodes"]:
                assert len(vec) == 768, f"样本{idx}: req_nodes向量维度必须为768"
            for vec in sample["code_nodes"]:
                assert len(vec) == 768, f"样本{idx}: code_nodes向量维度必须为768"
            # 校验对齐矩阵维度
            total_nodes = len(sample["req_nodes"]) + len(sample["code_nodes"])
            align_mat = sample["alignment_matrix"]
            assert len(align_mat) == total_nodes, f"样本{idx}: 对齐矩阵行数必须等于总节点数"
            assert all(len(row) == total_nodes for row in align_mat), f"样本{idx}: 对齐矩阵必须是方阵"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 1. 加载细粒度语义节点
        req_nodes = torch.tensor(sample["req_nodes"], dtype=torch.float32)
        code_nodes = torch.tensor(sample["code_nodes"], dtype=torch.float32)
        
        # 2. 加载动态维度的对齐矩阵
        alignment_matrix = torch.tensor(sample["alignment_matrix"], dtype=torch.float32)
        # 安全校验
        total_nodes = len(req_nodes) + len(code_nodes)
        assert alignment_matrix.shape == (total_nodes, total_nodes), \
            f"样本{idx}: 对齐矩阵维度{alignment_matrix.shape}与总节点数{total_nodes}不匹配"
        
        # 3. 标签
        label = torch.tensor(sample["label"], dtype=torch.float32)

        return {
            "req_nodes": req_nodes,
            "code_nodes": code_nodes,
            "alignment_matrix": alignment_matrix,
            "label": label,
            "total_nodes": total_nodes
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义批处理函数（处理变长细粒度节点）"""
    # 1. 提取批次内的关键信息
    req_nodes_list = [item["req_nodes"] for item in batch]
    code_nodes_list = [item["code_nodes"] for item in batch]
    alignment_matrices = [item["alignment_matrix"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    total_nodes_list = [item["total_nodes"] for item in batch]
    
    # 2. 计算批次内最大长度
    max_req_nodes = max([len(req) for req in req_nodes_list])
    max_code_nodes = max([len(code) for code in code_nodes_list])
    max_total_nodes = max(total_nodes_list)
    feature_dim = req_nodes_list[0].shape[-1]  # 768
    
    # 3. 填充req_nodes（[batch, max_req_nodes, 768]）
    padded_req_nodes = torch.zeros(len(batch), max_req_nodes, feature_dim, dtype=torch.float32)
    for i, req in enumerate(req_nodes_list):
        padded_req_nodes[i, :len(req)] = req
    
    # 4. 填充code_nodes（[batch, max_code_nodes, 768]）
    padded_code_nodes = torch.zeros(len(batch), max_code_nodes, feature_dim, dtype=torch.float32)
    for i, code in enumerate(code_nodes_list):
        padded_code_nodes[i, :len(code)] = code
    
    # 5. 填充alignment_matrix（[batch, max_total_nodes, max_total_nodes]）
    padded_alignment = torch.zeros(len(batch), max_total_nodes, max_total_nodes, dtype=torch.float32)
    for i, mat in enumerate(alignment_matrices):
        n = mat.shape[0]
        padded_alignment[i, :n, :n] = mat
    
    return {
        "req_nodes": padded_req_nodes,
        "code_nodes": padded_code_nodes,
        "alignment_matrix": padded_alignment,
        "label": labels,
        "total_nodes": torch.tensor(total_nodes_list, dtype=torch.int32)
    }


def train_model(
    data_file: str = "data/implicit_inconsistency_training_data_v3.json",
    model_save_path: str = "models/implicit_model.pth",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """训练隐性不一致检测模型（细粒度语义节点版）"""

    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在 {data_file}")
        return None

    print("=" * 60)
    print("GAT + Bi-GRU 隐性不一致检测模型 - 细粒度语义节点版训练")
    print("=" * 60)

    # 加载数据
    print(f"\n加载训练数据: {data_file}")
    dataset = InconsistencyDataset(data_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )

    print(f"  样本数: {len(dataset)}")
    print(f"  批大小: {batch_size}")
    print(f"  批数: {len(dataloader)}")
    # 新增：打印细粒度节点统计
    req_node_counts = [len(sample["req_nodes"]) for sample in dataset.data]
    code_node_counts = [len(sample["code_nodes"]) for sample in dataset.data]
    print(f"  需求节点数范围: {min(req_node_counts)}-{max(req_node_counts)}")
    print(f"  代码节点数范围: {min(code_node_counts)}-{max(code_node_counts)}")

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
            # 数据移到设备
            req_nodes = batch["req_nodes"].to(device)
            code_nodes = batch["code_nodes"].to(device)
            alignment_matrices = batch["alignment_matrix"].to(device)
            labels = batch["label"].to(device)
            total_nodes = batch["total_nodes"].to(device)

            optimizer.zero_grad()

            # 前向传播
            predictions, _ = model(
                req_nodes=req_nodes,
                code_nodes=code_nodes,
                alignment_matrix=alignment_matrices,
                total_nodes=total_nodes
            )

            # 计算损失
            loss = criterion(predictions.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计指标
            epoch_loss += loss.item()
            correct += ((predictions > 0.5).squeeze() == labels).sum().item()
            total += len(labels)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
        )

        # 早停逻辑
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
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
    test_file: str = "data/implicit_inconsistency_test_data_v3.json",
    device=torch.device("cpu"),
):
    """评估模型（细粒度语义节点版）"""

    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return

    print("\n评估模型...")
    print("-" * 60)

    # 加载测试数据
    dataset = InconsistencyDataset(test_file)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_fn
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            req_nodes = batch["req_nodes"].to(device)
            code_nodes = batch["code_nodes"].to(device)
            alignment_matrices = batch["alignment_matrix"].to(device)
            labels = batch["label"].to(device)
            total_nodes = batch["total_nodes"].to(device)

            # 前向传播
            predictions, _ = model(
                req_nodes=req_nodes,
                code_nodes=code_nodes,
                alignment_matrix=alignment_matrices,
                total_nodes=total_nodes
            )
            
            # 收集预测结果
            all_preds.extend((predictions > 0.5).squeeze().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

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
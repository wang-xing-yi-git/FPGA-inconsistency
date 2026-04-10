"""
简化版本训练脚本 - 用于快速测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import Dict


class SimpleInconsistencyModel(nn.Module):
    """简化的隐性不一致检测模型"""

    def __init__(self):
        super().__init__()
        # 简单的特征提取和分类
        self.feature_projection = nn.Sequential(
            nn.Linear(768 * 2, 256), nn.ReLU(), nn.Dropout(0.1)
        )

        self.classification = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, req_vector, code_vector, alignment_pairs):
        # 简单融合两个向量
        combined = torch.cat([req_vector, code_vector], dim=-1)
        features = self.feature_projection(combined)
        output = self.classification(features)
        return output.squeeze(1), {}


class SimpleDataset(Dataset):
    """简化数据集"""

    def __init__(self, data_file: str):
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "req_vector": torch.tensor(sample["req_vector"], dtype=torch.float32),
            "code_vector": torch.tensor(sample["code_vector"], dtype=torch.float32),
            "alignment_pairs": torch.zeros(2, 2),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


def train_simple():
    print("=" * 60)
    print("简化模型训练")
    print("=" * 60)

    # 加载数据
    dataset = SimpleDataset("data/implicit_inconsistency_training_data.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleInconsistencyModel().to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"数据集大小: {len(dataset)}")
    print(f"设备: {device}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")
    print()

    # 训练
    for epoch in range(20):
        model.train()
        total_loss = 0
        correct = 0

        for batch in dataloader:
            req_vectors = batch["req_vector"].to(device)
            code_vectors = batch["code_vector"].to(device)
            alignment_pairs = batch["alignment_pairs"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # 处理小批量
            predictions = []
            for i in range(req_vectors.size(0)):
                pred, _ = model(
                    req_vectors[i : i + 1], code_vectors[i : i + 1], alignment_pairs[0]
                )
                predictions.append(pred)

            predictions = torch.cat(predictions) if predictions else torch.tensor([])

            if len(predictions) > 0:
                loss = criterion(predictions, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                correct += ((predictions > 0.5) == labels).sum().item()

        accuracy = 100 * correct / len(dataset)
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:2d}/20 | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), "models/simple_model.pth")
    print("\n模型已保存")


if __name__ == "__main__":
    train_simple()

"""
模型评估和结果分析脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
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


def evaluate_model(model_path: str, test_file: str):
    """评估模型"""

    print("=" * 60)
    print("GAT + Bi-GRU 隐性不一致检测模型 - 评估")
    print("=" * 60)
    print()

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return

    if not os.path.exists(test_file):
        print(f"错误: 测试文件不存在 {test_file}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = ImplicitInconsistencyModel()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # 加载测试数据
    print(f"加载测试数据: {test_file}")
    dataset = InconsistencyDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"  样本数: {len(dataset)}")
    print()

    # 评估
    all_preds = []
    all_labels = []
    all_scores = []

    print("正在评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # DataLoader 带有 batch_size=1 会添加批维度
            # batch['alignment_matrix'] 的形状是 [1, 2, 2]
            req_vector = batch["req_vector"].to(device)  # [1, 768]
            code_vector = batch["code_vector"].to(device)  # [1, 768]
            alignment_matrix = batch["alignment_matrix"][0].to(device)  # [2, 2]
            label = batch["label"].to(device)  # [1]

            score, _ = model(req_vector, code_vector, alignment_matrix)
            pred = (score > 0.5).item()

            all_scores.append(score.item())
            all_preds.append(pred)
            all_labels.append(label.item())

    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = np.sum((all_preds == 1) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n评估结果:")
    print("-" * 60)
    print(f"准确率 (Accuracy):    {accuracy:.4f} ({tp+tn}/{len(all_labels)})")
    print(f"精确率 (Precision):   {precision:.4f}")
    print(f"召回率 (Recall):      {recall:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print()
    print("混淆矩阵:")
    print(f"  TP (True Positive):   {int(tp):3d}")
    print(f"  TN (True Negative):   {int(tn):3d}")
    print(f"  FP (False Positive):  {int(fp):3d}")
    print(f"  FN (False Negative):  {int(fn):3d}")
    print()

    # 分类分布
    consistent_count = np.sum(all_labels == 0)
    inconsistent_count = np.sum(all_labels == 1)
    print(f"样本分布:")
    print(
        f"  一致样本:   {consistent_count} ({consistent_count/len(all_labels)*100:.1f}%)"
    )
    print(
        f"  不一致样本: {inconsistent_count} ({inconsistent_count/len(all_labels)*100:.1f}%)"
    )
    print()

    # 预测得分分布
    print(f"预测得分统计:")
    print(f"  最小值: {np.min(all_scores):.4f}")
    print(f"  最大值: {np.max(all_scores):.4f}")
    print(f"  平均값: {np.mean(all_scores):.4f}")
    print(f"  中位数: {np.median(all_scores):.4f}")
    print()


if __name__ == "__main__":
    MODEL_PATH = "models/implicit_model_v2.pth"
    TEST_FILE = "data/implicit_inconsistency_test_data.json"

    evaluate_model(MODEL_PATH, TEST_FILE)

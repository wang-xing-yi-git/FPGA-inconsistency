# GAT + Bi-GRU 模型训练指南

## 概述

隐性不一致检测模型使用 **图注意力网络 (GAT) + 双向门控循环单元 (Bi-GRU)** 的深度学习架构。需要大量标注数据来训练这个模型。

---

## 🎯 需要训练什么？

### 核心概念

**GAT (图注意力网络)**：
- 作用：分析需求术语和代码元素之间的关系
- 学习：哪些需求对应哪些代码，对应强度有多强
- 输出：增强的节点表示（考虑了依赖关系）

**Bi-GRU (双向循环网络)**：
- 作用：处理对齐序列中的上下文信息
- 学习：前向依赖（"A是什么，所以B应该是..."）、后向影响（"如有B，说明A..."）
- 输出：融合了双向信息的序列编码

**最终分类器**：
- 输入：GAT输出 + Bi-GRU输出
- 输出：隐性不一致的概率 [0, 1]
  - 0 = 完全一致
  - 1 = 存在隐性不一致

---

## 📊 训练数据格式

### 文件路径
```
data/implicit_inconsistency_training_data.json
```

### JSON结构示例

```json
[
    {
        "id": 1,
        "req_text": "实现一个8位递增计数器，具有异步复位功能。",
        "code_text": "module counter(clk, rst_n, count); output [7:0] count; ...",
        "req_vector": [0.1, -0.2, 0.05, ...],        // 768维向量
        "code_vector": [0.15, -0.18, 0.08, ...],    // 768维向量
        "alignment_pairs": [
            {
                "req": "8位",
                "code": "output [7:0]",
                "confidence": 0.95
            },
            {
                "req": "递增",
                "code": "count <= count + 1",
                "confidence": 0.92
            },
            {
                "req": "异步复位",
                "code": "always @(negedge rst_n)",
                "confidence": 0.88
            }
        ],
        "label": 0,  // 0=一致, 1=不一致
        "inconsistency_details": null
    },
    {
        "id": 2,
        "req_text": "实现一个8位计数器。",
        "code_text": "module counter(clk, rst_n, count); output [6:0] count; ...",
        "req_vector": [0.12, -0.21, ...],
        "code_vector": [0.11, -0.19, ...],
        "alignment_pairs": [
            {
                "req": "8位", 
                "code": "output [6:0]",  // ❌ 只有7位！
                "confidence": 0.92
            }
        ],
        "label": 1,  // 存在不一致
        "inconsistency_details": {
            "type": "WIDTH_MISMATCH",
            "issue": "需求8位但代码只有7位",
            "severity": "CRITICAL"
        }
    }
]
```

---

## 📋 字段详解

### 必须字段

| 字段              | 类型    | 说明     | 范围/格式        |
| ----------------- | ------- | -------- | ---------------- |
| `id`              | int     | 唯一编号 | ≥ 1              |
| `req_text`        | string  | 需求原文 | 任意文本         |
| `code_text`       | string  | 代码原文 | 任意Verilog      |
| `req_vector`      | float[] | 需求向量 | 768维            |
| `code_vector`     | float[] | 代码向量 | 768维            |
| `alignment_pairs` | array   | 对齐对   | [{...}, ...]     |
| `label`           | 0\|1    | 标签     | 0=一致, 1=不一致 |

### alignment_pairs 结构

```json
{
    "req": "需求中的短语",      // 字符串
    "code": "代码中的元素",      // 字符串  
    "confidence": 0.92           // 0.7 ~ 1.0
}
```

### inconsistency_details 结构 (当 label=1 时)

```json
{
    "type": "WIDTH_MISMATCH",    // 不一致类型
    "issue": "位宽不匹配",        // 问题描述
    "severity": "CRITICAL",       // 严重程度
    "description": "详细说明",     // 可选
    "affected_elements": [...]    // 可选
}
```

---

## 🛠️ 如何准备训练数据

### 方式1：快速合成数据（推荐开始）

```python
import json
import numpy as np

def generate_synthetic_data(num_samples=500):
    data = []
    
    for i in range(num_samples):
        # 随机选择：一致(70%) 或 不一致(30%)
        is_consistent = np.random.random() > 0.3
        
        # 需求和代码向量（实际使用BERT/CNN生成）
        req_vector = np.random.randn(768).tolist()
        code_vector = np.random.randn(768).tolist()
        
        # 随机生成对齐对
        num_pairs = np.random.randint(2, 5)
        alignment_pairs = []
        for j in range(num_pairs):
            alignment_pairs.append({
                "req": f"需求要素{j}",
                "code": f"代码元素{j}",
                "confidence": np.random.uniform(0.75, 0.99)
            })
        
        # 标签
        label = 0 if is_consistent else 1
        
        data.append({
            "id": i + 1,
            "req_text": f"需求文本{i}",
            "code_text": f"代码文本{i}",
            "req_vector": req_vector,
            "code_vector": code_vector,
            "alignment_pairs": alignment_pairs,
            "label": label,
            "inconsistency_details": None if label == 0 else {
                "type": "LOGIC_ERROR",
                "issue": "随机不一致",
                "severity": "MEDIUM"
            }
        })
    
    # 保存
    with open('data/implicit_inconsistency_training_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ 生成{len(data)}个训练样本")

# 运行
generate_synthetic_data(500)
```

### 方式2：使用真实数据标注

**步骤**：
1. 收集5-10个真实的FPGA设计案例（需求+代码）
2. 对每个案例，识别对齐对和标注是否一致
3. 使用脚本生成向量
4. 重复此过程，累积500+样本

```python
from src.semantic_extraction import NLPSemanticExtractor, CodeSemanticExtractor

nlp_ext = NLPSemanticExtractor(model_name="bert-base-chinese")
code_ext = CodeSemanticExtractor()

# 为真实数据生成向量
req_vector = nlp_ext.get_semantic_vector(req_text).tolist()
code_vector = code_ext.get_semantic_vector(code_text).tolist()
```

---

## 📈 数据集要求

为了成功训练模型，检查以下要求：

- ✅ **最少500个样本**（建议1000+）
- ✅ **标签分布**：70% 一致，30% 不一致
- ✅ **所有向量**：恰好768维
- ✅ **对齐置信度**：0.7～1.0范围
- ✅ **JSON有效性**：能被`json.load()`解析
- ✅ **对齐对**：每个样本至少1对

---

## 🚀 训练模型

### 1. 检查数据文件

```bash
ls -la data/implicit_inconsistency_training_data.json
```

### 2. 运行训练

```bash
python train_implicit_model.py
```

### 3. 监控训练进度

```
Epoch   1/50 | Loss: 0.6931 | Acc: 50.00%
Epoch   5/50 | Loss: 0.5234 | Acc: 72.50%
Epoch  10/50 | Loss: 0.4102 | Acc: 81.25%
Epoch  20/50 | Loss: 0.3001 | Acc: 88.50%
Epoch  30/50 | Loss: 0.2234 | Acc: 91.20%
Epoch  40/50 | Loss: 0.1856 | Acc: 92.80%
Epoch  50/50 | Loss: 0.1523 | Acc: 94.30%
✅ 训练完成！
✅ 最佳模型已保存到: models/implicit_model.pth
```

### 4. 模型输出

```
models/implicit_model.pth (已训练的模型权重)
```

---

## 📊 期望性能

| 指标     | 目标值 |
| -------- | ------ |
| 准确率   | > 85%  |
| 精确率   | > 85%  |
| 召回率   | > 80%  |
| F1-Score | > 0.82 |

---

## ❓ FAQ

**Q: 一定需要真实标注数据吗？**
> 不必。可以从合成数据开始训练，然后用少量真实数据微调。

**Q: 最少需要多少数据？**
> 至少200个样本可以训练，但建议500+以获得更好效果。

**Q: 如何获取768维向量？**
> 使用系统内置的提取器：
> ```python
> req_vec = nlp_extractor.get_semantic_vector(req_text)  # 768维
> code_vec = code_extractor.get_semantic_vector(code_text)  # 768维
> ```

**Q: 训练需要多长时间？**
> - 500样本：5-10分钟 (GPU) / 30-50分钟 (CPU)
> - 1000样本：10-20分钟 (GPU) / 60-90分钟 (CPU)

**Q: 如何提升模型性能？**
> 1. 增加训练样本数量
> 2. 改进对齐对的质量（置信度高一些）
> 3. 添加更多真实数据
> 4. 微调超参数

---

## 完成检查清单

- [ ] 已生成/收集训练数据 (500+ 样本)
- [ ] 数据保存为 JSON 格式
- [ ] 所有向量维度都是 768
- [ ] 对齐对的置信度都在 0.7-1.0
- [ ] 标签只有 0 或 1
- [ ] 数据文件路径正确
- [ ] 已运行 `python train_implicit_model.py`
- [ ] 模型成功保存到 `models/implicit_model.pth`

---

**现在开始吧！** 🚀

`python train_implicit_model.py`

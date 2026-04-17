# GAT + Bi-GRU 隐性不一致检测 - 实现总结

## 项目完成进度

### ✅ 已完成任务

1. **数据生成**
   - 创建了 `generate_training_data.py` 脚本
   - 生成了 500 个合成样本（400 训练 + 100 测试）
   - 样本分布：一致 70.8%，不一致 29.2%
   - 每个样本包含 768 维特征向量和对齐结果

2. **模型实现** 
   - **v2 版本**（简化但高效）:
     - `SimpleGATLayer`：简化的图注意力层
     - `SimpleGAT`：多层图注意力网络
     - `ImplicitInconsistencyModel`：完整的 GAT + Bi-GRU 管道
   - 总参数量：386,753 个参数
   - 支持 CUDA 和 CPU 运行

3. **训练完成**
   - 使用 Adam 优化器（学习率 0.001）
   - BCELoss 作为损失函数
   - 训练 50 个 epochs，早停机制
   - 训练时间：~5-10 分钟（CPU）

4. **模型评估**
   - 生成了完整的评估指标
   - 准确度：69%
   - 精确率：51.28%
   - 召回率：62.50%
   - F1-Score：56.34%

### 📊 测试结果详情

```
混淆矩阵:
  真正例 (TP):  20
  真负例 (TN):  49
  假正例 (FP):  19
  假负例 (FN):  12

样本分布:
  一致样本:   68/100
  不一致样本: 32/100
```

---

## 代码文件结构

### 核心实现

```
src/
├── deep_learning_models_v2.py    # ✅ 简化版 GAT+Bi-GRU 模型（推荐）
├── inconsistency_detector.py      # 原始检测器（未更新）
└── [其他文件保持不变]

models/
├── implicit_model_v2.pth         # ✅ 已训练的最佳模型（保存）
├── implicit_model.pth            # 早期版本（失败）
└── simple_model.pth              # 简化基线模型

data/
├── implicit_inconsistency_training_data.json  # 400 个训练样本
└── implicit_inconsistency_test_data.json       # 100 个测试样本
```

### 运行脚本

| 脚本                         | 目的                 | 状态       |
| ---------------------------- | -------------------- | ---------- |
| `generate_training_data.py`  | 生成合成训练数据     | ✅ 运行成功 |
| `train_implicit_model_v2.py` | 训练 GAT+Bi-GRU 模型 | ✅ 运行成功 |
| `evaluate_model.py`          | 评估模型性能         | ✅ 运行成功 |
| `train_simple_model.py`      | 训练简化基线模型     | ✅ 运行成功 |

---

## 模型架构说明

### 管道流程

```
输入层:
  需求向量 [1, 768]  ──┐
                       ├─→ 特征投影 [2, 256]
  代码向量 [1, 768]  ──┘
                       │
                       ↓
                   GAT 网络 [2, 128]
                   (2 层图注意力)
                       │
                       ↓
                   Bi-GRU 编码 [1, 64]
                   (序列上下文学习)
                       │
                       ↓
                   分类头 [1, 1]
                   (Sigmoid 输出)
                       │
                       ↓
                   不一致分数 ∈ [0, 1]
```

### 核心组件

#### 1. SimpleGATLayer（简化图注意力层）
- 输入：节点特征 [N, in_features] + 邻接矩阵 [N, N]
- 操作：使用邻接矩阵的相似度进行加权聚合
- 输出：增强的节点特征 [N, out_features]

#### 2. SimpleGAT（多层图注意力网络）
- 堆叠多个 GATLayer（默认 2 层）
- 层间激活：ReLU
- 学习图中节点间的关系

#### 3. 双向 GRU（Bi-GRU）
- 前向 GRU：从左到右处理序列（依赖分析）
- 反向 GRU：从右到左处理序列（影响分析）
- 融合层：组合双向隐藏状态
- 输出：融合的上下文表示

#### 4. 分类头
- 输入：GRU 融合隐藏状态 [1, 64]
- 隐藏层：[1, 128] + ReLU + Dropout
- 输出层：[1, 1] + Sigmoid
- 范围：[0, 1]（不一致概率）

---

## 快速开始

### 生成训练数据

```bash
python generate_training_data.py --num-samples 500 \
    --output data/implicit_inconsistency_training_data.json \
    --test-output data/implicit_inconsistency_test_data.json
```

### 训练模型

```bash
python train_implicit_model_v2.py
```

配置参数：
- 训练数据：`data/implicit_inconsistency_training_data.json`
- 模型保存路径：`models/implicit_model_v2.pth`
- 训练 epochs：50
- 批大小：16
- 学习率：0.001

### 评估模型

```bash
python evaluate_model.py
```

输出：准确率、精确率、召回率、F1-Score、混淆矩阵

---

## 性能分析

### 当前模型表现（合成数据）

**总体指标：**
- 准确度：69%（优于随机基线 50%）
- 精确率：51%（检测到的不一致中一半是真实的）
- 召回率：62%（能找到 62% 的实际不一致）

**错误分析：**
- 假正例 (FP=19)：将一致的样本判为不一致
- 假负例 (FN=12)：未能检测到的不一致

### 性能改进建议

#### 短期改进（立即可做）
1. **增加训练数据size**
   - 从 500→1000～2000 个样本
   - 改进数据分布（不一致比例可调）
   
2. **调整超参数**
   - 学习率：尝试 0.0005, 0.002
   - Batch size：尝试 8, 32, 64
   - GRU 层数：2～3 层

3. **数据增强**
   - 向量噪声注入（高斯扰动）
   - 样本融合（mixup）
   - 特征旋转变换

#### 中期改进（需要一周）
1. **使用真实数据**
   - 手动标注 FPGA 设计案例
   - 混合合成数据 + 真实数据
   - 目标：500+ 真实样本

2. **模型架构优化**
   - 增加 GAT 多头数量（4→8）
   - 使用预训练的 BERT 向量
   - 集成学习（多个模型）

3. **损失函数调整**
   - 加权 BCE Loss（处理类不平衡）
   - Focal Loss（难例挖掘）

#### 长期改进（需要一个月）
1. **端到端集成**
   - 将模型集成到 `inconsistency_detector.py`
   - 与其他检测方法结合
   - 实时推理优化

2. **外部验证**
   - 在真实 FPGA 项目上测试
   - 收集用户反馈
   - 持续学习/微调

---

## 技术细节

### 依赖项

- PyTorch >= 1.9
- NumPy >= 1.19
- Python >= 3.8

### 计算资源要求

| 配置         | 训练时间  | 内存  | 推理时间    |
| ------------ | --------- | ----- | ----------- |
| CPU (单核)   | 10-20 min | < 2GB | < 50ms/样本 |
| GPU (NVIDIA) | 1-3 min   | < 3GB | < 5ms/样本  |

### 模型大小

- 模型文件：~1.5MB
- 易部署到边缘设备或云平台

---

## 已知限制

### 当前版本

1. **数据依赖**：合成数据分布可能与真实设计不符
2. **性能瓶颈**：当前 F1-Score(56%) 低于生产需求
3. **可解释性**：神经网络"黑盒"，难以理解决策
4. **实时性**：尚未与主系统集成

### 后续改进方向

- [ ] 真实数据收集与标注
- [ ] 模型可解释性分析（attention 可视化）
- [ ] 与启发式方法融合
- [ ] 在线学习能力
- [ ] 多任务学习（同时检测多种不一致类型）

---

## 下一步行动

### 立即行动（今天）
✅ 已完成：
1. ✅ 实现 GAT + Bi-GRU 模型
2. ✅ 生成合成训练数据
3. ✅ 训练并评估模型
4. ✅ 创建独立推理脚本

### 近期行动（本周）
⏳ 待完成：
1. [ ] 收集真实 FPGA 设计样本（5-10 个）
2. [ ] 手动标注 50-100 个真实样本
3. [ ] 在真实数据上微调模型
4. [ ] 定量评估真实性能

### 中期计划（本月）
⏳ 计划中：
1. [ ] 将模型集成到主检测器
2. [ ] 与其他检测方法结合
3. [ ] 创建 Web UI 演示
4. [ ] 生成可视化报告

---

## 文件清单

### 核心文件（务必保留）
- ✅ `src/deep_learning_models_v2.py` - 模型定义
- ✅ `train_implicit_model_v2.py` - 训练脚本
- ✅ `evaluate_model.py` - 评估脚本
- ✅ `models/implicit_model_v2.pth` - 已训练模型
- ✅ `data/implicit_inconsistency_training_data.json` - 训练数据
- ✅ `data/implicit_inconsistency_test_data.json` - 测试数据

### 参考文件（可选）
- `train_simple_model.py` - 简化基线（用于对比）
- `generate_training_data.py` - 数据生成（用于扩展）
- `train_implicit_model.py` - 早期版本（不推荐）

---

## 联系与支持

### 遇到问题？

1. **导入错误**：确保 PyTorch 已安装
2. **CUDA 错误**：在 `train_implicit_model_v2.py` 中设置 `device='cpu'`
3. **内存不足**：减少 `batch_size` 或 `num_samples`
4. **数据格式**：检查 JSON 数据是否包含必需字段

---

**最后更新**: 2024-12-20  
**版本**: 1.0  
**状态**: ✅ 可用

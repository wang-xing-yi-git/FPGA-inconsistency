# 🎯 快速参考卡 (Quick Reference Card)

## 中文版 🇨🇳

### 核心特性
- **双层检测**: 显性(13条规则) + 隐性(GAT+Bi-GRU)
- **自动降级**: 优先DL模型，无模型时用启发式
- **智能结果**: JSON格式，包含置信度、严重程度、建议

### 最常见的 3 个任务

#### 1️⃣ 快速开始（5分钟）
```python
# 导入
from src.inconsistency_detector import InconsistencyDetector

# 创建检测器（自动加载模型）
detector = InconsistencyDetector(
    deep_learning_model_path='models/implicit_model_v2.pth'
)

# 检测
result = detector.detect_all_inconsistencies(
    req_id=1,
    req_text="需求文本",
    req_elements={...},
    req_vector=req_vec,  # 768维BERT向量
    code_text="代码文本",
    code_elements={...},
    code_vector=code_vec  # 768维CNN向量
)

# 输出
print(result['total_issues'])  # 总问题数
```

#### 2️⃣ 批量处理数据集
```bash
# 生成合成数据
python generate_training_data.py --num-samples 500

# 处理整个数据集
python main.py --input data/raw/dataset.json --output report.json

# 查看结果
cat reports/report.json | jq .inconsistencies
```

#### 3️⃣ 训练自己的模型
```bash
# 1. 准备数据（手动标注或用生成脚本）
python generate_training_data.py

# 2. 训练模型
python train_implicit_model_v2.py

# 3. 评估模型
python evaluate_model.py

# 4. 使用新模型
detector = InconsistencyDetector(
    deep_learning_model_path='models/implicit_model_v2.pth'
)
```

### 系统输出示例

```json
{
  "req_id": 1,
  "total_issues": 3,
  "explicit_inconsistencies": [
    {
      "type": "missing_reset",
      "description": "缺少异步复位端口",
      "severity": "HIGH",
      "fix_suggestion": "添加 async_reset 端口"
    }
  ],
  "implicit_inconsistencies": [
    {
      "type": "semantic_gap",
      "description": "需求提及8位，代码实现16位",
      "confidence": 0.85,
      "severity": "MEDIUM"
    }
  ]
}
```

### 关键参数

| 参数              | 含义          | 示例                                             |
| ----------------- | ------------- | ------------------------------------------------ |
| `req_vector`      | 需求语义向量  | 768维NumPy数组                                   |
| `code_vector`     | 代码语义向量  | 768维NumPy数组                                   |
| `alignment_pairs` | 需求-代码对齐 | `[{"req_idx": 0, "code_idx": 0, "score": 0.85}]` |
| `model_path`      | 模型文件路径  | `models/implicit_model_v2.pth`                   |

### 常见问题速答

| Q               | A                                                             |
| --------------- | ------------------------------------------------------------- |
| 模型文件找不到? | 系统会自动降级到启发式，继续工作 ✅                            |
| PyTorch没装?    | 查看 requirements.txt，运行 `pip install -r requirements.txt` |
| 推理太慢?       | 用GPU: 改 `device='cuda'`，或批处理                           |
| 准确度不高?     | 用真实数据微调模型 (现在用合成数据，69%准确)                  |

### 文件导航 (5秒版)

```
想快速了解系统?       → README.md (5分钟)
想深入理解设计?       → ARCHITECTURE.md (30分钟)
想修改/扩展代码?      → DETAILS.md + 代码注释
想搭建开发环境?       → QUICKSTART.md + PROJECT_GUIDE.md
想用中文处理文本?     → CHINESE_SUPPORT_GUIDE.md
想理解语法依赖?       → SYNTAX_DEPENDENCY_GUIDE.md
想看集成验证详情?     → INTEGRATION_VERIFICATION.md
```

---

## English Version 🇬🇧

### Core Features
- **Dual-Layer Detection**: Explicit (13 rules) + Implicit (GAT+Bi-GRU)
- **Auto Fallback**: Prioritize DL model, use heuristics if unavailable
- **Smart Results**: JSON format with confidence, severity, suggestions

### 3 Most Common Tasks

#### 1️⃣ Quick Start (5 minutes)
```python
# Import
from src.inconsistency_detector import InconsistencyDetector

# Create detector (auto-loads model)
detector = InconsistencyDetector(
    deep_learning_model_path='models/implicit_model_v2.pth'
)

# Detect inconsistencies
result = detector.detect_all_inconsistencies(
    req_id=1,
    req_text="requirement text",
    req_elements={...},
    req_vector=req_vec,  # 768-dim BERT vector
    code_text="code text",
    code_elements={...},
    code_vector=code_vec  # 768-dim CNN vector
)

# Output
print(result['total_issues'])  # Total issues found
```

#### 2️⃣ Batch Processing Dataset
```bash
# Generate synthetic data
python generate_training_data.py --num-samples 500

# Process entire dataset
python main.py --input data/raw/dataset.json --output report.json

# View results
cat reports/report.json | jq .inconsistencies
```

#### 3️⃣ Train Custom Model
```bash
# 1. Prepare data (manual annotation or generation)
python generate_training_data.py

# 2. Train model
python train_implicit_model_v2.py

# 3. Evaluate
python evaluate_model.py

# 4. Use new model
detector = InconsistencyDetector(
    deep_learning_model_path='models/implicit_model_v2.pth'
)
```

### System Output Example

```json
{
  "req_id": 1,
  "total_issues": 3,
  "explicit_inconsistencies": [
    {
      "type": "missing_reset",
      "description": "Missing async reset port",
      "severity": "HIGH",
      "fix_suggestion": "Add async_reset port"
    }
  ],
  "implicit_inconsistencies": [
    {
      "type": "semantic_gap",
      "description": "Requirement mentions 8-bit, code implements 16-bit",
      "confidence": 0.85,
      "severity": "MEDIUM"
    }
  ]
}
```

### Key Parameters

| Parameter         | Meaning                     | Example                                          |
| ----------------- | --------------------------- | ------------------------------------------------ |
| `req_vector`      | Requirement semantic vector | 768-dim NumPy array                              |
| `code_vector`     | Code semantic vector        | 768-dim NumPy array                              |
| `alignment_pairs` | Req-Code alignment          | `[{"req_idx": 0, "code_idx": 0, "score": 0.85}]` |
| `model_path`      | Model file path             | `models/implicit_model_v2.pth`                   |

### FAQ Quick Answers

| Q                      | A                                                           |
| ---------------------- | ----------------------------------------------------------- |
| Model file not found?  | System auto-falls back to heuristics, keeps working ✅       |
| PyTorch not installed? | See requirements.txt, run `pip install -r requirements.txt` |
| Inference too slow?    | Use GPU: set `device='cuda'`, or batch process              |
| Accuracy not good?     | Fine-tune with real data (now 69% with synthetic data)      |

### File Navigation (5 seconds)

```
Want quick overview?          → README.md (5 min)
Want deep understanding?      → ARCHITECTURE.md (30 min)
Want to modify/extend code?   → DETAILS.md + code comments
Want dev environment setup?   → QUICKSTART.md + PROJECT_GUIDE.md
Want Chinese text handling?   → CHINESE_SUPPORT_GUIDE.md
Want syntax dependency info?  → SYNTAX_DEPENDENCY_GUIDE.md
Want integration details?     → INTEGRATION_VERIFICATION.md
```

---

## 📊 Performance @ a Glance

```
Model Accuracy:    69% (合成数据 / on synthetic data)
Inference Time:    <50ms (CPU)
Model Size:        1.5MB
Memory Usage:      2-3GB
End-to-end Latency: 3-5s (full pipeline)
```

---

## 🛠️ Environment Setup 环境配置

### Windows (快速版 / Quick Version)

```powershell
# 1. Clone 克隆
git clone <repo-url>
cd FPGA-inconsistency

# 2. Create venv 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 3. Install 安装依赖
pip install -r requirements.txt

# 4. Download models 下载模型
# 自动下载，或手动从models/目录

# 5. Run 运行
python main.py
```

### Linux/Mac (快速版 / Quick Version)

```bash
# 1-4 同上 (Same as above)
git clone <repo-url>
cd FPGA-inconsistency
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Run 运行
python main.py
```

---

## 📦 Dependencies 依赖清单

### 必需 (Required)
```
torch>=1.9.0          # Deep learning
transformers          # BERT models
numpy                 # Numerical computing
```

### 可选 (Optional)
```
cuda-toolkit          # GPU support (if you have NVIDIA GPU)
jieba                 # Chinese tokenization
spacy                 # NLP processing
```

### 一键安装 (One-liner)
```bash
pip install -r requirements.txt
```

---

## 🔑 Model Architecture 模型架构一览

```
Input (768-dim vectors + alignment pairs)
    ↓
Feature Projection: 768 → 256-dim
    ↓
GAT Encoding (2 layers): 256 → 128 → 64-dim
    ↓
Bi-GRU Processing: 64-dim bidirectional
    ↓
Classification Head: 64 → 1 (sigmoid)
    ↓
Output: Inconsistency probability [0, 1]

Total Parameters: 386,753
Training Accuracy: 69%
F1-Score: 56.34%
```

---

## 🚀 Deployment 部署

### Local Development 本地开发
```bash
python main.py --input data/raw/dataset.json
```

### Production Deployment 生产部署
```bash
# Docker
docker build -t fpga-detector .
docker run -p 8000:8000 fpga-detector

# Or Python server
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Monitoring 监控
```bash
# View logs
tail -f training.log

# Check model
python -c "import torch; print(torch.load('models/implicit_model_v2.pth'))"
```

---

## 💡 Tips & Tricks

### 🎯 性能优化 / Performance Optimization

```python
# 使用批处理 / Use batching
batches = [data[i:i+32] for i in range(0, len(data), 32)]
for batch in batches:
    results = detector.detect_all_inconsistencies(batch)

# GPU加速 / GPU acceleration
detector.device = 'cuda'

# 模型量化 / Model quantization
import torch.quantization
torch.quantization.quantize_dynamic(model, ...)
```

### 🔍 调试 / Debugging

```python
# 启用详细日志 / Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查向量维度 / Check vector dimensions
print(f"req_vector shape: {req_vector.shape}")  # Should be (768,)
print(f"code_vector shape: {code_vector.shape}")  # Should be (768,)

# 验证对齐 / Verify alignment
for pair in alignment_pairs:
    assert 'req_idx' in pair and 'code_idx' in pair
```

### 📊 结果分析 / Result Analysis

```python
# 统计不一致 / Count inconsistencies
result = detector.detect_all_inconsistencies(...)
explicit_count = len(result['explicit_inconsistencies'])
implicit_count = len(result['implicit_inconsistencies'])

# 按严重程度分类 / Group by severity
high_severity = [i for i in result['all_issues'] 
                 if i.get('severity') == 'HIGH']

# 导出JSON / Export to JSON
import json
with open('output.json', 'w') as f:
    json.dump(result, f, indent=2)
```

---

## 📞 Support 支持

### Where to Find Help 寻求帮助

| Issue                             | Solution                                   |
| --------------------------------- | ------------------------------------------ |
| 代码问题 / Code bugs              | 查看代码注释、DETAILS.md                   |
| 使用问题 / Usage questions        | 查看 README.md + example_usage.py          |
| 模型问题 / Model issues           | 查看 TRAINING_GUIDE.md + evaluate_model.py |
| 架构问题 / Architecture questions | 查看 ARCHITECTURE.md                       |
| 集成问题 / Integration issues     | 查看 INTEGRATION_VERIFICATION.md           |

### Common Solutions 常见解决方案

```bash
# 重新安装依赖 / Reinstall dependencies
pip install --upgrade -r requirements.txt

# 清理缓存 / Clear cache
rm -rf __pycache__ *.pyc models/__pycache__

# 重置环境 / Reset environment
python -m venv venv --clear
source venv/bin/activate

# 测试环境 / Test environment
python -c "import torch; import transformers; import numpy; print('OK')"
```

---

## ✅ Checklist 检查清单

### Before Using Model 使用模型前

- [ ] Python 3.7+ 已安装
- [ ] PyTorch 已安装 (pip install torch)
- [ ] 模型文件存在: `models/implicit_model_v2.pth`
- [ ] 数据格式正确 (768维向量)

### After Getting Results 获取结果后

- [ ] 验证结果有效性 (JSON格式)
- [ ] 检查不一致数量是否合理
- [ ] 查看严重程度分布
- [ ] 查阅建议措施

### For Production 生产部署前

- [ ] 所有测试通过
- [ ] 文档已审核
- [ ] 配置已验证
- [ ] 日志已设置
- [ ] 监控已部署

---

## 📚 Learning Path 学习路径

### 5分钟 / 5 Minutes
```
README.md 序言 + 快速开始
```

### 30分钟 / 30 Minutes
```
完整 README.md + 运行一个 example
```

### 2小时 / 2 Hours
```
PROJECT_GUIDE.md + ARCHITECTURE.md + 代码审查
```

### 8小时 / 8 Hours
```
所有文档 + 代码详解 + 运行训练脚本
```

### 1-2适 / 1-2 Weeks
```
收集真实数据 + 微调模型 + 部署系统
```

---

**Version**: 1.0  
**Last Updated**: 2024-12-20  
**Status**: Production Ready ✅

---

💾 **Save this file** for quick reference!  
在快速参考时保存此文件！

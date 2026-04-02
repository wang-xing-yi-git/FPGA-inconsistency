# FPGA设计文实不一致检测系统

FPGA设计需求文档与代码实现的智能化检测平台。采用双向语义提取、多维度对齐、显隐性不一致双重分析，精准识别文实脱节问题。

## ⚡ 核心技术

| 模块             | 技术方案                                        | 特点                              |
| ---------------- | ----------------------------------------------- | --------------------------------- |
| **需求语义提取** | BERT + jieba/NLTK分词                           | 支持中英文自动检测；768维语义向量 |
| **代码语义提取** | AST + PyTorch CNN                               | 结构化解析；256→768维深度编码     |
| **语义对齐**     | 向量相似度(35%) + 规则匹配(35%) + 模式匹配(30%) | 三维综合评分；13条增强规则库      |
| **不一致检测**   | 显性(规则库) + 隐性(GAT+Bi-GRU)                 | 双层分析；精准定位问题            |

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装Python包
pip install -r requirements.txt

# 下载语言模型
python -m spacy download en_core_web_sm

# 可选：中文模型支持
pip install zh_core_web_sm
```

### 2. 基本使用

```bash
# 处理英文数据集
python main.py --input data/raw/dataset.json --output reports/report.json

# 处理中文数据集
python main.py --input data/raw/dataset_chinese.json --output reports/report_zh.json
```

### 3. Python API

```python
from src.semantic_extraction import NLPSemanticExtractor, CodeSemanticExtractor

# 需求语义提取
nlp_extractor = NLPSemanticExtractor(model_name="bert-base-uncased", language="auto")
req_elements = nlp_extractor.extract_semantic_elements("实现8bit计数器")
req_vector = nlp_extractor.get_semantic_vector("实现8bit计数器")

# 代码语义提取  
code_extractor = CodeSemanticExtractor()
code_elements = code_extractor.extract_semantic_elements("module counter(...)")
code_vector = code_extractor.get_semantic_vector("module counter(...)")
```

---

### 📥 数据流向示意

```
输入 JSON
{
  "id": 1,
  "req_desc_origin": "需求描述文本...",
  "code_origin": "Verilog代码..."
}
        │
        ├─────────────────────────────────┬─────────────────────────────────┤
        │ 需求端                           │ 代码端
        ├─────────────────────────────────┼─────────────────────────────────┤
        │ 分词处理                          │ AST构建
        │ ↓ BERT编码                       │ ↓ 特征提取
        │ req_vector(768维)                │ ↓ CNN编码
        │                                 │ code_vector(768维)
        └────────────────┬────────────────┴────────────────┬────────────────┘
                         │                                 │
                    对齐评分 & 融合 (35% + 35% + 30%)
                         │                                 │
                    alignment_result (对齐对 + 置信度)
                         │
                    ┌────┴────┐
            不一致检测 ├─显性        不一致结果 → JSON输出
                    ├─隐性(GAT+Bi-GRU)
                    └────┬────┘
                         │
                    生成报告
```

---

### 💾 数据结构示例

**第1阶段输出**：
```python
req_elements = {
    'keywords': ['计数器', '8bit', '异步复位', ...],
    'entities': ['counter', 'clk', 'rst', ...],
    'concept': 'sequential_logic'
}
req_vector = numpy.array([0.1, -0.2, 0.5, ...])  # 768维
```

**第2阶段输出**：
```python
code_elements = {
    'modules': [{'name': 'counter', 'ports': ['clk', 'rst', 'count']}],
    'ports': {
        'input': [{'name': 'clk', 'width': 1}, {'name': 'rst', 'width': 1}],
        'output': [{'name': 'count', 'width': 8}]
    },
    'signals': [
        {'name': 'temp', 'type': 'wire', 'width': 8},
        {'name': 'count_reg', 'type': 'reg', 'width': 8}
    ],
    'always_blocks': [{'trigger': 'posedge clk', 'lines': 15}],
    'fpga_features': [
        {'type': 'sequential_logic', 'detected': True},
        {'type': 'reset_mechanism', 'detected': True}
    ]
}
code_vector = numpy.array([0.15, -0.18, 0.52, ...])  # 768维
```

**第3阶段输出**：
```python
alignment_result = {
    'alignment_pairs': [
        {'req': '8bit计数', 'code': 'counter [7:0]', 'confidence': 0.92},
        {'req': '异步复位', 'code': 'always @(posedge rst)', 'confidence': 0.85},
        ...
    ],
    'overall_confidence': 0.88,
    'status': 'WELL_ALIGNED'
}
```

**第4阶段输出**：
```python
inconsistency_result = {
    'explicit': [
        {'type': 'NAME_MISMATCH', 'severity': 'MEDIUM', 'detail': '...'},
        {'type': 'MISSING_FEATURE', 'severity': 'HIGH', 'detail': '...'}
    ],
    'implicit': [
        {'type': 'LOGIC_ERROR', 'probability': 0.78, 'detail': '...'}
    ],
    'overall_risk': 'MEDIUM'
}
```

## 📂 项目结构

```
FPGA-inconsistency/
├── README.md                    # 🟢 项目入门指南（本文件）
├── DETAILS.md                   # 📕 完整技术文档（所有详细信息）
├── config.yaml                  # 配置文件
├── requirements.txt             # 依赖包
├── main.py                      # 主程序入口
│
├── src/                         # 核心源代码
│   ├── semantic_extraction.py   # NLP + 代码语义提取
│   ├── semantic_alignment.py    # 多维度对齐模块
│   ├── inconsistency_detector.py # 不一致检测
│   ├── models.py                # 深度学习模型
│   ├── rules_engine.py          # 规则引擎
│   ├── data_processor.py        # 数据处理
│   └── report_generator.py      # 报告生成
│
├── data/                        # 数据文件
│   ├── raw/                     # 原始数据集
│   └── rules/                   # 规则库
│
├── models/                      # 预训练模型存储
│
├── reports/                     # 输出报告
│
├── tests/                       # 测试文件
│
└── demo_*.py                    # 演示脚本
```

## 🔧 完整技术说明

本文档精简介绍了核心内容。**完整的技术文档、中文支持、开发指南、修复报告、改进清单** 请阅读 [DETAILS.md](DETAILS.md)

### DETAILS.md 包含：
- ✅ 详细的安装配置步骤
- ✅ PyTorch迁移详解
- ✅ 正则表达式→AST优化
- ✅ 问题修复报告 (KeyError)
- ✅ 中文自然语言处理完整指南
- ✅ 语法依赖分析技术文档
- ✅ 开发修改指南
- ✅ 完整改进清单

## 💡 关键特性

### 多语言支持
- ✅ 中文自动检测与处理
- ✅ jieba分词 + BERT编码
- ✅ 11个中文FPGA术语映射

### 深度学习模型
- ✅ BERT 768维语义编码
- ✅ PyTorch CNN 3层编码器
- ✅ GAT + Bi-GRU 隐性分析

### 多维对齐算法
- ✅ 向量余弦相似度 (35%)
- ✅ 语义规则匹配 (35%)
- ✅ 模式识别匹配 (30%)

### 处理能力
- ✅ 显性不一致检测 (基于规则)
- ✅ 隐性不一致检测 (基于深度学习)
- ✅ 双向语义融合对齐

## 📝 数据格式

### 输入 JSON

```json
{
  "id": 1,
  "req_desc_origin": "设计一个8位计数器，具有异步复位功能",
  "code_origin": "module counter(input clk, input rst, output [7:0] count); ..."
}
```

### 输出报告

```json
{
  "id": 1,
  "alignment_status": "WELL_ALIGNED",
  "alignment_confidence": 0.88,
  "explicit_inconsistencies": [
    {"type": "NAME_MISMATCH", "severity": "MEDIUM", "detail": "..."}
  ],
  "implicit_inconsistencies": [
    {"type": "LOGIC_ERROR", "probability": 0.78, "detail": "..."}
  ],
  "overall_risk_level": "LOW"
}
```

## 🎓 学习资源

| 主题            | 位置                                                   |
| --------------- | ------------------------------------------------------ |
| 快速开始 & 安装 | [DETAILS.md #1](DETAILS.md#1-安装和配置)               |
| PyTorch/CNN改进 | [DETAILS.md #2](DETAILS.md#2-技术改进总结)             |
| Bug修复说明     | [DETAILS.md #3](DETAILS.md#3-代码修复报告)             |
| 中文处理        | [DETAILS.md #4](DETAILS.md#4-中文自然语言处理技术文档) |
| 语法分析        | [DETAILS.md #5](DETAILS.md#5-语法依赖分析技术文档)     |
| 开发指南        | [DETAILS.md #6](DETAILS.md#6-开发修改指南)             |
| 改进清单        | [DETAILS.md #7](DETAILS.md#7-完整改进清单)             |

## 📞 常见问题

**Q: 系统如何自动检测中英文？**  
A: 文本中CJK字符比例 > 30% 时判定为中文，否则为英文。详见 [DETAILS.md #4.2](DETAILS.md#42-自动语言检测)

**Q: 对齐分数如何计算？**  
A: 三维综合评分 = 相似度(35%) + 规则置信度(35%) + 模式匹配(30%)。详见 [DETAILS.md #2.3](DETAILS.md#23-语义对齐增强)

**Q: 如何修改规则库？**  
A: 编辑 `src/semantic_alignment.py` 中的 `SemanticMappingRulesLibrary` 类。详见 [DETAILS.md #6.3](DETAILS.md#63-扩展语义映射规则库)

**Q: PyTorch和TensorFlow的区别？**  
A: PyTorch更灵活、更快速、更易调试。详见 [DETAILS.md #2.1](DETAILS.md#21-pytorch迁移tensorflow--pytorch)

**Q: KeyError 问题已修复？**  
A: 是的，已修复数据结构不一致问题。详见 [DETAILS.md #3](DETAILS.md#3-代码修复报告)

## 🚀 下一步

1. **快速演示** → 运行 `python demo_syntax_dependency.py` 或 `python demo_chinese_nlp.py`
2. **处理你的数据** → 参考 [DETAILS.md #1](DETAILS.md#1-安装和配置)  
3. **理解技术细节** → 阅读完整的 [DETAILS.md](DETAILS.md)
4. **自定义系统** → 参考 [DETAILS.md #6](DETAILS.md#6-开发修改指南)

---

**系统状态**: ✅ 已修复所有已知问题 | ✅ 支持中英文 | ✅ PyTorch优化完成  
**最后更新**: 2024年 | **维护状态**: 主动维护



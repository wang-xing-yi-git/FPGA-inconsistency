# FPGA设计文实不一致检测系统

## 项目概述

本项目针对FPGA设计需求文档与代码实现之间的文实脱节问题，采用双向语义提取、语义对齐、双重不一致分析的技术路径，实现显性与隐性不一致的精准检测。

## 系统架构

### 核心模块

1. **双向语义提取 (Bidirectional Semantic Extraction)**
   - 自然语言语义提取：分词、停用词去除、BERT编码
   - FPGA代码语义提取：AST构建、CNN编码

2. **语义对齐机制 (Semantic Alignment)**
   - 相似度计算与规则匹配
   - 映射规则库管理
   - 候选对齐对验证

3. **双重不一致分析 (Dual Inconsistency Analysis)**
   - 显性不一致检测：基于规则库
   - 隐性不一致检测：基于GAT+Bi-GRU模型

## 完整处理流程详解

### 🔄 总体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据输入                                      │
│  JSON格式: {id, req_desc_origin, code_origin}                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   ┌──────────────────┐         ┌──────────────────┐
   │  需求语义提取    │         │  代码语义提取    │
   │ (NLP方向)       │         │ (代码方向)       │
   └────────┬─────────┘         └────────┬─────────┘
            │                            │
   ┌────────▼────────┐         ┌────────▼─────────┐
   │ • 分词处理      │         │ • AST构建        │
   │ • 停用词去除    │         │ • 结构特征提取   │
   │ • BERT编码      │         │ • CNN特征编码    │
   │ • 768维向量     │         │ • 768维向量      │
   └────────┬────────┘         └────────┬─────────┘
            │                           │
            └───────────────┬───────────┘
                            ▼
                    ┌────────────────────┐
                    │   语义对齐 (Align)  │
                    │ • 相似度计算 (35%)  │
                    │ • 映射规则匹配 (35%)│
                    │ • 模式匹配 (30%)    │
                    └────────┬───────────┘
                             ▼
                    ┌────────────────────┐
                    │  不一致检测 (Detect)│
                    │ • 显性不一致        │
                    │ • 隐性不一致        │
                    │ · GAT+Bi-GRU      │
                    └────────┬───────────┘
                             ▼
                    ┌────────────────────┐
                    │   结果生成和报告    │
                    │  JSON/HTML输出     │
                    └────────────────────┘
```

### 📊 逐阶段详解

#### 第1阶段：需求语义提取 (需求端)

**函数调用链**：
```
extract_semantic_elements(req_text)  ---需求文本字符串
    ↓
• 分词与清洗
    ├─ 中文：jieba分词 + 停用词过滤
    ├─ 英文：NLTK分词 + 停用词过滤
    └─ 返回: 词汇列表
    ↓
• 关键词提取
    ├─ 计算每个词的TF-IDF权重
    ├─ 排序并选取Top-N关键词
    └─ 返回: keywords[] 和权重
    ↓
get_semantic_vector(req_text)  ---文本向量化
    ↓
• BERT编码
    ├─ 加载预训练BERT模型
    ├─ 分词→Token IDs
    ├─ 通过BERT网络[768维]
    └─ 返回: 768维语义向量
```

**关键函数**：
- `NLPSemanticExtractor.extract_semantic_elements()`
  - 输入：需求文本字符串
  - 处理：分词、清洗、关键词提取
  - 输出：`{'keywords': [...], 'pos_tags': {...}, 'entities': [...]}`

- `NLPSemanticExtractor.get_semantic_vector()`
  - 输入：需求文本字符串
  - 处理：BERT模型编码
  - 输出：`numpy.ndarray` (768维)

---

#### 第2阶段：代码语义提取 (代码端)

**函数调用链**：
```
extract_semantic_elements(code_text)  ---Verilog代码字符串
    ↓
• 第1层：AST构建
    build_ast(code)
    ├─ 逐行解析Verilog代码
    ├─ 识别: module/port/signal/always/assign
    └─ 构建树结构: {'type': 'root', 'children': [...]}
    ↓
• 第2层：结构化特征提取
    extract_ast_features(ast_root)
    ├─ 模块数、端口计数、信号计数
    ├─ 时序逻辑检测、复位机制检测
    ├─ 代码复杂度评估
    └─ 返回: 特征向量
    ↓
• 第3层：解析关键代码构成
    parse_verilog_code(code)
    ├─ _parse_modules()      提取模块声明
    ├─ _parse_ports()        提取端口 (input/output/inout)
    ├─ _parse_signals()      提取信号声明 (wire/reg)
    ├─ _parse_always_blocks() 提取时序逻辑块
    ├─ _parse_assignments()  提取赋值语句
    └─ 返回: 结构化代码信息
    ↓
get_semantic_vector(code)  ---代码向量化
    ↓
• PyTorch CNN编码
    encode_with_cnn(features)
    ├─ 特征预处理: numpy → PyTorch张量
    ├─ Conv1D层1: [output] → 32通道，激活ReLU
    ├─ Conv1D层2: 32 → 64通道，激活ReLU
    ├─ Conv1D层3: 64 → 128通道，激活ReLU
    ├─ 全局平均池化: [128, L] → [128]
    ├─ Dense层1: 128 → 256维，ReLU激活
    ├─ Dense层2: 256 → 768维，输出层
    └─ 返回: 768维语义向量
```

**关键函数**：
- `CodeSemanticExtractor.build_ast()` [第1子步]
  - 输入：Verilog代码字符串
  - 处理：逐行解析，构建抽象语法树
  - 输出：AST树结构 {'type': 'Module', 'children': [...]}

- `CodeSemanticExtractor.extract_ast_features()` [第2子步]
  - 输入：AST树结构
  - 处理：递归遍历，提取结构特征
  - 输出：NumPy特征数组

- `CodeSemanticExtractor.parse_verilog_code()` [辅助]
  - 包含多个子函数：
    - `_parse_modules()`: 提取module XXX(...)
    - `_parse_ports()`: 提取input/output/inout端口
    - `_parse_signals()`: 提取wire/reg信号
    - `_parse_always_blocks()`: 提取always @(...)块
    - `_parse_assignments()`: 提取 <= 或 = 赋值

- `CodeSemanticExtractor.get_semantic_vector()` [第3子步]
  - 输入：Verilog代码字符串
  - 处理：AST构建 → 特征提取 → CNN编码
  - 输出：768维语义向量

---

#### 第3阶段：语义对齐 (双向融合)

**函数调用链**：
```
SemanticAligner.align_requirements_to_code()
    ↓
• 第1步：相似度计算 (权重35%)
    compute_cosine_similarity(req_vector, code_vector)
    ├─ 需求向量 (768维) vs 代码向量 (768维)
    ├─ 余弦相似度公式: cos(θ) = A·B / (|A||B|)
    ├─ 归一化到 [0, 1] 范围
    └─ 返回: similarity_score (0~1)
    ↓
• 第2步：语义映射规则匹配 (权重35%)
    apply_semantic_mapping_rules()
    ├─ 加载映射规则库 (13条规则)
    ├─ 规则示例：
    │  ├─ 需求中"计数" + 代码中"always" → 高置信度
    │  ├─ 需求中"复位" + 代码中"rst/reset" → 高置信度
    │  └─ ... (共13条规则)
    ├─ 计算规则匹配置信度
    └─ 返回: rule_confidence (0.78~0.95)
    ↓
• 第3步：模式匹配 (权重30%)
    pattern_matching()
    ├─ NLP语法库 (6种模式)
    ├─ 代码语法库 (6种构造)
    ├─ 交叉匹配计分
    └─ 返回: pattern_score (0~1)
    ↓
• 第4步：综合评分
    alignment_score = 
        0.35 × similarity_score +
        0.35 × rule_confidence +
        0.30 × pattern_score
    ↓
    返回: AlignmentResult {
        alignment_pairs: [...],
        confidence: confidence_score,
        metadata: {...}
    }
```

**关键类/函数**：
- `SemanticAligner` 类
  - 初始化时加载：
    - `NLPSyntaxLibrary()`: 自然语言句式库
    - `CodeSyntaxLibrary()`: Verilog构造库
    - `SemanticMappingRulesLibrary()`: 13条映射规则

- `SemanticAligner.compute_cosine_similarity(vec1, vec2)`
  - 输入：两个768维向量
  - 输出：相似度分数 [0, 1]

- `SemanticAligner.align_requirements_to_code()`
  - 输入：需求元素、需求向量、代码元素、代码向量
  - 处理：三维对齐评分
  - 输出：对齐结果对象

---

#### 第4阶段：不一致检测 (双重分析)

**函数调用链**：
```
InconsistencyDetector.detect_all_inconsistencies()
    ↓
├─ 检测1：显性不一致 (基于规则)
│  detect_explicit_inconsistencies()
│  ├─ 名称不匹配: 需求中"计数器"vs代码中其他名字
│  ├─ 功能不符: 需求"8bit计数"but代码"4bit"
│  ├─ 端口缺失: 需求需要端口但代码无
│  ├─ 信号冲突: 需求和代码信号名冲突
│  └─ 返回: explicit_issues[] {type, severity, detail}
│
└─ 检测2：隐性不一致 (基于深度学习)
   detect_implicit_inconsistencies()
   ├─ 输入: 对齐结果 + 向量差异
   ├─ GAT图注意力网络
   │  ├─ 构建代码和需求的图表示
   │  ├─ 节点: 需求术语/代码元素
   │  ├─ 边: 对齐关系
   │  └─ GAT计算注意力权重
   ├─ Bi-GRU双向循环网络
   │  ├─ 前向GRU处理对齐序列
   │  ├─ 后向GRU处理对齐序列
   │  ├─ 融合前后信息
   │  └─ 输出: 隐性不一致预测
   └─ 返回: implicit_issues[] {type, probability, detail}

返回: {
    explicit_inconsistencies: [...],
    implicit_inconsistencies: [...],
    overall_confidence: score,
    recommendations: [...]
}
```

**关键函数**：
- `InconsistencyDetector.detect_explicit_inconsistencies()`
  - 检测：名称/功能/结构层面不一致
  - 基于：规则库和直接对比

- `InconsistencyDetector.detect_implicit_inconsistencies()`
  - 检测：语义层面隐性不一致
  - 基于：GAT + Bi-GRU深度学习模型

---

#### 第5阶段：报告生成 (结果输出)

**函数调用链**：
```
ReportGenerator.generate_report()
    ↓
• 数据聚合
    compose_result()
    ├─ ID: 样本唯一标识
    ├─ 需求摘要: 前200字 + 关键词计数
    ├─ 代码摘要: 前300字 + 模块/端口计数
    ├─ 对齐结果: 候选对列表
    ├─ 显性不一致: 问题列表 + 严重程度
    ├─ 隐性不一致: 预测列表 + 概率
    └─ 综合评分
    ↓
• 输出格式化
    ├─ JSON格式: {id, req_summary, code_summary, ...}
    ├─ HTML格式: 可视化报告 (可选)
    └─ CSV格式: 表格统计 (可选)
    ↓
• 保存输出
    save_report()
    └─ 输出至: reports/report.json
```

**关键函数**：
- `ReportGenerator.generate_report()`
  - 输入：检测结果对象
  - 处理：数据格式化、聚合
  - 输出：JSON/HTML/CSV格式

---

### 🔗 完整函数调用顺序

```python
# main.py - 主程序流程

FPGAInconsistencyDetectionSystem
    └─ process_item(item)  # 处理单个数据项
        │
        ├─ [1] 双向语义提取
        │   ├─ nlp_extractor.extract_semantic_elements(req_text)
        │   ├─ nlp_extractor.get_semantic_vector(req_text)
        │   ├─ code_extractor.extract_semantic_elements(code_text)
        │   │   ├─ build_ast(code_text)
        │   │   ├─ extract_ast_features(ast_root)
        │   │   └─ parse_verilog_code(code_text)
        │   └─ code_extractor.get_semantic_vector(code_text)
        │       ├─ build_ast(code_text)  [重复]
        │       ├─ extract_ast_features(ast_root)  [重复]
        │       └─ encode_with_cnn(features)
        │
        ├─ [2] 语义对齐
        │   └─ aligner.align_requirements_to_code(
        │       req_elements, req_vector, 
        │       code_elements, code_vector
        │   )
        │       ├─ compute_cosine_similarity()  [35%权重]
        │       ├─ apply_semantic_mapping_rules()  [35%权重]
        │       └─ pattern_matching()  [30%权重]
        │
        ├─ [3] 不一致检测
        │   └─ detector.detect_all_inconsistencies(
        │       req_elements, req_vector,
        │       code_elements, code_vector
        │   )
        │       ├─ detect_explicit_inconsistencies()
        │       └─ detect_implicit_inconsistencies()
        │           ├─ GAT网络处理
        │           └─ Bi-GRU网络处理
        │
        └─ [4] 结果整合
            └─ compose_result_dict()
                └─ ReportGenerator.generate_report()
                    └─ save_report(output_path)
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

code_elements = {
    'modules': ['counter'],
    'ports': {'input': ['clk', 'rst'], 'output': ['count']},
    'signals': [{'name': 'temp', 'type': 'wire'}, ...],
    'behaviors': [{'trigger': 'posedge clk'}]
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

## 项目结构

```
FPGA-inconsistency/
├── config.yaml                  # 配置文件
├── requirements.txt             # 依赖包
├── README.md                    # 项目说明
│
├── src/                         # 源代码
│   ├── __init__.py
│   ├── semantic_extraction.py   # 语义提取模块
│   ├── semantic_alignment.py    # 语义对齐模块
│   ├── inconsistency_detector.py # 不一致检测模块
│   ├── models.py                # 深度学习模型
│   ├── rules_engine.py          # 规则引擎
│   ├── data_processor.py        # 数据处理
│   └── report_generator.py      # 报告生成
│
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后数据
│   └── rules/                   # 规则库
│
├── models/                      # 预训练模型存储
│   ├── bert/                    # BERT模型
│   └── trained/                 # 训练好的模型
│
├── reports/                     # 输出报告
│
├── tests/                       # 测试文件
│   ├── __init__.py
│   ├── test_semantic_extraction.py
│   ├── test_alignment.py
│   └── test_detector.py
│
└── main.py                      # 主程序入口

```

## 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt i https://pypi.tuna.tsinghua.edu.cn/simple
python -m spacy download en_core_web_sm -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m spacy download zh_core_web_sm -i https://pypi.tuna.tsinghua.edu.cn/simple
#python -m nltk.downloader stopwords wordnet punkt
```

### 2. 数据准备

准备JSON格式的数据文件 (`data/raw/dataset.json`)：

```json
[
  {
    "id": 1,
    "req_desc_origin": "实现一个具有上升沿触发的异步复位计数器，计数宽度为8bit，计数范围0-255",
    "code_origin": "module counter (clk, rst_n, count); input clk, rst_n; output [7:0] count; ..."
  }
]
```

### 3. 运行检测

```bash
python main.py --input data/raw/dataset.json --output reports/report.json
```

## 核心特性

- ✅ 双向语义提取与编码
- ✅ 精准的语义对齐机制
- ✅ 显性不一致规则检测
- ✅ 隐性不一致深度学习检测
- ✅ 标准化检测报告生成
- ✅ 可扩展的规则库

## 配置说明

详见 `config.yaml` 文件，包含以下主要配置项：

- NLP模型配置
- 代码处理配置
- 对齐算法配置
- GAT/Bi-GRU模型配置
- 输出配置

## 许可证

MIT License

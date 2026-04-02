# 快速开始指南

## 项目概述

这是一个FPGA设计文实不一致检测系统，用于检测需求文档与代码实现之间的不一致。

**新增功能**：语法依赖分析增强的语义提取，可以更精准地理解自然语言中的句法结构。

## 安装与配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载语言模型

#### 英文模型
```bash
# Spacy英文模型
python -m spacy download en_core_web_sm
```

#### 中文模型
```bash
# 中文分词（jieba已包含在requirements中）
# 可选：Spacy中文模型（需要额外安装）
pip install zh_core_web_sm

# 推荐：使用中文BERT模型获得更好效果
# huggingface会在首次使用时自动下载以下模型之一
# - bert-base-chinese
# - hfl/chinese-roberta-wwm-ext
# - hfl/chinese-electra-base
```

### 3. 验证安装

```bash
# 英文演示
python demo_syntax_dependency.py

# 中文演示
python demo_chinese_nlp.py
```

## 使用方法

### 方法1：运行主程序

```bash
# 处理英文数据集（默认）
python main.py --input data/raw/dataset.json --output reports/report.json

# 处理中文数据集
python main.py --input data/raw/dataset_chinese.json --output reports/report_zh.json

# 生成HTML报告
python main.py --input data/raw/dataset.json --output reports/report --format html
```

### 方法2：查看语法依赖分析演示

```bash
# 英文演示
python demo_syntax_dependency.py

# 中文演示
python demo_chinese_nlp.py
```

### 方法3：运行完整示例

```bash
# 运行简单示例和批处理示例
python example_usage.py
```

### 方法4：在Python代码中使用

```python
from main import FPGAInconsistencyDetectionSystem
from src.semantic_extraction import NLPSemanticExtractor

# 方式1：自动语言检测（默认）
extractor = NLPSemanticExtractor(model_name="bert-base-uncased", language="auto")

# 方式2：指定中文BERT模型（获得更好效果）
extractor = NLPSemanticExtractor(model_name="bert-base-chinese", language="zh")

# 方式3：指定英文BERT模型
extractor = NLPSemanticExtractor(model_name="bert-base-uncased", language="en")

# 提取语义
text = "实现一个8位计数器，具有异步复位功能"
elements = extractor.extract_semantic_elements(text)
vector = extractor.get_semantic_vector(text)

# 处理数据集
system = FPGAInconsistencyDetectionSystem()
results = system.process_dataset('data/raw/dataset_chinese.json')
system.save_results(results, 'reports/report_zh.json', format='json')
```

## 数据格式

输入JSON格式：

```json
[
  {
    "id": 1,
    "req_desc_origin": "需求文档文本",
    "code_origin": "FPGA代码文本"
  }
]
```

输出JSON格式：

```json
[
  {
    "id": 1,
    "req_summary": {...},
    "code_summary": {...},
    "alignment": {
      "status": "aligned",
      "similarity_score": 0.85,
      "confidence": 0.88
    },
    "inconsistency_detection": {
      "explicit_inconsistencies": [...],
      "implicit_inconsistencies": [...],
      "total_issues": 2,
      "severity_distribution": {...}
    }
  }
]
```

## 核心模块说明

### semantic_extraction.py - 语义提取（带语法依赖分析）

**NLPSemanticExtractor** - 自然语言语义提取器

核心特性：
- **分词与清理**：文本预处理、停用词去除、词干提取
- **语法依赖分析**：使用spacy进行深层次的句法结构分析
  - POS标签识别（词性标注）
  - 依赖关系提取（主语、谓语、宾语、修饰词等）
  - 依赖树构建（显示词之间的修饰关系）
- **BERT语义编码**：使用预训练的BERT模型生成768维语义向量
- **向量增强**：利用依赖分析结果增强语义向量表示
  - 70% 基础BERT向量
  - 30% 依赖关系增强向量

提取的语义要素包括：
```python
{
    'keywords': [...],              # 关键词列表
    'fpga_terms': [...],            # FPGA领域术语
    'element_type': 'nlp_text',
    'syntax_dependencies': {
        'subjects': [...],          # 主语列表
        'predicates': [...],        # 谓语列表
        'objects': [...],           # 宾语列表
        'modifiers': [...],         # 修饰词列表
        'pos_tags': [...],          # POS标签
        'dependency_pairs': [...]   # 依赖对
    }
}
```

**CodeSemanticExtractor** - FPGA代码语义提取器
- Verilog代码解析
- 结构化特征提取
- FPGA特征识别

### semantic_alignment.py - 语义对齐

- `MappingRulesLibrary`: FPGA领域的映射规则库
- `SemanticAligner`: 对齐需求与代码的语义
- `AlignmentResult`: 对齐结果

### inconsistency_detector.py - 不一致检测

- `RulesEngine`: 显性不一致规则引擎
- `ImplicitInconsistencyDetector`: 隐性不一致检测器
- `InconsistencyDetector`: 统一的不一致检测器

### data_processor.py - 数据处理

- `DataProcessor`: 数据加载、验证、保存
- `ConfigLoader`: 配置文件加载
- `ReportGenerator`: 报告生成

## 配置文件

编辑 `config.yaml` 来自定义系统参数：

```yaml
nlp:
  bert_model: "bert-base-uncased"  # BERT模型
  bert_max_length: 512             # 最大序列长度
  
code:
  ast_encoding_dim: 256            # AST编码维度

alignment:
  similarity_threshold: 0.7        # 相似度阈值
  
inconsistency:
  confidence_threshold: 0.5        # 置信度阈值
```

## 不一致类型

系统检测以下类型的不一致：

- **显性不一致** (Explicit): 基于规则库的明确不一致
  - 存在性不一致: 需求中指定但代码中缺失
  - 匹配性不一致: 需求与代码不匹配
  - 完整性不一致: 实现不完整

- **隐性不一致** (Implicit): 基于语义分析的潜在不一致
  - 语义间隙: 语义向量距离过大
  - 上下文不一致: FPGA特征不一致
  - 行为不一致: 设计行为不一致

## 严重程度分类

- **CRITICAL** (严重): 设计中存在严重错误
- **HIGH** (高): 重要功能实现不完整或不正确
- **MEDIUM** (中): 部分功能或约束不满足
- **LOW** (低): 细微差异或建议改进
- **INFO** (信息): 参考性信息

## 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_semantic_extraction.py -v
python -m pytest tests/test_alignment.py -v
```

## 项目结构

```
FPGA-inconsistency/
├── src/
│   ├── semantic_extraction.py      # 语义提取
│   ├── semantic_alignment.py       # 语义对齐
│   ├── inconsistency_detector.py   # 不一致检测
│   ├── data_processor.py           # 数据处理
│   └── __init__.py
├── tests/
│   ├── test_semantic_extraction.py
│   ├── test_alignment.py
│   └── __init__.py
├── data/
│   ├── raw/
│   │   └── dataset.json            # 示例数据
│   └── rules/
├── reports/                        # 输出报告目录
├── models/
│   └── trained/                    # 训练好的模型存储
├── config.yaml                     # 配置文件
├── requirements.txt                # 依赖包
├── main.py                         # 主程序
├── example_usage.py                # 使用示例
├── README.md                       # 项目说明
└── QUICKSTART.md                   # 本文件
```

## 常见问题

### Q: spacy模型下载慢怎么办？

A: 可以通过以下方式加速：
1. 使用代理或VPN
2. 手动下载后离线安装：
   ```bash
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
   ```
3. 不安装spacy，系统会自动降级使用NLTK进行简单的POS标注

### Q: 为什么语法依赖分析很重要？

A: 语法依赖分析可以：
1. **更好理解句法结构**：识别主语、谓语、宾语等成分，避免表面相似但含义不同的文本被误认为相关
2. **提取精确的语义关系**：理解"A控制B"和"A由B控制"这两种表达中不同的语义关系
3. **增强语义向量**：基于句法结构调整向量权重，使其更准确地反映文本含义
4. **FPGA域特定性**：识别FPGA领域术语如何被修饰和组织，例如"异步复位"vs "同步复位"

### Q: BERT模型下载慢怎么办？

A: 可以通过以下方式加速：
1. 使用代理或VPN
2. 手动下载模型后放在 `~/.cache/huggingface/hub/` 目录
3. 在代码中指定本地模型路径

### Q: 内存不足怎么办？

A: 可以在配置中减小模型参数：
- 降低 `bert_max_length`
- 减小 `embedding_dim`
- 使用较小的BERT模型

### Q: 如何添加新的检测规则？

A: 编辑 `inconsistency_detector.py` 中的 `RulesEngine._initialize_rules()` 方法，添加新的规则字典。

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或Pull Request。

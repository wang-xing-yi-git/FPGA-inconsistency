# 【修改】快速参考 - FPGA语义系统升级清单

## 🎯 三大核心改进

### 1️⃣ 代码语义提取升级【修改】

**Old**  ❌
```python
# 简单正则表达式 → 8维特征 → 随机扩展
features = [module_count, port_count, ...]
vector = random_expand(features, 768)  # ❌ 信息损失
```

**New** ✅
```python
# Verilog → AST → 256维特征 → CNN编码 → 768维向量
ast = extractor.build_ast(code)              # 【修改】AST构建
features = extractor.extract_ast_features(ast)
vector = extractor.encode_with_cnn(features) # 【修改】CNN编码
# ✅ 完整捕捉逻辑功能和约束条件
```

**关键方法**【修改】：
- `build_ast()` - 完整AST构建
- `extract_ast_features()` - 256维结构化特征
- `encode_with_cnn()` - CNN深度编码
- `_parse_modules/ports/signals/always/assignments()` - 节点解析

---

### 2️⃣ 语义对齐升级【修改】

**Old** ❌
```python
# 仅使用向量相似度 + 简单规则
similarity = cosine_similarity(vec1, vec2)
status = ALIGNED if similarity > 0.7 else UNALIGNED
```

**New** ✅
```python
# 多维度综合 = 向量相似度(35%) + 映射置信度(35%) + 模式匹配(30%)
composite_score = (
    similarity * 0.35 +
    mapping_confidence * 0.35 +
    pattern_match_score * 0.30
)
# 基于composite_score进行多阈值决策
```

**关键库**【修改】：
- `NLPSyntaxLibrary` - 自然语言语义模式
- `CodeSyntaxLibrary` - Verilog语法构造
- `SemanticMappingRulesLibrary` - 13条映射规则

---

### 3️⃣ 对齐算法增强【修改】

**Four-Stage Pipeline**：

```
【Stage 1】特征提取与编码
  ├─ NLP特征 + 代码向量
  └─ 检测语义模式 + 代码构造

【Stage 2】规则匹配
  ├─ 13条语义映射规则
  └─ 计算映射置信度

【Stage 3】多维度评分
  ├─ 向量相似度 (cosine)
  ├─ 语义映射置信度 (规则库)
  └─ 语法模式匹配度 (双语法库)

【Stage 4】综合决策
  ├─ score ≥ 0.85 ➜ ALIGNED
  ├─ score ≥ 0.75 ➜ ALIGNED
  ├─ score ≥ 0.65 ➜ SUSPICIOUS
  └─ score < 0.65 ➜ UNALIGNED/INDIRECT
```

---

## 📊 改进统计表

### semantic_extraction.py 【修改】

| 项目 | 改进前 | 改进后 | 说明 |
|------|--------|--------|------|
| 代码解析方式 | 正则表达式 | AST | 从浅层到深层 |
| 特征维度 | 8维 | 256维 | 40倍提升 |
| 编码方式 | 随机 | CNN深度学习 | 机器学习驱动 |
| AST支持 | ❌ | ✅ | 新增 |
| 逻辑表征 | ❌ | ✅ | 通过AST |
| 约束建模 | ❌ | ✅ | 通过赋值和参数 |
| 复杂度评估 | ❌ | ✅ | 动态计算 |
| 新增方法 | 0 | 15+ | 见下表 |

**新增方法清单**【修改】：
1. `_init_verilog_syntax_rules()`
2. `_build_cnn_model()`
3. `build_ast()`
4. `_parse_modules()`
5. `_parse_ports()`
6. `_parse_signals()`
7. `_parse_always_blocks()`
8. `_parse_instantiations()`
9. `_parse_assignments()`
10. `_estimate_complexity()`
11. `_calculate_nesting_depth()`
12. `_extract_module_content()`
13. `extract_ast_features()`
14. `encode_with_cnn()`
15. `_encode_features_fallback()`

### semantic_alignment.py 【修改】

| 项目 | 改进前 | 改进后 | 说明 |
|------|--------|--------|------|
| 对齐维度 | 1 | 3 | 向量 + 映射 + 模式 |
| 规则数量 | 10 | 23 | 新增语义映射规则 |
| 语法库 | 0 | 2 | NLP + Code |
| 映射规则库 | ❌ | ✅ | 13条增强规则 |
| 中文支持 | 部分 | 完全 | 完整语义库 |
| 新增类 | 0 | 3 | 3个语法库 |
| 新增字段 | 0 | 1 | mapping_confidence |
| 决策阈值 | 1 | 4 | 多级决策 |

**新增类清单**【修改】：
1. `NLPSyntaxLibrary` - 自然语言语法库
2. `CodeSyntaxLibrary` - Verilog语法库
3. `SemanticMappingRulesLibrary` - 映射规则库

---

## 🔑 关键参数与常量

### CNN模型参数【修改】

```python
# 输入
input_shape = (256, 1)

# 网络结构
conv1_filters = 32, kernel_size = 3
pool1_size = 2
conv2_filters = 64, kernel_size = 3
pool2_size = 2
conv3_filters = 128, kernel_size = 3

# 完全连接层
dense1_units = 256
dense2_units = 768  # 最终输出维度

# 激活函数
activation = 'relu' for Conv & Dense
activation = 'linear' for final Dense layer
```

### 对齐决策阈值【修改】

```python
# 综合评分权重
sim_weight = 0.35
mapping_weight = 0.35
pattern_weight = 0.30

# 对齐阈值（按优先级）
ALIGNED_HIGH = 0.85        # 高质量对齐
ALIGNED = 0.75             # 良好对齐
SUSPICIOUS = 0.65          # 可疑对齐
UNALIGNED = 0.0            # 低于此为未对齐

# 类别判定规则
if score >= 0.85 AND semantic_mappings AND rules:
    status = ALIGNED (confidence = max)
elif score >= 0.75 AND (semantic_mappings OR rules):
    status = ALIGNED (confidence = high)
elif score >= 0.65 AND semantic_mappings:
    status = SUSPICIOUS (confidence = medium)
else:
    status = INDIRECT/UNALIGNED (confidence = low)
```

### 映射规则置信度【修改】

```python
# 基础规则置信度范围
rule_confidence_min = 0.78   # rule_timing_delay
rule_confidence_max = 0.95   # rule_timing_clock

# 映射置信度计算
mapping_conf = rule.score * match_ratio
match_ratio = (nlp_matches + code_matches) / (nlp_patterns + code_patterns)

# 最终使用
final_conf = min(mapping_conf, 1.0)
```

---

## 📦 类和方法导航

### CodeSemanticExtractor 【修改】

```
class CodeSemanticExtractor:
    ├─ __init__() 
    │  ├─ self.cnn_model = _build_cnn_model()           【修改】
    │  └─ self.verilog_syntax_rules = _init_verilog_syntax_rules()  【修改】
    │
    ├─ _init_verilog_syntax_rules()                      【修改】新增
    ├─ _build_cnn_model()                                【修改】新增
    │
    ├─ build_ast(code)                                   【修改】新增
    │  ├─ _parse_modules(code)                           【修改】新增
    │  ├─ _parse_ports(module_code)                      【修改】新增
    │  ├─ _parse_signals(module_code)                    【修改】新增
    │  ├─ _parse_always_blocks(module_code)              【修改】新增
    │  ├─ _parse_instantiations(module_code)             【修改】新增
    │  ├─ _parse_assignments(module_code)                【修改】新增
    │  └─ _estimate_complexity(code)                     【修改】新增
    │
    ├─ extract_ast_features(ast_node)                    【修改】新增
    │  └─ traverse(node) 递归统计
    │
    ├─ encode_with_cnn(features)                         【修改】新增
    │  └─ _encode_features_fallback(features)            【修改】新增
    │
    ├─ extract_semantic_elements(code)                   【修改】改进
    │  ├─ build_ast()                                    【修改】新增调用
    │  └─ _ast_to_dict()                                 【修改】新增
    │
    ├─ get_semantic_vector(code)  ← Main Method!         【修改】改进
    │  ├─ build_ast()                                    【修改】
    │  ├─ extract_ast_features()                         【修改】
    │  └─ encode_with_cnn()                              【修改】
    │
    ├─ parse_verilog_code(code)
    ├─ _extract_keywords(code)
    └─ _extract_fpga_features(code)
```

### SemanticAligner 【修改】

```
class SemanticAligner:
    ├─ __init__()
    │  ├─ self.nlp_syntax = NLPSyntaxLibrary()          【修改】新增
    │  ├─ self.code_syntax = CodeSyntaxLibrary()        【修改】新增
    │  ├─ self.semantic_mapping_lib = SemanticMappingRulesLibrary()  【修改】新增
    │  └─ self.rules_library = MappingRulesLibrary()    （保留）
    │
    ├─ align_requirements_to_code()  ← Main Method!      【修改】改进
    │  │  新增参数：req_text, code_text
    │  ├─ compute_cosine_similarity()
    │  ├─ nlp_syntax.extract_semantic_patterns()        【修改】新增调用
    │  ├─ code_syntax.extract_code_constructs()         【修改】新增调用
    │  ├─ semantic_mapping_lib.find_semantic_mappings() 【修改】新增调用
    │  ├─ _determine_alignment_status()                 【修改】增强
    │  └─ return AlignmentResult  (含mapping_confidence) 【修改】
    │
    ├─ _determine_alignment_status()                     【修改】增强
    │  │  新增参数：semantic_mappings, mapping_confidence, nlp_patterns, code_constructs
    │  ├─ 计算综合评分
    │  ├─ 进行多级阈值判决
    │  └─ return (status, confidence, reason)
    │
    ├─ compute_cosine_similarity()
    └─ batch_align()
```

### NLPSyntaxLibrary 【修改】新增

```
class NLPSyntaxLibrary:
    ├─ __init__()
    │  ├─ self.chinese_patterns = {...}
    │  │  ├─ timing_constraints
    │  │  ├─ io_specifications
    │  │  ├─ memory_operations
    │  │  ├─ control_logic
    │  │  ├─ datapath
    │  │  └─ synchronization
    │  └─ self.english_patterns = {...}  (同上)
    │
    └─ extract_semantic_patterns(text) → Dict
       ├─ 自动语言检测 (中/英)
       ├─ 匹配关键字
       └─ return {pattern_name: [matched_keywords]}
```

### CodeSyntaxLibrary 【修改】新增

```
class CodeSyntaxLibrary:
    ├─ __init__()
    │  └─ self.verilog_constructs = {
    │     ├─ timing_logic
    │     ├─ io_declarations
    │     ├─ memory_structures
    │     ├─ control_structures
    │     ├─ datapath_logic
    │     └─ synchronization
    │     }
    │
    └─ extract_code_constructs(code) → Dict
       ├─ 正则匹配
       ├─ 返回构造+示例
       └─ return {construct_name: [matched_patterns]}
```

### SemanticMappingRulesLibrary 【修改】新增

```
class SemanticMappingRulesLibrary:
    ├─ __init__()
    │  └─ self.mapping_rules = _initialize_mapping_rules()
    │     (13条映射规则)
    │
    ├─ find_semantic_mappings(nlp_keywords, code_constructs) → List[Dict]
    │  └─ return [Rule1, Rule2, ...] (匹配的规则)
    │
    └─ calculate_mapping_confidence(rule, nlp_kw, code_con) → float
       └─ return confidence ∈ [0, 1]
```

---

## 💡 使用示例

### 基本流程【修改】

```python
# Step 1: 导入
from semantic_extraction import CodeSemanticExtractor, extract_bidirectional_semantics
from semantic_alignment import SemanticAligner

# Step 2: 准备输入
req_text = "FPGA双端口RAM模块，数据位宽8比特..."
code_text = "module Bus8_DPRAM #(DEPTH=256)... endmodule"

# Step 3: 【修改】语义提取 (AST + CNN)
extraction = extract_bidirectional_semantics(req_text, code_text)

# Step 4: 【修改】语义对齐 (多维度)
aligner = SemanticAligner()
result = aligner.align_requirements_to_code(
    req_id=1,
    req_elements=extraction['requirement']['semantic_elements'],
    req_vector=extraction['requirement']['semantic_vector'],
    code_elements=extraction['code']['semantic_elements'],
    code_vector=extraction['code']['semantic_vector'],
    code_segment=code_text,
    req_text=req_text,       # 【修改】新增
    code_text=code_text      # 【修改】新增
)

# Step 5: 观察结果
print(f"Status: {result.status.value}")
print(f"Confidence: {result.confidence:.3f}")
print(f"【修改】Mapping Confidence: {result.mapping_confidence:.3f}")
```

### 访问新增特性【修改】

```python
# 访问AST信息
ast_nodes = extraction['code']['semantic_elements']['ast_nodes']
print(f"AST Type: {ast_nodes['type']}")
print(f"Children: {ast_nodes['children_summary']}")

# 访问复杂度
complexity = extraction['code']['semantic_elements']['code_complexity']
print(f"Code Complexity: {complexity:.2%}")

# 访问映射置信度
map_conf = result.mapping_confidence
print(f"Mapping Score: {map_conf:.3f}")

# 访问对齐原因
print(f"Reason: {result.reason}")
```

---

## ✅ 改进验证清单

核实所有【修改】是否已实现：

### semantic_extraction.py 【修改】验证清单

- ✅ TensorFlow导入（try-except）
- ✅ CodeSemanticExtractor.__init__ 增强
- ✅ _init_verilog_syntax_rules 新增
- ✅ _build_cnn_model 新增
- ✅ build_ast 新增
- ✅ extract_ast_features 新增
- ✅ encode_with_cnn 新增
- ✅ _encode_features_fallback 新增
- ✅ 各 _parse_* 方法新增
- ✅ extract_semantic_elements 改进
- ✅ get_semantic_vector 改进

### semantic_alignment.py 【修改】验证清单

- ✅ AlignmentResult.mapping_confidence 新增
- ✅ NLPSyntaxLibrary 新增
- ✅ CodeSyntaxLibrary 新增
- ✅ SemanticMappingRulesLibrary 新增
- ✅ SemanticAligner.__init__ 增强
- ✅ align_requirements_to_code 增强（new params）
- ✅ _determine_alignment_status 增强（new logic）
- ✅ align_semantics 增强（new params）
- ✅ main 程序演示增强

---

## 🚀 快速开始

### 运行演示

```bash
python src/semantic_alignment.py
```

### 查看文档

- `IMPROVEMENTS_SUMMARY.md` - 详细改进说明
- `MODIFICATION_GUIDE.md` - 完整使用指南
- `MODIFICATION_CHECKLIST.md` - 此页（快速参考）

### 测试改进

```python
# 测试AST
from semantic_extraction import CodeSemanticExtractor
ec = CodeSemanticExtractor()
ast = ec.build_ast("module test(input clk); endmodule")
assert ast['type'] == 'root'
print("✅ AST Test Passed")

# 测试语法库
from semantic_alignment import NLPSyntaxLibrary
nl = NLPSyntaxLibrary()
patterns = nl.extract_semantic_patterns("时钟信号")
assert len(patterns) > 0
print("✅ Syntax Library Test Passed")

# 测试对齐
from semantic_alignment import SemanticAligner
al = SemanticAligner()
assert hasattr(al, 'nlp_syntax')
assert hasattr(al, 'code_syntax')
assert hasattr(al, 'semantic_mapping_lib')
print("✅ Aligner Test Passed")
```

---

## 版本信息

- **版本**：2.0 Enhanced
- **更新日期**：2026年3月4日
- **所有修改**：标注为【修改】
- **兼容性**：完全向后兼容
- **新增代码行**：~800行
- **改进方法**：15+ 个新方法
- **新增类**：3 个
- **新增规则**：13 条

---

**📄 文档**：
- 此页 = 快速参考
- IMPROVEMENTS_SUMMARY.md = 详细总结
- MODIFICATION_GUIDE.md = 完整使用指南
- semantic_extraction.py = 代码实现（搜索【修改】）
- semantic_alignment.py = 代码实现（搜索【修改】）

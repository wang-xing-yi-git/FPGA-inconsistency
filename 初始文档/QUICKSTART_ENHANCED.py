#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始指南 - FPGA需求-代码一致性检测系统
增强的语义提取功能集成教程

这是一个实际的代码示例，展示如何使用新的语义提取功能
"""

# ============================================================================
# 第1步: 导入新的语义提取工具
# ============================================================================

from src.semantic_extraction import (
    AttentionMechanism,  # 注意力机制
    SentenceVectorAggregator,  # 句向量聚合
    EnhancedSemanticElementExtractor,  # 要素提取
    NLPSemanticExtractor,  # 增强的NLP提取
)
import numpy as np

print(
    """
╔════════════════════════════════════════════════════════════════╗
║     FPGA需求-代码一致性检测系统                               ║
║     增强的语义提取功能 - 快速开始指南                         ║
╚════════════════════════════════════════════════════════════════╝
"""
)

# ============================================================================
# 第2步: 初始化工具
# ============================================================================

print("\n[步骤1] 初始化语义提取工具...")

# 创建NLP提取器（自动初始化所有组件）
extractor = NLPSemanticExtractor(language="auto")  # 自动检测语言

print("✓ 提取器初始化完成")
print(f"  - 注意力机制: {type(extractor.attention).__name__}")
print(f"  - 聚合器: {type(extractor.aggregator).__name__}")
print(f"  - 要素提取器: {type(extractor.element_extractor).__name__}")

# ============================================================================
# 第3步: 使用场景1 - 长文本语义向量生成
# ============================================================================

print("\n" + "=" * 60)
print("[使用场景1] 长文本语义向量生成")
print("=" * 60)

# 多句需求文本示例
requirement_text = """
设计FPGA双端口RAM模块。
该模块应具有以下特点：
1. 数据位宽为8比特，地址宽度10比特
2. 支持单口时钟设计，频率为100MHz
3. 实现同步读写操作
4. 集成复位信号
"""

print("\n📝 需求文本:")
print(f"   {requirement_text[:100]}...")

try:
    # 方法1: 简单平均 (推荐用于快速处理)
    vector_avg = extractor.get_semantic_vector_for_long_text(
        requirement_text, method="sentence_average"
    )
    print(f"\n✓ 方法1: sentence_average")
    print(f"  向量维度: {vector_avg.shape}")
    print(f"  向量范数: {np.linalg.norm(vector_avg):.4f}")

    # 方法2: 加权平均 (关键句重点)
    vector_weighted = extractor.get_semantic_vector_for_long_text(
        requirement_text, method="weighted_attention"
    )
    print(f"\n✓ 方法2: weighted_attention")
    print(f"  向量维度: {vector_weighted.shape}")

    # 方法3: 最大池化 (信息最丰富)
    vector_max = extractor.get_semantic_vector_for_long_text(
        requirement_text, method="max_pooling"
    )
    print(f"\n✓ 方法3: max_pooling")
    print(f"  向量维度: {vector_max.shape}")

    # 计算向量间的相似度
    from scipy.spatial.distance import cosine

    sim_avg_weighted = 1 - cosine(vector_avg, vector_weighted)
    sim_avg_max = 1 - cosine(vector_avg, vector_max)

    print(f"\n📊 方法间相似度:")
    print(f"  average ↔ weighted: {sim_avg_weighted:.4f}")
    print(f"  average ↔ max:     {sim_avg_max:.4f}")

except ImportError:
    print("✗ BERT模型未下载，跳过向量生成")
    print("  提示: 首次使用会自动下载BERT模型 (~500MB)")

# ============================================================================
# 第4步: 使用场景2 - 注意力机制
# ============================================================================

print("\n" + "=" * 60)
print("[使用场景2] 注意力机制 - 关键词突出")
print("=" * 60)

# 创建示例向量（10个tokens）
sample_tokens = np.random.randn(10, 768)

# 计算注意力权重
attention_mechanism = AttentionMechanism(attention_type="scaled_dot_product")
weights = attention_mechanism.compute_attention_weights(sample_tokens)

print("\n🎯 注意力权重计算:")
print(f"  Token数量: {len(weights)}")
print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"  权重和: {weights.sum():.6f}")

# 找出最重要的tokens
top_3_indices = np.argsort(weights)[-3:][::-1]
print(f"\n  排名前3的tokens:")
for i, idx in enumerate(top_3_indices, 1):
    print(f"    {i}. Token #{idx}: 权重={weights[idx]:.4f}")

# ============================================================================
# 第5步: 使用场景3 - 完整语义要素提取
# ============================================================================

print("\n" + "=" * 60)
print("[使用场景3] 完整语义要素提取")
print("=" * 60)

# 进行完整的语义要素提取
try:
    result = extractor.extract_complete_semantic_elements(
        requirement_text, requirement_id=1
    )

    print("\n✓ 要素提取成功")

    # 显示统计信息
    print(f"\n📊 要素统计:")
    print(f"  总要素数: {result['elements_summary']['total_count']}")
    if result["elements_summary"]["by_type"]:
        print(f"  按类型分布:")
        for elem_type, count in result["elements_summary"]["by_type"].items():
            print(f"    - {elem_type}: {count}")

    # 显示提取的参数
    if result["parameters"]:
        print(f"\n⚙️ 自动提取的参数:")
        for param, value in result["parameters"].items():
            if value is not None:
                print(f"    - {param}: {value}")

    # 显示置信度统计
    print(f"\n📈 置信度统计:")
    stats = result["statistics"]
    print(f"  平均置信度: {stats['avg_confidence']:.4f}")
    print(f"  最小置信度: {stats['min_confidence']:.4f}")
    print(f"  最大置信度: {stats['max_confidence']:.4f}")

    # 显示前3个要素详情
    if result["elements"]:
        print(f"\n🔍 前3个提取的要素:")
        for i, elem in enumerate(result["elements"][:3], 1):
            print(f"\n  {i}. {elem.get('value', 'N/A')}")
            print(f"     类型: {elem.get('type', 'N/A')}")
            print(f"     置信度: {elem.get('confidence', 0):.2f}")
            if elem.get("parameter"):
                print(f"     参数: {elem.get('parameter')}")

except Exception as e:
    print(f"✗ 要素提取出错: {e}")
    print("  提试试查看日志或检查文本格式")

# ============================================================================
# 第6步: 使用场景4 - 句向量聚合策略比较
# ============================================================================

print("\n" + "=" * 60)
print("[使用场景4] 句向量聚合策略")
print("=" * 60)

# 创建示例句向量 (5句，每句768维)
sentence_vectors = [np.random.randn(768) for _ in range(5)]

aggregator = SentenceVectorAggregator()

print("\n📊 比较不同的聚合方法:")

methods_info = {
    "mean": "简单平均 - 快速、通用",
    "weighted": "加权平均 - 关键句优先",
    "max": "最大池化 - 信息最丰富",
}

for method, description in methods_info.items():
    doc_vector = aggregator.aggregate_multi_sentences(sentence_vectors, method=method)
    norm = np.linalg.norm(doc_vector)
    print(f"\n  {method:10s}: {description}")
    print(f"    - 输出维度: {doc_vector.shape}")
    print(f"    - 向量范数: {norm:.4f}")

# ============================================================================
# 第7步: 实际应用示例 - 集成到不一致检测
# ============================================================================

print("\n" + "=" * 60)
print("[实际应用] 集成到不一致检测流程")
print("=" * 60)

print(
    """
使用新功能进行需求-代码一致性检测的推荐流程:

1️⃣ 需求处理:
   requirement_text = "..."  # 多句需求
   req_vector = extractor.get_semantic_vector_for_long_text(
       requirement_text,
       method="sentence_average"
   )  # → (768,) 向量
   
   req_elements = extractor.extract_complete_semantic_elements(
       requirement_text,
       requirement_id=req_id
   )  # → 完整要素dict

2️⃣ 代码处理:
   code_vector = extractor.extract_semantic_vector(code_text)
   code_elements = extractor.extract_elements(code_text)

3️⃣ 对比计算:
   # 向量相似度
   vector_similarity = cosine_similarity(req_vector, code_vector)
   
   # 要素对齐度
   element_alignment = compute_alignment(req_elements, code_elements)
   
   # 综合不一致性
   inconsistency_score = (1 - vector_similarity) * w1 + 
                         (1 - element_alignment) * w2

4️⃣ 结果输出:
   {
       'requirement_id': 1,
       'inconsistency_score': 0.15,
       'missing_elements': [...],
       'parameter_mismatches': [...],
       'recommendations': [...]
   }
"""
)

# ============================================================================
# 第8步: 配置和调优建议
# ============================================================================

print("\n" + "=" * 60)
print("[配置建议]")
print("=" * 60)

print(
    """
根据您的应用场景选择:

🔸 快速处理 (实时响应):
   - 使用 method="sentence_average"
   - 聚合器: aggregation_method="mean"
   - 跳过注意力计算
   
🔸 高精度 (批处理):
   - 使用 method="weighted_attention"
   - 聚合器: aggregation_method="weighted"
   - 启用注意力加权
   
🔸 GPU加速 (有CUDA):
   - 系统自动启用 torch.cuda.is_available()
   - 速度提升 5-10倍
   
🔸 内存受限:
   - 使用lightweigh模型: model_name="distilbert-base-uncased"
   - 批大小: batch_size=1
"""
)

# ============================================================================
# 第9步: 常见操作速查表
# ============================================================================

print("\n" + "=" * 60)
print("[操作速查表]")
print("=" * 60)

operations = {
    "1. 生成长文本向量": """
    vector = extractor.get_semantic_vector_for_long_text(text)
    """,
    "2. 提取语义要素": """
    result = extractor.extract_complete_semantic_elements(text, requirement_id=1)
    elements = result['elements']
    """,
    "3. 使用注意力机制": """
    attention = AttentionMechanism()
    weights = attention.compute_attention_weights(embeddings)
    """,
    "4. 聚合句向量": """
    aggregator = SentenceVectorAggregator()
    doc_vec = aggregator.aggregate_multi_sentences(sent_vecs)
    """,
    "5. 中文特化": """
    extractor = NLPSemanticExtractor(language="zh")
    # 或者使用auto: NLPSemanticExtractor(language="auto")
    """,
    "6. 获取统计信息": """
    result = extractor.extract_complete_semantic_elements(text)
    stats = result['statistics']  # avg_confidence, min, max等
    """,
}

for title, code in operations.items():
    print(f"\n{title}")
    print("-" * 50)
    print(code.strip())

# ============================================================================
# 第10步: 下一步行动
# ============================================================================

print("\n" + "=" * 60)
print("[下一步行动]")
print("=" * 60)

print(
    """
✅ 现在您已经了解了新功能！

📚 建议的后续步骤:

1. 阅读详细文档:
   - SEMANTIC_ENHANCEMENT_GUIDE.md - 完整API文档
   - IMPLEMENTATION_SUMMARY.md - 实现细节

2. 运行演示脚本:
   - python demo_enhanced_semantic.py - 完整演示
   - python minimal_verify.py - 快速验证

3. 集成到项目:
   - 更新 main.py 使用新的向量生成
   - 更新 inconsistency_detector.py 使用完整要素
   - 修改 semantic_alignment.py 使用新的对齐方法

4. 调试和优化:
   - 根据您的数据调整超参数
   - 测试不同的聚合方法
   - 监控性能指标

💡 联系方式:
   - 查看 PROJECT_CHECKLIST.md 了解项目完整情况
   - 查看 IMPLEMENTATION_COMPLETE.md 了解技术细节
   - 可视化文档和脚本位于项目根目录
"""
)

print("\n" + "=" * 60)
print("🎉 开始体验新的语义提取功能吧！")
print("=" * 60 + "\n")

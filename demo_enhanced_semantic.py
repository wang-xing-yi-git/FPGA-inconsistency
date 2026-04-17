#!/usr/bin/env python3
"""
演示脚本：展示增强的语义提取功能
包括：
1. 长文本句向量平均法
2. 注意力机制聚焦关键信息
3. 完整的语义要素提取（类型、值、需求编号等）
"""

import json
import numpy as np
from src.semantic_extraction import (
    NLPSemanticExtractor,
    AttentionMechanism,
    SentenceVectorAggregator,
    EnhancedSemanticElementExtractor,
)


def demo_long_text_semantic_vector():
    """演示1：长文本句向量平均法"""
    print("\n" + "=" * 100)
    print("【演示1】长文本句向量平均法 - 确保向量表征文本核心含义")
    print("=" * 100)

    # 多句需求描述（来自FPGA双端口RAM模块）
    long_text = (
        "FPGA双端口RAM模块，数据位宽固定为8比特。"
        "采用单总线时钟实现双端口RAM逻辑。"
        "端口A与总线绑定，端口B为通用业务端口。"
        "总线侧读写控制规则：在1个时钟周期内同时置位片选信号、8位地址信号、读写控制信号。"
        "写操作时序：写数据在寻址时立即被写入对应内存地址。"
        "读操作时序：读请求触发后，有效数据标志信号延迟1个时钟周期脉冲。"
        "此时总线读数据端口输出对应内存数据。"
        "模块可配置参数：DEPTH为双端口RAM的存储深度，代表存储的8比特字长数据的个数。"
    )

    print(f"\n输入长文本 ({len(long_text)} 字符)：")
    print(f"  {long_text[:80]}...")

    # 初始化提取器
    extractor = NLPSemanticExtractor(language="auto")

    # 【方法1】句向量平均法（推荐）- 表征核心含义
    print("\n【方法1】句向量平均法（推荐）：")
    vector_mean = extractor.get_semantic_vector_for_long_text(
        long_text, method="sentence_average"
    )
    if vector_mean is not None:
        print(f"  ✓ 生成整体语义向量：{vector_mean.shape}")
        print(f"    向量范数: {np.linalg.norm(vector_mean):.4f} (已规范化)")
        print(f"    前10维: {vector_mean[:10]}")

    # 【方法2】注意力加权平均 - 聚焦关键句
    print("\n【方法2】注意力加权平均（聚焦关键句）：")
    vector_attention = extractor.get_semantic_vector_for_long_text(
        long_text, method="weighted_attention"
    )
    if vector_attention is not None:
        print(f"  ✓ 生成聚焦向量：{vector_attention.shape}")
        print(
            f"    与平均向量的余弦相似度: {np.dot(vector_mean, vector_attention):.4f}"
        )

    # 【方法3】最大池化 - 保留信息最丰富句
    print("\n【方法3】最大池化（信息最丰富句）：")
    vector_max = extractor.get_semantic_vector_for_long_text(
        long_text, method="max_pooling"
    )
    if vector_max is not None:
        print(f"  ✓ 生成最大向量：{vector_max.shape}")
        print(f"    与平均向量的余弦相似度: {np.dot(vector_mean, vector_max):.4f}")


def demo_attention_mechanism():
    """演示2：注意力机制聚焦关键信息"""
    print("\n" + "=" * 100)
    print("【演示2】注意力机制 - 聚焦文本中的关键信息")
    print("=" * 100)

    # 创建模拟的token嵌入
    num_tokens = 20
    embedding_dim = 768
    token_embeddings = np.random.randn(num_tokens, embedding_dim)
    token_embeddings = token_embeddings / np.linalg.norm(
        token_embeddings, axis=1, keepdims=True
    )

    # 模拟关键词（赋予更高的相似度）
    query = token_embeddings[5]  # 使用第5个token作为查询

    # 初始化注意力机制
    attention = AttentionMechanism(attention_type="scaled_dot_product")

    # 计算注意力权重
    weights = attention.compute_attention_weights(token_embeddings, query)

    print(f"\n Token个数：{num_tokens}")
    print(f" 嵌入维度：{embedding_dim}")
    print(f"\n 注意力权重统计：")
    print(f"   平均权重: {np.mean(weights):.6f}")
    print(f"   最大权重: {np.max(weights):.6f}")
    print(f"   最小权重: {np.min(weights):.6f}")
    print(f"   标准差: {np.std(weights):.6f}")

    # 找出最受关注的tokens
    top_k = 5
    top_indices = np.argsort(weights)[-top_k:][::-1]
    print(f"\n 前{top_k}个最受关注的tokens：")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. Token {idx}: 权重 = {weights[idx]:.6f}")


def demo_complete_semantic_elements():
    """演示3：完整的语义要素提取"""
    print("\n" + "=" * 100)
    print("【演示3】完整的语义要素提取 - 提取类型、值、需求编号等")
    print("=" * 100)

    # 需求文本
    requirement_text = (
        "设计一个8位同步计数器模块。"
        "输入信号包括时钟信号clk、复位信号rst_n、使能信号enable。"
        "输出8位计数值count。"
        "功能要求：在使能信号为高时，每个时钟上升沿计数器加1。"
        "当计数器达到255时，复位到0。"
        "支持异步复位功能，rst_n为低时计数器复位。"
    )

    print(f"\n需求文本：")
    print(f"  {requirement_text}\n")

    # 初始化提取器
    extractor = NLPSemanticExtractor(language="auto")

    # 提取完整的语义要素
    result = extractor.extract_complete_semantic_elements(
        requirement_text, requirement_id=1
    )

    print(f"✓ 提取结果统计：")
    print(f"  需求编号: {result['requirement_id']}")
    print(f"  提取方法: {result['extraction_method']}")
    print(f"  要素总数: {result['elements_summary']['total_count']}")

    print(f"\n✓ 要素类型分布：")
    for element_type, count in result["elements_summary"]["by_type"].items():
        print(f"  {element_type}: {count} 个")

    print(f"\n✓ 提取的参数：")
    if result["parameters"]:
        for param_name, param_value in result["parameters"].items():
            print(f"  {param_name}: {param_value}")
    else:
        print("  (未提取到具体参数值)")

    print(f"\n✓ 要素置信度统计：")
    stats = result["statistics"]
    print(f"  平均置信度: {stats['avg_confidence']:.4f}")
    print(f"  最高置信度: {stats['max_confidence']:.4f}")
    print(f"  最低置信度: {stats['min_confidence']:.4f}")

    # 展示前5个提取的要素
    print(f"\n✓ 前5个提取的要素示例：")
    for i, elem in enumerate(result["elements"][:5], 1):
        print(f"\n  {i}. 类型: {elem['type']}")
        print(f"     值: {elem['value']}")
        print(f"     上下文: ...{elem['context']}...")
        print(f"     置信度: {elem['confidence']:.4f}")
        if elem.get("parameter"):
            print(f"     参数: {elem['parameter']}")


def demo_sentence_vector_aggregator():
    """演示4：句向量聚合器"""
    print("\n" + "=" * 100)
    print("【演示4】句向量聚合器 - 多种聚合策略")
    print("=" * 100)

    # 模拟多个句子的向量
    num_sentences = 5
    embedding_dim = 768

    # 生成5个不同的句向量
    sentence_vectors = [np.random.randn(embedding_dim) for _ in range(num_sentences)]

    # 规范化
    for i in range(len(sentence_vectors)):
        sentence_vectors[i] = sentence_vectors[i] / np.linalg.norm(sentence_vectors[i])

    print(f"\n输入：")
    print(f"  句子数: {num_sentences}")
    print(f"  向量维度: {embedding_dim}")

    # 初始化聚合器
    aggregator = SentenceVectorAggregator(aggregation_method="weighted_mean")

    # 【方法1】平均聚合
    print(f"\n【方法1】平均聚合：")
    vec_mean = aggregator.aggregate_multi_sentences(sentence_vectors, method="mean")
    print(f"  向量形状: {vec_mean.shape}")
    print(f"  范数: {np.linalg.norm(vec_mean):.4f}")

    # 【方法2】加权聚合
    print(f"\n【方法2】加权聚合（注意力权重）：")
    vec_weighted = aggregator.aggregate_multi_sentences(
        sentence_vectors, method="weighted"
    )
    print(f"  向量形状: {vec_weighted.shape}")
    print(f"  范数: {np.linalg.norm(vec_weighted):.4f}")
    print(f"  与平均向量相似度: {np.dot(vec_mean, vec_weighted):.4f}")

    # 【方法3】最大聚合
    print(f"\n【方法3】最大池化聚合：")
    vec_max = aggregator.aggregate_multi_sentences(sentence_vectors, method="max")
    print(f"  向量形状: {vec_max.shape}")
    print(f"  范数: {np.linalg.norm(vec_max):.4f}")


def main():
    """主演示函数"""
    print("\n" + "=" * 100)
    print("FPGA需求-代码不一致检测系统:")
    print("增强的语义提取功能演示")
    print("=" * 100)

    try:
        # 演示1：长文本句向量平均法
        demo_long_text_semantic_vector()

        # 演示2：注意力机制
        demo_attention_mechanism()

        # 演示3：完整的语义要素提取
        demo_complete_semantic_elements()

        # 演示4：句向量聚合器
        demo_sentence_vector_aggregator()

        print("\n" + "=" * 100)
        print("✓ 所有演示完成！")
        print("=" * 100 + "\n")

    except Exception as e:
        print(f"\n✗ 执行过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

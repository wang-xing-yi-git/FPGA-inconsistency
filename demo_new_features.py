#!/usr/bin/env python3
"""
快速演示脚本 - 不需要BERT模型就能演示新功能
"""

import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"


def test_attention():
    """测试注意力机制"""
    print("\n" + "=" * 80)
    print("[演示1] 注意力机制类")
    print("=" * 80)

    try:
        from src.semantic_extraction import AttentionMechanism
        import numpy as np

        attention = AttentionMechanism(attention_type="scaled_dot_product")
        print("✓ AttentionMechanism 类已成功导入")

        # 创建模拟数据
        token_embeddings = np.random.randn(10, 768)
        weights = attention.compute_attention_weights(token_embeddings)

        print(f"✓ 注意力权重计算成功")
        print(f"  输出形状: {weights.shape}")
        print(f"  权重范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        print(f"  权重和: {np.sum(weights):.6f}")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_aggregator():
    """测试句向量聚合器"""
    print("\n" + "=" * 80)
    print("[演示2] 句向量聚合器类")
    print("=" * 80)

    try:
        from src.semantic_extraction import SentenceVectorAggregator
        import numpy as np

        aggregator = SentenceVectorAggregator(aggregation_method="weighted_mean")
        print("✓ SentenceVectorAggregator 类已成功导入")

        # 测试聚合
        sentence_vectors = [np.random.randn(768) for _ in range(5)]
        result = aggregator.aggregate_multi_sentences(sentence_vectors, method="mean")

        print(f"✓ 句向量聚合成功")
        print(f"  输入句数: 5")
        print(f"  输出向量形状: {result.shape}")
        print(f"  向量范数: {np.linalg.norm(result):.4f}")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False


def test_element_extractor():
    """测试语义要素提取器"""
    print("\n" + "=" * 80)
    print("[演示3] 增强的语义要素提取器")
    print("=" * 80)

    try:
        from src.semantic_extraction import EnhancedSemanticElementExtractor

        extractor = EnhancedSemanticElementExtractor(language="auto")
        print("✓ EnhancedSemanticElementExtractor 类已成功导入")

        # 测试要素提取
        text = "Design an 8-bit counter module with clock and reset signals."
        elements = extractor.extract_elements(text, req_id=1)

        print(f"✓ 要素提取成功")
        print(f"  输入文本: {text[:50]}...")
        print(f"  提取要素数: {len(elements)}")

        if elements:
            elem = elements[0]
            print(f"  示例要素:")
            print(f"    类型: {elem['type']}")
            print(f"    值: {elem['value']}")
            print(f"    置信度: {elem['confidence']:.2f}")

        # 测试参数提取
        params = extractor.extract_parameters("WIDTH=256 DEPTH=1024")
        print(f"✓ 参数提取成功")
        print(f"  提取参数: {params}")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_nlp_extractor_new_methods():
    """测试NLPSemanticExtractor的新方法"""
    print("\n" + "=" * 80)
    print("[演示4] NLPSemanticExtractor 新增方法")
    print("=" * 80)

    try:
        from src.semantic_extraction import NLPSemanticExtractor

        extractor = NLPSemanticExtractor(language="auto")
        print("✓ NLPSemanticExtractor 已成功加载")

        # 检查新方法
        new_methods = [
            "get_semantic_vector_for_long_text",
            "extract_complete_semantic_elements",
            "_split_into_sentences",
        ]

        print("\n✓ 新增方法检查:")
        for method_name in new_methods:
            if hasattr(extractor, method_name):
                print(f"  ✓ {method_name}")
            else:
                print(f"  ✗ {method_name} (缺失)")
                return False

        # 检查新属性
        new_attrs = ["attention", "aggregator", "element_extractor"]

        print("\n✓ 新增属性检查:")
        for attr_name in new_attrs:
            if hasattr(extractor, attr_name):
                print(f"  ✓ self.{attr_name}")
            else:
                print(f"  ✗ self.{attr_name} (缺失)")
                return False

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_class_features():
    """展示新类的特性"""
    print("\n" + "=" * 80)
    print("[总结] 新增功能类汇总")
    print("=" * 80)

    features = {
        "AttentionMechanism": [
            "- 支持多种注意力机制 (scaled_dot_product, additive, multiplicative)",
            "- 自动计算token级别的注意力权重",
            "- Softmax规范化保证权重和为1.0",
            "- 用于文本中关键信息聚焦",
        ],
        "SentenceVectorAggregator": [
            "- 支持多种聚合策略 (mean, weighted, max, concat_weighted)",
            "- 句向量→文档向量聚合",
            "- 与注意力机制集成",
            "- 表征文本核心含义",
        ],
        "EnhancedSemanticElementExtractor": [
            "- 完整FPGA领域本体库 (中英文双语)",
            "- 多类型要素识别 (component, io, timing, 等)",
            "- 参数提取功能",
            "- 上下文和置信度评分",
        ],
        "NLPSemanticExtractor (增强)": [
            "- get_semantic_vector_for_long_text() - 长文本语义向量",
            "- extract_complete_semantic_elements() - 完整要素提取",
            "- 集成注意力机制和聚合器",
            "- 要素类型、值、位置、参数、需求ID一体化提取",
        ],
    }

    for class_name, features_list in features.items():
        print(f"\n【{class_name}】")
        for feature in features_list:
            print(f"  {feature}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("FPGA需求-代码不一致检测系统")
    print("增强的语义提取功能 - 快速演示")
    print("=" * 80)

    tests = [
        ("注意力机制", test_attention),
        ("句向量聚合器", test_aggregator),
        ("语义要素提取器", test_element_extractor),
        ("NLP提取器新方法", test_nlp_extractor_new_methods),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ 测试异常: {e}")
            results.append((test_name, False))

    # 显示功能总结
    show_class_features()

    # 汇总报告
    print("\n" + "=" * 80)
    print("[验证结果汇总]")
    print("=" * 80)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")

    print(f"\n总体: {passed_count}/{total_count} 通过")

    if passed_count == total_count:
        print("\n✓ 所有增强功能验证通过!")
        print("\n【后续步骤】")
        print("  1. 查看文档: SEMANTIC_ENHANCEMENT_GUIDE.md")
        print("  2. 运行演示: python demo_enhanced_semantic.py")
        print("  3. 集成使用: 在其他模块中导入使用新类")
        return 0
    else:
        print("\n✗ 部分功能验证失败，请查看错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

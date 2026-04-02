"""
中文自然语言处理演示脚本
展示系统对中文FPGA需求文档的处理能力
"""

from src.semantic_extraction import NLPSemanticExtractor, CodeSemanticExtractor


def demo_chinese_nlp():
    """演示中文NLP处理"""
    
    print("="*70)
    print("FPGA文实不一致检测 - 中文处理演示")
    print("="*70)
    
    # 初始化提取器（使用自动语言检测）
    extractor = NLPSemanticExtractor(model_name="bert-base-uncased", language="auto")
    
    # 中文需求文本示例
    chinese_examples = [
        "实现一个具有上升沿触发的异步复位计数器，计数宽度为8bit，计数范围0-255",
        "设计一个多路选择器，具有4个输入信号和2个控制信号，根据控制信号输出对应的输入",
        "实现一个频率为100MHz，分频比为10的时钟分频器，输出频率为10MHz",
        "设计一个移位寄存器，支持左移和右移，具有并行加载功能"
    ]
    
    for idx, text in enumerate(chinese_examples, 1):
        print(f"\n{'='*70}")
        print(f"示例 {idx}: {text}")
        print(f"{'='*70}")
        
        # 提取语义要素
        semantic_elements = extractor.extract_semantic_elements(text)
        
        # 显示检测到的语言
        print(f"\n【语言检测】")
        print(f"  检测语言: {semantic_elements['language']}")
        
        # 显示分词结果
        print(f"\n【分词结果】({len(semantic_elements['keywords'])} 个词)")
        print(f"  {', '.join(semantic_elements['keywords'][:15])}")
        if len(semantic_elements['keywords']) > 15:
            print(f"  ... 共 {len(semantic_elements['keywords'])} 个词")
        
        # 显示FPGA术语
        if semantic_elements['fpga_terms']:
            print(f"\n【识别的FPGA术语】({len(semantic_elements['fpga_terms'])} 个)")
            for term_info in semantic_elements['fpga_terms']:
                print(f"  - {term_info['term']}: {term_info['type']}")
        
        # 显示语法依赖分析结果
        deps = semantic_elements['syntax_dependencies']
        
        print(f"\n【语法依赖分析】")
        print(f"  主语数量: {len(deps['subjects'])}")
        if deps['subjects']:
            for subj in deps['subjects'][:2]:
                print(f"    - {subj['word']}")
        
        print(f"  谓语数量: {len(deps['predicates'])}")
        if deps['predicates']:
            for pred in deps['predicates'][:2]:
                print(f"    - {pred['word']}")
        
        print(f"  修饰词数量: {len(deps['modifiers'])}")
        if deps['modifiers']:
            for mod in deps['modifiers'][:2]:
                print(f"    - {mod['word']}")
        
        # 显示语义向量信息
        print(f"\n【语义向量】")
        vector = extractor.get_semantic_vector(text)
        if vector is not None:
            import numpy as np
            print(f"  维度: {vector.shape[0]}")
            print(f"  范数: {np.linalg.norm(vector):.4f}")


def demo_chinese_english_comparison():
    """展示中英文处理的对比"""
    
    print("\n" + "="*70)
    print("中英文处理对比")
    print("="*70)
    
    extractor = NLPSemanticExtractor(language="auto")
    
    # 中英文对应的相同需求
    comparison_pairs = [
        ("中文", "实现一个8位计数器，具有异步复位功能"),
        ("English", "Implement an 8-bit counter with asynchronous reset"),
    ]
    
    for lang, text in comparison_pairs:
        print(f"\n【{lang}】")
        print(f"文本: {text}")
        
        semantic_elements = extractor.extract_semantic_elements(text)
        
        print(f"  语言: {semantic_elements['language']}")
        print(f"  关键词数: {len(semantic_elements['keywords'])}")
        print(f"  FPGA术语数: {len(semantic_elements['fpga_terms'])}")
        
        if semantic_elements['fpga_terms']:
            terms = [t['term'] for t in semantic_elements['fpga_terms']]
            print(f"  术语: {', '.join(terms)}")


def demo_chinese_with_chinese_bert():
    """演示使用中文BERT模型的效果（需要下载模型）"""
    
    print("\n" + "="*70)
    print("中文BERT模型演示")
    print("="*70)
    print("\n说明: 若要获得最佳效果，请使用中文BERT模型")
    print("推荐模型:")
    print("  1. bert-base-chinese - 原始中文BERT")
    print("  2. hfl/chinese-roberta-wwm-ext - 中文RoBERTa")
    print("  3. hfl/chinese-electra-base - 中文ELECTRA")
    print("\n使用方法:")
    print("  extractor = NLPSemanticExtractor(model_name='bert-base-chinese')")
    print("\n下载方式:")
    print("  pip install transformers")
    print("  (模型会在首次使用时自动下载)")


def demo_language_detection():
    """演示语言自动检测功能"""
    
    print("\n" + "="*70)
    print("语言自动检测演示")
    print("="*70)
    
    from src.semantic_extraction import _detect_language
    
    test_texts = [
        ("纯中文文本", "这是一个关于FPGA设计的中文文档"),
        ("纯英文文本", "This is an English document about FPGA design"),
        ("混合文本", "这是一个FPGA的counter设计文档，包含Clock和Reset信号"),
        ("中文占少数", "Design a 8-bit counter with reset signal, 包含FPGA特定功能"),
    ]
    
    for name, text in test_texts:
        lang = _detect_language(text)
        print(f"\n{name}:")
        print(f"  文本: {text}")
        print(f"  检测结果: {lang}")


if __name__ == '__main__':
    import numpy as np
    
    print("\n")
    print("█" * 70)
    print("█ " + " " * 66 + " █")
    print("█ " + "中文自然语言处理演示".center(66) + " █")
    print("█ " + " " * 66 + " █")
    print("█" * 70)
    
    # 运行演示
    demo_chinese_nlp()
    demo_chinese_english_comparison()
    demo_language_detection()
    demo_chinese_with_chinese_bert()
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    print("\n关键特性:")
    print("  ✓ 自动语言检测（中文/英文）")
    print("  ✓ 中文分词（使用jieba）")
    print("  ✓ FPGA领域中文术语识别")
    print("  ✓ 兼容中英文BERT模型")
    print("  ✓ 语法依赖分析（支持中英文spacy模型）")
    print("\n下一步:")
    print("  1. 运行主程序: python main.py")
    print("  2. 处理中文数据集: python main.py --input data/raw/dataset_zh.json")
    print()

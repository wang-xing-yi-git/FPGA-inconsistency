"""
演示语法依赖分析在语义提取中的应用
"""

import json
from src.semantic_extraction import NLPSemanticExtractor


def demo_syntax_dependency_analysis():
    """演示语法依赖分析功能"""
    
    print("="*70)
    print("FPGA文实不一致检测 - 语法依赖分析演示")
    print("="*70)
    
    # 初始化提取器
    extractor = NLPSemanticExtractor()
    
    # 示例需求文本
    examples = [
        "实现一个具有上升沿触发的异步复位计数器，计数宽度为8bit，计数范围0-255。",
        "The counter must have a clock input and a reset signal that clears the counter to zero.",
        "Design a synchronous flip-flop with set and reset control signals.",
        "Implement a multiplexer with 4 inputs and 2-bit select signal to output selected input."
    ]
    
    for idx, text in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"示例 {idx}: {text[:50]}...")
        print(f"{'='*70}")
        
        # 提取语义要素（包含依赖分析）
        semantic_elements = extractor.extract_semantic_elements(text)
        
        # 显示关键词
        print(f"\n【关键词】({len(semantic_elements['keywords'])} 个)")
        print(f"  {', '.join(semantic_elements['keywords'][:10])}")
        if len(semantic_elements['keywords']) > 10:
            print(f"  ... 共 {len(semantic_elements['keywords'])} 个")
        
        # 显示FPGA术语
        if semantic_elements['fpga_terms']:
            print(f"\n【FPGA领域术语】({len(semantic_elements['fpga_terms'])} 个)")
            for term_info in semantic_elements['fpga_terms']:
                print(f"  - {term_info['term']}: {term_info['type']}")
        
        # 显示语法依赖分析结果
        deps = semantic_elements['syntax_dependencies']
        
        print(f"\n【POS标签】({len(deps['pos_tags'])} 个)")
        if deps['pos_tags']:
            for item in deps['pos_tags'][:5]:
                print(f"  - {item['word']}: {item.get('pos', item.get('tag', 'N/A'))}")
            if len(deps['pos_tags']) > 5:
                print(f"  ... 共 {len(deps['pos_tags'])} 个")
        
        print(f"\n【语法依赖成分】")
        print(f"  主语数量: {len(deps['subjects'])}")
        if deps['subjects']:
            for subj in deps['subjects'][:3]:
                print(f"    - {subj['word']} → {subj['head']}")
        
        print(f"  谓语数量: {len(deps['predicates'])}")
        if deps['predicates']:
            for pred in deps['predicates'][:3]:
                print(f"    - {pred['word']} ({pred['lemma']})")
        
        print(f"  宾语数量: {len(deps['objects'])}")
        if deps['objects']:
            for obj in deps['objects'][:3]:
                print(f"    - {obj['word']} ← {obj['head']}")
        
        print(f"  修饰词数量: {len(deps['modifiers'])}")
        if deps['modifiers']:
            for mod in deps['modifiers'][:3]:
                print(f"    - {mod['word']} [{mod['type']}] → {mod['head']}")
        
        print(f"  依赖对数量: {len(deps['dependency_pairs'])}")
        if deps['dependency_pairs']:
            print(f"  依赖关系示例:")
            for pair in deps['dependency_pairs'][:3]:
                print(f"    - {pair['parent']} --[{pair['relation']}]--> {pair['child']}")
        
        # 显示语义向量信息
        print(f"\n【语义向量】")
        vector = extractor.get_semantic_vector(text)
        if vector is not None:
            print(f"  维度: {vector.shape[0]}")
            print(f"  范数: {np.linalg.norm(vector):.4f}")
            print(f"  值域: [{vector.min():.4f}, {vector.max():.4f}]")
            print(f"  (已通过语法依赖分析增强)")
        else:
            print(f"  无法生成向量")


def demo_comparison_with_without_dependency():
    """比较有无依赖分析的语义提取差异"""
    
    print("\n" + "="*70)
    print("语法依赖分析对语义向量的增强效果")
    print("="*70)
    
    import numpy as np
    
    extractor = NLPSemanticExtractor()
    
    test_text = "The reset signal asynchronously clears the counter to zero on negative edge."
    
    print(f"\n文本: {test_text}\n")
    
    # 获取增强后的向量（包含依赖分析）
    semantic_elements = extractor.extract_semantic_elements(test_text)
    vector_with_deps = extractor.get_semantic_vector(test_text)
    
    if vector_with_deps is not None:
        print(f"【包含语法依赖分析的向量】")
        print(f"  向量范数: {np.linalg.norm(vector_with_deps):.4f}")
        print(f"  向量均值: {np.mean(vector_with_deps):.6f}")
        print(f"  向量方差: {np.var(vector_with_deps):.6f}")
        print(f"  非零元素比例: {np.count_nonzero(vector_with_deps) / len(vector_with_deps) * 100:.2f}%")
        
        # 分析依赖成分对向量的影响
        deps = semantic_elements['syntax_dependencies']
        print(f"\n【依赖分析结果汇总】")
        print(f"  - 句子中包含 {len(deps['subjects'])} 个主语")
        print(f"  - 句子中包含 {len(deps['predicates'])} 个谓语")
        print(f"  - 句子中包含 {len(deps['objects'])} 个宾语")
        print(f"  - 句子中包含 {len(deps['modifiers'])} 个修饰词")
        print(f"  - 句子中包含 {len(deps['dependency_pairs'])} 个依赖关系")
        
        print(f"\n【增强机制说明】")
        print(f"  - 基础BERT向量权重: 70%")
        print(f"  - 依赖关系增强权重: 30%")
        print(f"  - 根据语法成分类型和数量分配权重")
        print(f"  - 主语、谓语、宾语、修饰词各有不同的权重分配策略")


def demo_fpga_specific_analysis():
    """展示FPGA特定词汇的依赖分析"""
    
    print("\n" + "="*70)
    print("FPGA特定词汇的语法依赖分析")
    print("="*70)
    
    extractor = NLPSemanticExtractor()
    
    fpga_examples = [
        "Rising edge triggered counter with asynchronous active-low reset",
        "Implement a parallel-to-serial converter with shift and load control",
        "Design a frequency divider that generates half-frequency clock output"
    ]
    
    for text in fpga_examples:
        print(f"\n文本: {text}")
        
        semantic_elements = extractor.extract_semantic_elements(text)
        deps = semantic_elements['syntax_dependencies']
        
        # 找出FPGA相关的谓语和修饰词
        fpga_terms = semantic_elements['fpga_terms']
        print(f"FPGA术语: {[t['term'] for t in fpga_terms]}")
        
        # 分析这些术语在依赖树中的角色
        fpga_keywords = {t['term'] for t in fpga_terms}
        relevant_deps = [
            d for d in deps['dependency_pairs']
            if d['parent'] in fpga_keywords or d['child'] in fpga_keywords
        ]
        
        if relevant_deps:
            print(f"FPGA术语的依赖关系:")
            for dep in relevant_deps[:3]:
                if dep['parent'] in fpga_keywords:
                    print(f"  - {dep['parent']} --[{dep['relation']}]--> {dep['child']}")
                else:
                    print(f"  - {dep['parent']} --[{dep['relation']}]--> {dep['child']}")


if __name__ == '__main__':
    import numpy as np
    
    print("\n")
    print("█" * 70)
    print("█ " + " " * 66 + " █")
    print("█ " + "语法依赖分析在NLP语义提取中的应用".center(66) + " █")
    print("█ " + " " * 66 + " █")
    print("█" * 70)
    
    # 运行演示
    demo_syntax_dependency_analysis()
    demo_comparison_with_without_dependency()
    demo_fpga_specific_analysis()
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    print("\n说明:")
    print("  1. 语法依赖分析使用spacy库进行深层次的句法结构分析")
    print("  2. 提取的依赖关系包括: 主语、谓语、宾语、修饰词等")
    print("  3. 这些依赖关系被用来增强BERT语义向量")
    print("  4. 在语义对齐和不一致检测中提供更丰富的上下文信息")
    print()

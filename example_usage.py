"""
快速开始示例脚本
"""

import json
import numpy as np
from src.semantic_extraction import NLPSemanticExtractor, CodeSemanticExtractor
from src.semantic_alignment import SemanticAligner
from src.inconsistency_detector import InconsistencyDetector


def example_simple_detection():
    """简单示例：检测一个需求-代码对"""
    
    print("="*60)
    print("FPGA文实不一致检测 - 简单示例")
    print("="*60)
    
    # 示例需求与代码
    requirement = """
    实现一个具有上升沿触发的异步复位计数器，
    计数宽度为8bit，计数范围0-255。
    """
    
    code = """
    module counter (
        input clk,
        input rst_n,
        output [7:0] count
    );
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 8'b0;
        else
            count <= count + 1;
    end
    endmodule
    """
    
    # 1. 语义提取
    print("\n[1] 语义提取...")
    nlp_extractor = NLPSemanticExtractor()
    code_extractor = CodeSemanticExtractor()
    
    req_elements = nlp_extractor.extract_semantic_elements(requirement)
    req_vector = nlp_extractor.get_semantic_vector(requirement)
    
    code_elements = code_extractor.extract_semantic_elements(code)
    code_vector = code_extractor.get_semantic_vector(code)
    
    print(f"  需求关键字: {req_elements['keywords'][:5]}")
    print(f"  代码模块: {code_elements['modules']}")
    print(f"  代码端口数: {code_elements['port_count']}")
    
    # 2. 语义对齐
    print("\n[2] 语义对齐...")
    aligner = SemanticAligner()
    
    alignment = aligner.align_requirements_to_code(
        req_id=1,
        req_elements=req_elements,
        req_vector=req_vector,
        code_elements=code_elements,
        code_vector=code_vector,
        code_segment=code
    )
    
    print(f"  对齐状态: {alignment.status.value}")
    print(f"  相似度: {alignment.similarity_score:.2f}")
    print(f"  置信度: {alignment.confidence:.2f}")
    
    # 3. 不一致检测
    print("\n[3] 不一致检测...")
    detector = InconsistencyDetector()
    
    inconsistencies = detector.detect_all_inconsistencies(
        req_id=1,
        req_text=requirement,
        req_elements=req_elements,
        req_vector=req_vector,
        code_text=code,
        code_elements=code_elements,
        code_vector=code_vector
    )
    
    print(f"  发现不一致: {inconsistencies['total_issues']} 个")
    print(f"  严重程度分布: {inconsistencies['severity_distribution']}")
    
    if inconsistencies['explicit_inconsistencies']:
        print("\n  显性不一致:")
        for inc in inconsistencies['explicit_inconsistencies']:
            print(f"    - {inc['description']}")
    
    if inconsistencies['implicit_inconsistencies']:
        print("\n  隐性不一致:")
        for inc in inconsistencies['implicit_inconsistencies']:
            print(f"    - {inc['description']}")
    
    print("\n✓ 示例完成")


def example_batch_processing():
    """批处理示例：处理多个需求-代码对"""
    
    print("\n" + "="*60)
    print("FPGA文实不一致检测 - 批处理示例")
    print("="*60)
    
    # 示例数据
    samples = [
        {
            "id": 1,
            "req": "实现一个2输入的与门",
            "code": "module and_gate (input a, input b, output y); assign y = a & b; endmodule"
        },
        {
            "id": 2,
            "req": "设计4个输入的多路选择器，选择信号为2位",
            "code": "module mux (input [3:0] in, input [1:0] sel, output out); assign out = in[sel]; endmodule"
        }
    ]
    
    print(f"\n处理 {len(samples)} 个样本...")
    
    nlp_extractor = NLPSemanticExtractor()
    code_extractor = CodeSemanticExtractor()
    aligner = SemanticAligner()
    detector = InconsistencyDetector()
    
    results = []
    
    for sample in samples:
        req_id = sample['id']
        requirement = sample['req']
        code = sample['code']
        
        # 提取语义
        req_elements = nlp_extractor.extract_semantic_elements(requirement)
        req_vector = nlp_extractor.get_semantic_vector(requirement)
        
        code_elements = code_extractor.extract_semantic_elements(code)
        code_vector = code_extractor.get_semantic_vector(code)
        
        # 对齐
        alignment = aligner.align_requirements_to_code(
            req_id=req_id,
            req_elements=req_elements,
            req_vector=req_vector,
            code_elements=code_elements,
            code_vector=code_vector,
            code_segment=code
        )
        
        # 检测不一致
        inconsistencies = detector.detect_all_inconsistencies(
            req_id=req_id,
            req_text=requirement,
            req_elements=req_elements,
            req_vector=req_vector,
            code_text=code,
            code_elements=code_elements,
            code_vector=code_vector
        )
        
        result = {
            'id': req_id,
            'requirement': requirement,
            'alignment_status': alignment.status.value,
            'similarity': alignment.similarity_score,
            'total_issues': inconsistencies['total_issues']
        }
        results.append(result)
        
        print(f"\n  样本 {req_id}:")
        print(f"    对齐: {alignment.status.value}")
        print(f"    相似度: {alignment.similarity_score:.2f}")
        print(f"    不一致: {inconsistencies['total_issues']} 个")
    
    # 统计汇总
    total_issues = sum(r['total_issues'] for r in results)
    avg_similarity = sum(r['similarity'] for r in results) / len(results)
    
    print(f"\n汇总:")
    print(f"  总样本数: {len(results)}")
    print(f"  平均相似度: {avg_similarity:.2f}")
    print(f"  总不一致数: {total_issues}")
    
    print("\n✓ 批处理完成")


if __name__ == '__main__':
    # 运行简单示例
    example_simple_detection()
    
    # 运行批处理示例
    example_batch_processing()

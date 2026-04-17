"""
测试语义对齐模块
"""

import unittest
import numpy as np
from src.semantic_alignment import (
    MappingRulesLibrary,
    SemanticAligner,
    AlignmentStatus
)


class TestMappingRulesLibrary(unittest.TestCase):
    """映射规则库测试"""
    
    def setUp(self):
        self.library = MappingRulesLibrary()
    
    def test_rules_initialization(self):
        """测试规则库初始化"""
        self.assertGreater(len(self.library.rules), 0)
        self.assertGreater(self.library.rule_count, 0)
    
    def test_find_matching_rules(self):
        """测试规则匹配"""
        req_keywords = ['clock', 'frequency']
        code_keywords = ['always', 'clk', 'posedge']
        
        matched = self.library.find_matching_rules(req_keywords, code_keywords)
        self.assertGreater(len(matched), 0)
    
    def test_verify_rule_match(self):
        """测试规则验证"""
        rule = self.library.rules[0]
        req_keywords = rule['pattern_nlp'][:2]
        code_keywords = rule['pattern_code'][:2]
        
        matched, confidence = self.library.verify_rule_match(
            rule, req_keywords, code_keywords
        )
        self.assertTrue(matched)
        self.assertGreater(confidence, 0)


class TestSemanticAligner(unittest.TestCase):
    """语义对齐器测试"""
    
    def setUp(self):
        self.aligner = SemanticAligner()
    
    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        
        similarity = self.aligner.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(similarity, 1.0, places=2)
    
    def test_cosine_similarity_orthogonal(self):
        """测试正交向量的相似度"""
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        
        similarity = self.aligner.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(similarity, 0.5, places=2)
    
    def test_align_requirements_to_code(self):
        """测试需求到代码的对齐"""
        req_elements = {
            'keywords': ['clock', 'counter'],
            'fpga_terms': [{'term': 'clock', 'type': 'clock'}]
        }
        code_elements = {
            'keywords': {'always': {'count': 1, 'type': 'behavior'}},
            'modules': ['counter']
        }
        
        req_vector = np.random.randn(768)
        code_vector = np.random.randn(768)
        
        result = self.aligner.align_requirements_to_code(
            req_id=1,
            req_elements=req_elements,
            req_vector=req_vector,
            code_elements=code_elements,
            code_vector=code_vector,
            code_segment="module counter..."
        )
        
        self.assertEqual(result.req_id, 1)
        self.assertIsNotNone(result.status)
        self.assertGreaterEqual(result.mapping_confidence, 0)
        self.assertLessEqual(result.mapping_confidence, 1)


if __name__ == '__main__':
    unittest.main()

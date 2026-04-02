"""
【修改】增强的语义对齐模块
支持语法库、语义映射规则库、相似度计算与规则匹配结合
实现自然语言与代码的双向语义精准对齐
"""

import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from src.semantic_extraction import extract_bidirectional_semantics

class AlignmentStatus(Enum):
    """对齐状态"""
    ALIGNED = "aligned"
    SUSPICIOUS = "suspicious"
    UNALIGNED = "unaligned"
    INDIRECT = "indirect"


@dataclass
class AlignmentResult:
    """对齐结果"""
    req_id: int
    code_segment: str
    similarity_score: float
    status: AlignmentStatus
    confidence: float
    matched_rule: Optional[str] = None
    reason: str = ""
    mapping_confidence: float = 0.0  # 【修改】语义映射置信度


class NLPSyntaxLibrary:
    """
    【修改】自然语言语法库
    用于解析需求文本的语法结构，提取关键语义成分
    """
    
    def __init__(self):
        """初始化自然语言语法库"""
        # 【修改】中文FPGA领域语义模式库
        self.chinese_patterns = {
            'timing_constraints': {
                'keywords': ['时钟周期', '延迟', '频率', '时序', '脉冲宽度', '建立时间', '保持时间'],
                'semantic_type': 'timing'
            },
            'io_specifications': {
                'keywords': ['输入', '输出', '端口', '信号', '数据', '地址', '控制'],
                'semantic_type': 'io'
            },
            'memory_operations': {
                'keywords': ['存储', '读取', '写入', '内存', 'RAM', '地址', '数据深度', '位宽'],
                'semantic_type': 'memory'
            },
            'control_logic': {
                'keywords': ['控制', '条件', '选择', '使能', '复位', '同步', '异步'],
                'semantic_type': 'control'
            },
            'datapath': {
                'keywords': ['数据通路', '处理', '运算', '计数', '计数器', '累加'],
                'semantic_type': 'datapath'
            },
            'synchronization': {
                'keywords': ['同步', '异步', '时钟域', '握手', '信号同步'],
                'semantic_type': 'synchronization'
            }
        }
        
        # 【修改】英文FPGA领域语义模式库
        self.english_patterns = {
            'timing_constraints': {
                'keywords': ['clock', 'cycle', 'delay', 'frequency', 'timing', 'pulse', 'setup', 'hold'],
                'semantic_type': 'timing'
            },
            'io_specifications': {
                'keywords': ['input', 'output', 'port', 'signal', 'data', 'address', 'control'],
                'semantic_type': 'io'
            },
            'memory_operations': {
                'keywords': ['memory', 'read', 'write', 'ram', 'address', 'depth', 'width'],
                'semantic_type': 'memory'
            },
            'control_logic': {
                'keywords': ['control', 'condition', 'select', 'enable', 'reset', 'sync', 'async'],
                'semantic_type': 'control'
            },
            'datapath': {
                'keywords': ['datapath', 'process', 'compute', 'count', 'counter', 'accumulate'],
                'semantic_type': 'datapath'
            }
        }
    
    def extract_semantic_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        【修改】从文本提取语义模式
        
        Args:
            text: 输入文本
            
        Returns:
            语义模式及匹配的关键字
        """
        text_lower = text.lower()
        patterns_found = {}
        
        # 判断语言
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        pattern_lib = self.chinese_patterns if is_chinese else self.english_patterns
        
        for pattern_name, pattern_info in pattern_lib.items():
            matched_keywords = []
            for keyword in pattern_info['keywords']:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                patterns_found[pattern_name] = matched_keywords
        
        return patterns_found


class CodeSyntaxLibrary:
    """
    【修改】代码语法库
    用于解析FPGA代码的语法结构，提取关键代码构造
    """
    
    def __init__(self):
        """初始化代码语法库"""
        # 【修改】Verilog语法构造库
        self.verilog_constructs = {
            'timing_logic': {
                'patterns': [
                    r'always\s*@\s*\(\s*(?:posedge|negedge)\s+\w+',
                    r'#\d+',
                    r'@\(.*?clk.*?\)'
                ],
                'semantic_type': 'timing'
            },
            'io_declarations': {
                'patterns': [
                    r'(?:input|output|inout)\s+(?:reg|wire)?\s*\[?\d*:\d*\]?\s+\w+',
                    r'parameter\s+\w+\s*=\s*\d+'
                ],
                'semantic_type': 'io'
            },
            'memory_structures': {
                'patterns': [
                    r'reg\s+\[?\d*:\d*\]?\s+\w+\s*\[[\d:]+\]',
                    r'reg\s+(?:.*?)\s+\w+_mem',
                    r'RAM|memory|mem'
                ],
                'semantic_type': 'memory'
            },
            'control_structures': {
                'patterns': [
                    r'if\s*\(',
                    r'case\s*\(',
                    r'always\s*@\s*\(\*\)',
                    r'rst.*?(?:==|!=)',
                    r'enable|en'
                ],
                'semantic_type': 'control'
            },
            'datapath_logic': {
                'patterns': [
                    r'<=',
                    r'\+=',
                    r'count\s*<=',
                    r'&|\\||\^|-|\+|\*'
                ],
                'semantic_type': 'datapath'
            },
            'synchronization': {
                'patterns': [
                    r'always\s*@\s*\(\s*posedge\s+clk',
                    r'synchronized|sync',
                    r'metastable'
                ],
                'semantic_type': 'synchronization'
            }
        }
    
    def extract_code_constructs(self, code: str) -> Dict[str, List[str]]:
        """
        【修改】从代码提取语法构造
        
        Args:
            code: Verilog代码
            
        Returns:
            检测到的语法构造
        """
        constructs_found = {}
        code_lower = code.lower()
        
        for construct_name, construct_info in self.verilog_constructs.items():
            matched_patterns = []
            for pattern in construct_info['patterns']:
                matches = re.findall(pattern, code_lower, re.IGNORECASE)
                if matches:
                    matched_patterns.extend(matches[:3])  # 最多保留3个匹配
            
            if matched_patterns:
                constructs_found[construct_name] = list(set(matched_patterns))
        
        return constructs_found


class SemanticMappingRulesLibrary:
    """
    【修改】语义映射规则库
    定义自然语言与代码之间的语义对应关系
    """
    
    def __init__(self):
        """【修改】初始化增强的语义映射规则库"""
        self.mapping_rules = self._initialize_mapping_rules()
        self.rule_count = len(self.mapping_rules)
    
    def _initialize_mapping_rules(self) -> List[Dict]:
        """
        【修改】初始化FPGA领域的双向语义映射规则
        
        Returns:
            映射规则列表
        """
        rules = [
            # 【修改】时钟与时序规则
            {
                'id': 'rule_timing_clock',
                'nlp_keywords': ['时钟', '时钟周期', '频率', 'mhz', 'clock', 'frequency'],
                'code_patterns': ['always.*clk', 'posedge', 'negedge', 'clk', '@(.*clk'],
                'semantic_type': 'timing',
                'mapping_score': 0.95,
                'constraint_type': 'functional'
            },
            # 【修改】复位与初始化规则
            {
                'id': 'rule_control_reset',
                'nlp_keywords': ['复位', '重置', '初始化', '清零', 'reset', 'rst', 'clear'],
                'code_patterns': ['rst', 'reset', 'negedge.*rst', 'if.*rst'],
                'semantic_type': 'control',
                'mapping_score': 0.92,
                'constraint_type': 'functional'
            },
            # 【修改】数据位宽规则
            {
                'id': 'rule_datapath_width',
                'nlp_keywords': ['位宽', '比特', '数据宽度', 'bit', 'width', 'bits'],
                'code_patterns': [r'\[\d+:\d+\]', 'wire.*\[', 'reg.*\[', '[7:0]'],
                'semantic_type': 'io',
                'mapping_score': 0.90,
                'constraint_type': 'structural'
            },
            # 【修改】存储/RAM规则
            {
                'id': 'rule_memory_ram',
                'nlp_keywords': ['存储', '内存', '双端口', 'RAM', '深度', 'memory', 'port'],
                'code_patterns': ['reg.*\\[.*\\]', 'ram', 'memory', 'dpram', 'port'],
                'semantic_type': 'memory',
                'mapping_score': 0.88,
                'constraint_type': 'structural'
            },
            # 【修改】读写操作规则
            {
                'id': 'rule_memory_readwrite',
                'nlp_keywords': ['读取', '写入', '读操作', '写操作', 'read', 'write', 'wr_rd'],
                'code_patterns': ['<=', '=', 'wr_rd_n', 'we', 'write_enable'],
                'semantic_type': 'memory',
                'mapping_score': 0.85,
                'constraint_type': 'functional'
            },
            # 【修改】端口连接规则
            {
                'id': 'rule_io_ports',
                'nlp_keywords': ['端口', '输入', '输出', '信号', 'port', 'input', 'output', 'io'],
                'code_patterns': ['input', 'output', 'inout', 'port'],
                'semantic_type': 'io',
                'mapping_score': 0.87,
                'constraint_type': 'structural'
            },
            # 【修改】条件控制规则
            {
                'id': 'rule_logic_condition',
                'nlp_keywords': ['条件', '选择', '控制', '如果', 'if', 'condition', 'select'],
                'code_patterns': ['if', 'else', 'case', '?:', 'ternary'],
                'semantic_type': 'control',
                'mapping_score': 0.86,
                'constraint_type': 'functional'
            },
            # 【修改】同步设计规则
            {
                'id': 'rule_sync_design',
                'nlp_keywords': ['同步', '时钟', '同步设计', 'synchronous', 'sync'],
                'code_patterns': ['always.*@.*posedge', 'clocked', 'synchronized'],
                'semantic_type': 'synchronization',
                'mapping_score': 0.84,
                'constraint_type': 'architectural'
            },
            # 【修改】异步设计规则
            {
                'id': 'rule_async_design',
                'nlp_keywords': ['异步', '独立', '异步设计', 'asynchronous', 'async'],
                'code_patterns': ['always.*@.*rst', 'async', 'independent'],
                'semantic_type': 'control',
                'mapping_score': 0.80,
                'constraint_type': 'architectural'
            },
            # 【修改】模块参数化规则
            {
                'id': 'rule_param_module',
                'nlp_keywords': ['参数', '可配置', '模块', 'parameter', 'configurable'],
                'code_patterns': ['parameter', '#(', 'DEPTH', 'WIDTH'],
                'semantic_type': 'structural',
                'mapping_score': 0.82,
                'constraint_type': 'structural'
            },
            # 【修改】延迟与时序规则
            {
                'id': 'rule_timing_delay',
                'nlp_keywords': ['延迟', '时序', '脉冲', 'delay', 'timing', 'latency'],
                'code_patterns': ['#', 'timescale', 'delay', '@'],
                'semantic_type': 'timing',
                'mapping_score': 0.78,
                'constraint_type': 'temporal'
            },
            # 【修改】模块实例化规则
            {
                'id': 'rule_struct_instance',
                'nlp_keywords': ['模块', '实例', '子模块', 'module', 'instance'],
                'code_patterns': ['module', '\\w+\\s+\\w+\\s*\\(', 'instantiat'],
                'semantic_type': 'component',
                'mapping_score': 0.85,
                'constraint_type': 'structural'
            }
        ]
        return rules
    
    def find_semantic_mappings(self, nlp_keywords: List[str], code_constructs: List[str]) -> List[Dict]:
        """
        【修改】查找语义映射规则
        
        Args:
            nlp_keywords: NLP提取的关键字
            code_constructs: 代码提取的构造
            
        Returns:
            匹配的语义映射规则列表
        """
        matched_mappings = []
        
        for rule in self.mapping_rules:
            nlp_match = any(kw in nlp_keywords for kw in rule['nlp_keywords'])
            code_match = any(pattern in code_constructs for pattern in rule['code_patterns'])
            
            if nlp_match and code_match:
                matched_mappings.append(rule)
        
        return matched_mappings
    
    def calculate_mapping_confidence(self, rule: Dict, nlp_keywords: List[str], 
                                    code_constructs: List[str]) -> float:
        """
        【修改】计算语义映射置信度
        
        Args:
            rule: 映射规则
            nlp_keywords: NLP关键字
            code_constructs: 代码构造
            
        Returns:
            映射置信度 [0, 1]
        """
        nlp_match_count = sum(1 for kw in rule['nlp_keywords'] if kw in nlp_keywords)
        code_match_count = sum(1 for pattern in rule['code_patterns'] if pattern in code_constructs)
        
        if nlp_match_count == 0 or code_match_count == 0:
            return 0.0
        
        # 【修改】组合匹配度和规则置信度
        match_ratio = min((nlp_match_count + code_match_count) / 
                         (len(rule['nlp_keywords']) + len(rule['code_patterns'])), 1.0)
        
        confidence = rule.get('mapping_score', 0.8) * match_ratio
        return min(confidence, 1.0)


class MappingRulesLibrary:
    """【修改】保留原有接口的规则库（兼容性）"""
    
    def __init__(self):
        """初始化规则库"""
        self.rules = self._initialize_rules()
        self.rule_count = len(self.rules)
    
    def _initialize_rules(self) -> List[Dict]:
        """
        初始化FPGA领域的映射规则
        
        Returns:
            规则列表
        """
        rules = [
            # 时钟相关规则
            {
                'id': 'rule_clock_1',
                'pattern_nlp': ['clock', 'clk', 'frequency', 'mhz'],
                'pattern_code': ['always.*clk', 'posedge', 'negedge'],
                'type': 'timing',
                'confidence': 0.95
            },
            # 复位相关规则
            {
                'id': 'rule_reset_1',
                'pattern_nlp': ['reset', 'rst', 'initialization', 'clear'],
                'pattern_code': ['rst', 'reset_n', 'negedge.*rst'],
                'type': 'control',
                'confidence': 0.9
            },
            # 计数器规则
            {
                'id': 'rule_counter_1',
                'pattern_nlp': ['counter', 'count', 'increment', 'decrement'],
                'pattern_code': ['count.*=.*count', 'counter.*<='],
                'type': 'logic',
                'confidence': 0.85
            },
            # 宽度/位宽规则
            {
                'id': 'rule_width_1',
                'pattern_nlp': ['width', 'bit', 'bits', 'range'],
                'pattern_code': [r'\[\d+:\d+\]', 'wire.*\[', 'reg.*\['],
                'type': 'dimension',
                'confidence': 0.8
            },
            # 存储元件规则
            {
                'id': 'rule_register_1',
                'pattern_nlp': ['register', 'storage', 'memory', 'buffer'],
                'pattern_code': ['reg', 'always.*<=', 'ff'],
                'type': 'storage',
                'confidence': 0.85
            },
            # 信号规则
            {
                'id': 'rule_signal_1',
                'pattern_nlp': ['signal', 'wire', 'input', 'output'],
                'pattern_code': ['wire', 'input', 'output', 'inout'],
                'type': 'io',
                'confidence': 0.9
            },
            # 异步设计规则
            {
                'id': 'rule_async_1',
                'pattern_nlp': ['asynchronous', 'async', 'independent'],
                'pattern_code': ['always.*@.*rst', 'async'],
                'type': 'control',
                'confidence': 0.8
            },
            # 延迟规则
            {
                'id': 'rule_delay_1',
                'pattern_nlp': ['delay', 'latency', 'timing'],
                'pattern_code': ['#', 'timescale', 'delay'],
                'type': 'timing',
                'confidence': 0.75
            },
            # 模块实例化规则
            {
                'id': 'rule_module_1',
                'pattern_nlp': ['module', 'component', 'submodule'],
                'pattern_code': ['module', 'instantiate'],
                'type': 'component',
                'confidence': 0.9
            },
            # 条件控制规则
            {
                'id': 'rule_condition_1',
                'pattern_nlp': ['if', 'condition', 'conditional'],
                'pattern_code': ['if', 'else', 'case', '?:'],
                'type': 'control',
                'confidence': 0.85
            }
        ]
        return rules
    
    def find_matching_rules(self, req_keywords: List[str], code_keywords: List[str]) -> List[Dict]:
        """
        查找匹配的规则
        
        Args:
            req_keywords: 需求文档关键字
            code_keywords: 代码关键字
            
        Returns:
            匹配的规则列表
        """
        matched_rules = []
        
        for rule in self.rules:
            req_match = any(kw in req_keywords for kw in rule['pattern_nlp'])
            code_match = any(kw in code_keywords for kw in rule['pattern_code'])
            
            if req_match and code_match:
                matched_rules.append(rule)
        
        return matched_rules
    
    def verify_rule_match(self, rule: Dict, req_keywords: List[str], 
                         code_keywords: List[str]) -> Tuple[bool, float]:
        """
        验证规则匹配
        
        Args:
            rule: 规则
            req_keywords: 需求关键字
            code_keywords: 代码关键字
            
        Returns:
            (是否匹配, 置信度) 元组
        """
        req_match_count = sum(1 for kw in rule['pattern_nlp'] if kw in req_keywords)
        code_match_count = sum(1 for kw in rule['pattern_code'] if kw in code_keywords)
        
        if req_match_count > 0 and code_match_count > 0:
            # 计算匹配度
            match_ratio = (req_match_count * code_match_count) / (
                len(rule['pattern_nlp']) * len(rule['pattern_code'])
            )
            confidence = rule['confidence'] * match_ratio
            return True, confidence
        
        return False, 0.0



class SemanticAligner:
    """【修改】增强的语义对齐器 - 支持语法库和语义映射规则"""
    
    def __init__(self, rules_library: Optional[MappingRulesLibrary] = None):
        """
        【修改】初始化增强的语义对齐器
        
        Args:
            rules_library: 映射规则库
        """
        self.rules_library = rules_library or MappingRulesLibrary()
        
        # 【修改】初始化语法库
        self.nlp_syntax = NLPSyntaxLibrary()
        self.code_syntax = CodeSyntaxLibrary()
        
        # 【修改】初始化语义映射规则库
        self.semantic_mapping_lib = SemanticMappingRulesLibrary()
        
        self.similarity_threshold = 0.7
        self.high_similarity_threshold = 0.8
        self.low_similarity_threshold = 0.5
    
    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度值 [0, 1]
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        # 确保值在 [0, 1] 范围内
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def align_requirements_to_code(
        self,
        req_id: int,
        req_elements: Dict,
        req_vector: np.ndarray,
        code_elements: Dict,
        code_vector: np.ndarray,
        code_segment: str,
        req_text: str = "",
        code_text: str = ""
    ) -> AlignmentResult:
        """
        【修改】对齐需求与代码 - 使用语法库和语义映射
        
        Args:
            req_id: 需求ID
            req_elements: 需求语义要素
            req_vector: 需求语义向量
            code_elements: 代码语义要素
            code_vector: 代码语义向量
            code_segment: 代码片段
            req_text: 完整需求文本（用于语法分析）
            code_text: 完整代码文本（用于构造分析）
            
        Returns:
            对齐结果
        """
        # 【修改】第1步：计算向量相似度
        similarity = self.compute_cosine_similarity(req_vector, code_vector)
        
        # 【修改】第2步：提取NLP关键字和代码构造
        req_keywords = req_elements.get('keywords', [])
        code_keywords = list(code_elements.get('keywords', {}).keys())
        
        # 【修改】第3步：使用语法库进行深层分析
        nlp_patterns = self.nlp_syntax.extract_semantic_patterns(req_text) if req_text else {}
        code_constructs = self.code_syntax.extract_code_constructs(code_text) if code_text else {}
        
        # 【修改】第4步：使用语义映射规则库进行双向对齐
        semantic_mappings = self.semantic_mapping_lib.find_semantic_mappings(
            req_keywords, code_keywords
        )
        
        # 【修改】计算语义映射置信度
        mapping_confidence = 0.0
        if semantic_mappings:
            mapping_scores = []
            for rule in semantic_mappings:
                score = self.semantic_mapping_lib.calculate_mapping_confidence(
                    rule, req_keywords, code_keywords
                )
                mapping_scores.append(score)
            mapping_confidence = np.mean(mapping_scores) if mapping_scores else 0.0
        
        # 【修改】第5步：原有规则匹配（保留兼容性）
        matched_rules = self.rules_library.find_matching_rules(req_keywords, code_keywords)
        
        # 【修改】第6步：综合决定对齐状态
        status, confidence, reason = self._determine_alignment_status(
            similarity, matched_rules, req_keywords, code_keywords,
            semantic_mappings, mapping_confidence, nlp_patterns, code_constructs
        )
        
        aligned_rule = matched_rules[0]['id'] if matched_rules else None
        
        result = AlignmentResult(
            req_id=req_id,
            code_segment=code_segment[:100],
            similarity_score=similarity,
            status=status,
            confidence=confidence,
            matched_rule=aligned_rule,
            reason=reason,
            mapping_confidence=mapping_confidence  # 【修改】加入语义映射置信度
        )
        
        return result
    
    def _determine_alignment_status(
        self,
        similarity: float,
        matched_rules: List[Dict],
        req_keywords: List[str],
        code_keywords: List[str],
        semantic_mappings: List[Dict] = None,
        mapping_confidence: float = 0.0,
        nlp_patterns: Dict = None,
        code_constructs: Dict = None
    ) -> Tuple[AlignmentStatus, float, str]:
        """
        【修改】确定对齐状态 - 综合多种因素
        
        Args:
            similarity: 向量相似度
            matched_rules: 原有规则匹配结果
            req_keywords: 需求关键字
            code_keywords: 代码关键字
            semantic_mappings: 语义映射规则匹配结果
            mapping_confidence: 语义映射置信度
            nlp_patterns: NLP提取的模式
            code_constructs: 代码提取的构造
            
        Returns:
            (状态, 置信度, 原因) 元组
        """
        semantic_mappings = semantic_mappings or []
        nlp_patterns = nlp_patterns or {}
        code_constructs = code_constructs or {}
        
        # 【修改】综合相似度、规则匹配、语义映射、语法库匹配
        pattern_match_score = 0.0
        if nlp_patterns and code_constructs:
            # 计算语义模式与代码构造的匹配度
            pattern_overlaps = []
            for pattern_name in nlp_patterns:
                for construct_name in code_constructs:
                    if pattern_name.replace('_', '') == construct_name.replace('_', ''):
                        pattern_overlaps.append(1.0)
            pattern_match_score = np.mean(pattern_overlaps) if pattern_overlaps else 0.0
        
        # 【修改】综合评分
        composite_score = (
            similarity * 0.35 +
            mapping_confidence * 0.35 +
            pattern_match_score * 0.30
        )
        
        # 【修改】基于综合评分判定状态
        if composite_score >= 0.85 and semantic_mappings and matched_rules:
            # 高质量对齐：向量相似、语义映射、规则都匹配
            confidence = min(composite_score, 1.0)
            reason = f"High-quality alignment: similarity={similarity:.3f}, mapping_conf={mapping_confidence:.3f}, pattern_match={pattern_match_score:.3f}"
            return AlignmentStatus.ALIGNED, confidence, reason
        
        elif composite_score >= 0.75 and (semantic_mappings or matched_rules):
            # 良好对齐：综合评分不错且有规则支持
            confidence = min(composite_score * 0.95, 1.0)
            reason = f"Good alignment with rule support: composite_score={composite_score:.3f}"
            return AlignmentStatus.ALIGNED, confidence, reason
        
        elif composite_score >= 0.65 and semantic_mappings:
            # 可疑对齐：有语义映射但其他指标一般
            confidence = min(composite_score * 0.75, 1.0)
            reason = f"Suspicious: semantic mappings found but weak similarity (sim={similarity:.3f})"
            return AlignmentStatus.SUSPICIOUS, confidence, reason
        
        elif similarity >= self.high_similarity_threshold and matched_rules:
            max_rule_confidence = max(rule['confidence'] for rule in matched_rules)
            confidence = (similarity + max_rule_confidence) / 2
            return AlignmentStatus.ALIGNED, confidence, "High similarity with rule match"
        
        elif similarity >= self.similarity_threshold:
            if matched_rules or semantic_mappings:
                max_confidence = max(
                    [rule['confidence'] for rule in matched_rules] +
                    [mapping_confidence] if semantic_mappings else [rule['confidence'] for rule in matched_rules]
                )
                confidence = (similarity + max_confidence) / 2
                return AlignmentStatus.ALIGNED, confidence, "Threshold met with rule/mapping support"
            else:
                return AlignmentStatus.SUSPICIOUS, similarity * 0.8, "Similarity ok but no rule match"
        
        elif similarity >= self.low_similarity_threshold and (matched_rules or semantic_mappings):
            max_confidence = max(
                [rule['confidence'] for rule in matched_rules] +
                [mapping_confidence] if semantic_mappings else [rule['confidence'] for rule in matched_rules]
            )
            confidence = (similarity + max_confidence) / 2 * 0.7
            return AlignmentStatus.INDIRECT, confidence, "Indirect mapping via rules"
        
        else:
            return AlignmentStatus.UNALIGNED, similarity * 0.5, "No sufficient alignment evidence"
    
    def batch_align(
        self,
        requirements: List[Dict],
        code_elements_list: List[Dict],
        code_vectors_list: List[np.ndarray]
    ) -> List[AlignmentResult]:
        """
        批量对齐
        
        Args:
            requirements: 需求列表
            code_elements_list: 代码要素列表
            code_vectors_list: 代码向量列表
            
        Returns:
            对齐结果列表
        """
        results = []
        
        for i, req in enumerate(requirements):
            if i < len(code_elements_list) and i < len(code_vectors_list):
                result = self.align_requirements_to_code(
                    req_id=req.get('id', i),
                    req_elements=req.get('elements', {}),
                    req_vector=req.get('vector'),
                    code_elements=code_elements_list[i],
                    code_vector=code_vectors_list[i],
                    code_segment=req.get('code_segment', ''),
                    req_text=req.get('text', ''),
                    code_text=req.get('code_text', '')
                )
                results.append(result)
        
        return results



def align_semantics(
    req_data: Dict,
    code_data: Dict,
    req_text: str = "",
    code_text: str = ""
) -> Dict:
    """
    【修改】便捷函数：对齐语义 - 使用增强的对齐器
    
    Args:
        req_data: 需求语义数据
        code_data: 代码语义数据
        req_text: 完整需求文本
        code_text: 完整代码文本
        
    Returns:
        对齐结果
    """
    aligner = SemanticAligner()
    
    result = aligner.align_requirements_to_code(
        req_id=req_data.get('id', 0),
        req_elements=req_data.get('elements', {}),
        req_vector=req_data.get('vector'),
        code_elements=code_data.get('elements', {}),
        code_vector=code_data.get('vector'),
        code_segment=code_data.get('code_segment', ''),
        req_text=req_text,
        code_text=code_text
    )
    
    return {
        'alignment_result': result,
        'status': result.status.value,
        'confidence': result.confidence,
        'similarity': result.similarity_score,
        'mapping_confidence': result.mapping_confidence  # 【修改】加入映射置信度
    }


# ==========================================
# 【修改】端到端测试：语义提取 -> 语义对齐
# ==========================================

if __name__ == "__main__":
    # 1. 定义你的测试数据（双端口RAM案例）
    print("🚀 开始处理双端口RAM需求与代码...")
    eq_text = "FPGA双端口RAM模块，数据位宽固定为8比特；采用单总线时钟实现双端口RAM逻辑；端口A与总线绑定，端口B为通用业务端口；总线侧读写控制规则：在1个时钟周期内同时置位片选信号、8位地址信号、读写控制信号，即可执行读或写操作；写操作时序：写数据在寻址时立即被写入对应内存地址；读操作时序：读请求触发后，有效数据标志信号将延迟1个时钟周期脉冲，此时总线读数据端口输出对应内存数据；模块可配置参数：DEPTH为双端口RAM的存储深度，代表存储的8比特字长数据的个数。"
    code_text = "module Bus8_DPRAM #(DEPTH = 256)(input i_Bus_Rst_L,input i_Bus_Clk,input i_Bus_CS,input i_Bus_Wr_Rd_n,input [$clog2(DEPTH)-1:0] i_Bus_Addr8,input [7:0] i_Bus_Wr_Data,output [7:0] o_Bus_Rd_Data,output reg o_Bus_Rd_DV,input [7:0] i_PortB_Data,input [$clog2(DEPTH)-1:0] i_PortB_Addr8,input i_PortB_WE,output [7:0] o_PortB_Data);Dual_Port_RAM_Single_Clock #(.WIDTH(8),.DEPTH(DEPTH)) Bus_RAM_Inst(.i_Clk(i_Bus_Clk),.i_PortA_Data(i_Bus_Wr_Data),.i_PortA_Addr(i_Bus_Addr8),.i_PortA_WE(i_Bus_Wr_Rd_n & i_Bus_CS),.o_PortA_Data(o_Bus_Rd_Data),.i_PortB_Data(i_PortB_Data),.i_PortB_Addr(i_PortB_Addr8),.i_PortB_WE(i_PortB_WE),.o_PortB_Data(o_PortB_Data));always @(posedge i_Bus_Clk)begin o_Bus_Rd_DV <= i_Bus_CS & ~i_Bus_Wr_Rd_n;end endmodule"

    # 2. 第一步：运行语义提取（复用之前的代码）
    print("📊 正在进行语义提取...")
    extraction_result = extract_bidirectional_semantics(eq_text, code_text)
    
    # 3. 第二步：准备对齐数据（把提取结果整理成对齐模块需要的格式）
    # 注意：这里要小心处理数据结构，确保和对齐模块的输入一致
    req_data_for_align = {
        'id': 1,
        'elements': extraction_result['requirement']['semantic_elements'],
        'vector': extraction_result['requirement']['semantic_vector'],
        'text': eq_text  # 【修改】加入原文本用于语法库分析
    }
    
    code_data_for_align = {
        'id': 1,
        'elements': extraction_result['code']['semantic_elements'],
        'vector': extraction_result['code']['semantic_vector'],
        'code_segment': code_text,
        'code_text': code_text  # 【修改】加入代码文本用于构造分析
    }

    # 4. 第三步：运行增强的语义对齐
    print("🔗 正在进行【增强】语义对齐（使用语法库和语义映射规则库）...")
    aligner = SemanticAligner()
    
    alignment_result = aligner.align_requirements_to_code(
        req_id=req_data_for_align['id'],
        req_elements=req_data_for_align['elements'],
        req_vector=req_data_for_align['vector'],
        code_elements=code_data_for_align['elements'],
        code_vector=code_data_for_align['vector'],
        code_segment=code_data_for_align['code_segment'],
        req_text=req_data_for_align.get('text', ''),
        code_text=code_data_for_align.get('code_text', '')
    )

    # 5. 第四步：打印美观的结果
    print("\n" + "="*100)
    print("✅ 【双端口RAM】需求-代码对齐最终报告（【修改】增强版本）")
    print("="*100)
    
    # 状态映射成中文，方便阅读
    status_cn_map = {
        'aligned': '✅ 完全对齐',
        'suspicious': '⚠️ 存疑对齐',
        'unaligned': '❌ 未对齐',
        'indirect': '🔗 间接对齐'
    }
    
    print(f"1. 对齐状态: {status_cn_map.get(alignment_result.status.value, '未知')}")
    print(f"2. 向量相似度: {alignment_result.similarity_score:.4f}")
    print(f"3. 综合置信度: {alignment_result.confidence:.4f}")
    print(f"4. 【修改】语义映射置信度: {alignment_result.mapping_confidence:.4f}")
    print(f"5. 匹配规则ID: {alignment_result.matched_rule if alignment_result.matched_rule else '无'}")
    print(f"6. 判定原因: {alignment_result.reason}")
    print(f"7. 代码片段预览: {alignment_result.code_segment[:80]}...")
    
    # 额外打印匹配到的FPGA术语（如果有）
    fpga_terms = extraction_result['requirement']['semantic_elements'].get('fpga_terms', [])
    if fpga_terms:
        print(f"\n💡 需求中识别到的FPGA关键术语: {[t['term'] for t in fpga_terms]}")
    
    # 【修改】打印语法库分析结果
    print("\n【修改】语法库分析结果：")
    nlp_syntax = NLPSyntaxLibrary()
    nlp_patterns = nlp_syntax.extract_semantic_patterns(eq_text)
    if nlp_patterns:
        print(f"  - NLP语义模式: {list(nlp_patterns.keys())}")
    
    code_syntax = CodeSyntaxLibrary()
    code_constructs = code_syntax.extract_code_constructs(code_text)
    if code_constructs:
        print(f"  - 代码语法构造: {list(code_constructs.keys())}")
    
    # 【修改】打印AST信息
    ast_info = extraction_result['code']['semantic_elements'].get('ast_nodes', {})
    if ast_info:
        print(f"  - AST根节点: {ast_info.get('type')}")
        print(f"  - AST子节点统计: {ast_info.get('children_summary', {})}")
    
    code_complexity = extraction_result['code']['semantic_elements'].get('code_complexity', 'N/A')
    print(f"  - 代码复杂度: {code_complexity}")
    
    print("="*100 + "\n")

# 严格按照你的格式编写测试
eq_text = "FPGA双端口RAM模块，数据位宽固定为8比特；采用单总线时钟实现双端口RAM逻辑；端口A与总线绑定，端口B为通用业务端口；总线侧读写控制规则：在1个时钟周期内同时置位片选信号、8位地址信号、读写控制信号，即可执行读或写操作；写操作时序：写数据在寻址时立即被写入对应内存地址；读操作时序：读请求触发后，有效数据标志信号将延迟1个时钟周期脉冲，此时总线读数据端口输出对应内存数据；模块可配置参数：DEPTH为双端口RAM的存储深度，代表存储的8比特字长数据的个数。"
code_text = "module Bus8_DPRAM #(DEPTH = 256)(input i_Bus_Rst_L,input i_Bus_Clk,input i_Bus_CS,input i_Bus_Wr_Rd_n,input [$clog2(DEPTH)-1:0] i_Bus_Addr8,input [7:0] i_Bus_Wr_Data,output [7:0] o_Bus_Rd_Data,output reg o_Bus_Rd_DV,input [7:0] i_PortB_Data,input [$clog2(DEPTH)-1:0] i_PortB_Addr8,input i_PortB_WE,output [7:0] o_PortB_Data);Dual_Port_RAM_Single_Clock #(.WIDTH(8),.DEPTH(DEPTH)) Bus_RAM_Inst(.i_Clk(i_Bus_Clk),.i_PortA_Data(i_Bus_Wr_Data),.i_PortA_Addr(i_Bus_Addr8),.i_PortA_WE(i_Bus_Wr_Rd_n & i_Bus_CS),.o_PortA_Data(o_Bus_Rd_Data),.i_PortB_Data(i_PortB_Data),.i_PortB_Addr(i_PortB_Addr8),.i_PortB_WE(i_PortB_WE),.o_PortB_Data(o_PortB_Data));always @(posedge i_Bus_Clk)begin o_Bus_Rd_DV <= i_Bus_CS & ~i_Bus_Wr_Rd_n;end endmodule"

# 1. 执行双向语义提取（保留你原有的函数）
extraction_result = extract_bidirectional_semantics(eq_text, code_text)

# 2. 执行语义对齐（新增核心功能，无缝衔接）
req_data = {
    "id": 1,
    "elements": extraction_result["requirement"]["semantic_elements"],
    "vector": extraction_result["requirement"]["semantic_vector"]
}
code_data = {
    "id": 1,
    "elements": extraction_result["code"]["semantic_elements"],
    "vector": extraction_result["code"]["semantic_vector"],
    "code_segment": code_text
}
alignment_result = align_semantics(req_data, code_data, eq_text, code_text)

# 整合完整结果（语义提取+语义对齐）
complete_result = {
    "双向语义提取": extraction_result,
    "语义对齐结果": alignment_result
}

# 严格按照你的格式打印输出
print("="*100)
print("✅ 等价文本与代码的语义提取+对齐完整Result结果：")
print("="*100) 
print(complete_result)
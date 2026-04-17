"""
【修改】增强的语义对齐模块
支持语法库、语义映射规则库、相似度计算与规则匹配结合
实现自然语言与代码的双向语义精准对齐
"""

import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
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
    status: AlignmentStatus
    confidence: float
    reason: str = ""
    mapping_confidence: float = 0.0  # 【改进】细粒度关键词匹配度（标识符拆分+映射表法）
    alignment_pairs: List[Dict] = field(
        default_factory=list
    )  # ✨ 新增：对齐对列表（用于深度学习）
    debug_info: Dict = field(default_factory=dict)  # 【新增】调试信息（映射详情、代码词汇等）


class NLPSyntaxLibrary:
    """
    【修改】自然语言语法库
    用于解析需求文本的语法结构，提取关键语义成分
    """

    def __init__(self):
        """初始化自然语言语法库"""
        # 【修改】中文FPGA领域语义模式库
        self.chinese_patterns = {
            "timing_constraints": {
                "keywords": [
                    "时钟周期","延迟","频率","时序",
                    "脉冲宽度","建立时间","保持时间",
                ],
                "semantic_type": "timing",
            },
            "io_specifications": {
                "keywords": ["输入", "输出", "端口", "信号", "数据", "地址", "控制"],
                "semantic_type": "io",
            },
            "memory_operations": {
                "keywords": [
                    "存储","读取","写入","内存",
                    "RAM","地址","数据深度","位宽",
                ],
                "semantic_type": "memory",
            },
            "control_logic": {
                "keywords": ["控制", "条件", "选择", "使能", "复位", "同步", "异步"],
                "semantic_type": "control",
            },
            "datapath": {
                "keywords": ["数据通路", "处理", "运算", "计数", "计数器", "累加"],
                "semantic_type": "datapath",
            },
            "synchronization": {
                "keywords": ["同步", "异步", "时钟域", "握手", "信号同步"],
                "semantic_type": "synchronization",
            },
        }

        # 【修改】英文FPGA领域语义模式库
        self.english_patterns = {
            "timing_constraints": {
                "keywords": [
                    "clock","cycle","delay","frequency",
                    "timing","pulse","setup","hold",
                ],
                "semantic_type": "timing",
            },
            "io_specifications": {
                "keywords": [
                    "input","output","port","signal",
                    "data","address","control",
                ],
                "semantic_type": "io",
            },
            "memory_operations": {
                "keywords": [
                    "memory","read","write","ram",
                    "address","depth","width",
                ],
                "semantic_type": "memory",
            },
            "control_logic": {
                "keywords": [
                    "control","condition","select",
                    "enable","reset","sync","async",
                ],
                "semantic_type": "control",
            },
            "datapath": {
                "keywords": [
                    "datapath","process","compute",
                    "count","counter","accumulate",
                ],
                "semantic_type": "datapath",
            },
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
        is_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)
        pattern_lib = self.chinese_patterns if is_chinese else self.english_patterns

        for pattern_name, pattern_info in pattern_lib.items():
            matched_keywords = []
            for keyword in pattern_info["keywords"]:
                if keyword in text_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:
                patterns_found[pattern_name] = {
                    "keywords": matched_keywords,
                    "semantic_type": pattern_info["semantic_type"]
                }

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
            "timing_logic": {
                "patterns": [
                    r"always\s*@\s*\(\s*(?:posedge|negedge)\s+\w+",
                    r"#\d+",
                    r"@\(.*?clk.*?\)",
                ],
                "semantic_type": "timing",
            },
            "io_declarations": {
                "patterns": [
                    r"(?:input|output|inout)\s+(?:reg|wire)?\s*\[?\d*:\d*\]?\s+\w+",
                    r"parameter\s+\w+\s*=\s*\d+",
                ],
                "semantic_type": "io",
            },
            "memory_structures": {
                "patterns": [
                    r"reg\s+\[?\d*:\d*\]?\s+\w+\s*\[[\d:]+\]",
                    r"reg\s+(?:.*?)\s+\w+_mem",
                    r"RAM|memory|mem",
                ],
                "semantic_type": "memory",
            },
            "control_structures": {
                "patterns": [
                    r"if\s*\(",
                    r"case\s*\(",
                    r"always\s*@\s*\(\*\)",
                    r"rst.*?(?:==|!=)",
                    r"enable|en",
                ],
                "semantic_type": "control",
            },
            "datapath_logic": {
                "patterns": [r"<=", r"\+=", r"count\s*<=", r"&|\\||\^|-|\+|\*"],
                "semantic_type": "datapath",
            },
            "synchronization": {
                "patterns": [
                    r"always\s*@\s*\(\s*posedge\s+clk",
                    r"synchronized|sync",
                    r"metastable",
                ],
                "semantic_type": "synchronization",
            },
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
            for pattern in construct_info["patterns"]:
                matches = re.findall(pattern, code_lower, re.IGNORECASE)
                if matches:
                    matched_patterns.extend(matches[:3])  # 最多保留3个匹配

            if matched_patterns:
                constructs_found[construct_name] = {
                    "patterns": list(set(matched_patterns)),
                    "semantic_type": construct_info["semantic_type"]
                }

        return constructs_found


class SemanticMappingRulesLibrary:
    """
    【修改】语义映射规则库
    定义自然语言与代码之间的语义对应关系
    """

    def __init__(self):
        """【修改】初始化增强的语义映射规则库"""
        # 【新增】中文关键词 → 英文关键词的映射表（用于细粒度关键词对应）
        self.chinese_to_english_mapping = self._build_chinese_english_mapping()

    def _build_chinese_english_mapping(self) -> Dict[str, List[str]]:
        """
        【新增】构建中文关键词 → 英文关键词的映射表
        
        这个映射表用于实现真正的细粒度关键词对应关系（已大幅扩展）

        Returns:
            中文关键词到英文关键词列表的映射字典
        """
        mapping = {
            # 时钟与时序相关
            "时钟": ["clock", "clk", "cycle", "frequency"],
            "时钟周期": ["cycle", "period", "clk"],
            "频率": ["frequency", "hz", "mhz"],
            "延迟": ["delay", "latency", "delaytime"],
            "时序": ["timing", "time"],
            "脉冲": ["pulse", "clk"],
            "脉冲宽度": ["pulse", "width"],
            "建立时间": ["setup", "time"],
            "保持时间": ["hold", "time"],
            "触发": ["trigger", "fire", "edge", "posedge", "negedge"],  # 【新增】
            "上升": ["rise", "posedge", "rising", "edge"],  # 【新增】
            "下降": ["fall", "negedge", "falling", "edge"],  # 【新增】
            "沿": ["edge", "posedge", "negedge"],  # 【新增】
            
            # 输入输出相关
            "输入": ["input", "in"],
            "输出": ["output", "out"],
            "端口": ["port", "pin"],
            "信号": ["signal", "sig"],
            "数据": ["data", "dat"],
            "地址": ["address", "addr"],
            "控制": ["control", "ctrl"],
            
            # 存储与内存相关
            "存储": ["memory", "ram", "storage"],
            "读取": ["read", "rd"],
            "写入": ["write", "wr"],
            "内存": ["memory", "ram", "mem"],
            "RAM": ["ram", "memory"],
            "深度": ["depth", "deep"],
            "位宽": ["width", "bit", "bits"],
            "宽度": ["width", "bit", "bits"],  # 【新增】
            "比特": ["bit", "bits"],
            
            # 控制逻辑相关
            "复位": ["reset", "rst"],
            "重置": ["reset", "clear"],
            "初始化": ["initial", "init"],
            "清零": ["clear", "clr"],
            "清除": ["clear", "clr"],  # 【新增】
            "条件": ["condition", "if"],
            "选择": ["select", "mux"],
            "使能": ["enable", "en"],
            "同步": ["sync", "synchronous"],
            "异步": ["async", "asynchronous"],
            
            # 数据通路与移位相关
            "计数": ["count", "counter"],
            "计数器": ["counter", "cnt"],
            "累加": ["accumulate", "add"],
            "运算": ["compute", "operation"],
            "处理": ["process", "handle"],
            "移位": ["shift"],  # 【新增】
            "左移": ["shift", "left", "sll"],  # 【新增】
            "右移": ["shift", "right", "srl"],  # 【新增】
            "位移": ["shift", "slr"],  # 【新增】
            "寄存器": ["register", "reg"],  # 【新增】
            
            # 模块与结构相关
            "模块": ["module", "block"],
            "实例": ["instance", "inst"],
            "子模块": ["submodule", "module"],
            "参数": ["parameter", "param"],
            "可配置": ["configurable", "parameter"],
            
            # 分频相关
            "分频": ["divide", "divider"],  # 【新增】
            "分频器": ["divider", "divide"],  # 【新增】
            "分倍": ["frequency"],  # 【新增】
            
            # 时钟域相关
            "时钟域": ["clock", "domain"],
            "握手": ["handshake", "ack"],
            "信号同步": ["synchronize", "sync"],
            "双端口": ["dual", "port", "dpram"],
            "总线": ["bus", "interface"],
            "片选": ["chip_select", "cs"],
            "读写控制": ["wr_rd", "control"],
            
            # 并行相关
            "并行": ["parallel", "parallel"],  # 【新增】
            "串行": ["serial", "serial"],  # 【新增】
            "加载": ["load", "latch"],  # 【新增】
            "锁存": ["latch", "hold"],  # 【新增】
            
            # 编码/优先级相关
            "编码": ["encode", "coder"],  # 【新增】
            "编码器": ["encoder", "coder"],  # 【新增】
            "优先级": ["priority", "prior"],  # 【新增】
            "二进制": ["binary", "bin"],  # 【新增】
            "十进制": ["decimal", "dec"],  # 【新增】
            
            # 多路选择相关
            "多路": ["mux", "multiplexer", "select"],  # 【新增】
            "选择器": ["selector", "mux"],  # 【新增】
            "多路选择": ["multiplexer", "mux"],  # 【新增】
        }
        return mapping

    def find_semantic_mappings(
        self, nlp_keywords: List[str], code_keywords: List[str], debug: bool = False
    ) -> Tuple[float, Dict]:
        """
        【改进】计算细粒度关键词对应置信度 - 智能拆分+映射表查询
        
        改进策略：
        1. 只计算在映射表中的NLP关键词
        2. 对Code标识符进行"拆分"（underscore分离）
        3. 计算映射成功率
        
        Args:
            nlp_keywords: 需求提取的关键字（通常包含中文）
            code_keywords: 代码提取的关键字（标识符）
            debug: 是否返回调试信息

        Returns:
            (关键词匹配置信度, 调试信息字典) 元组
        """
        debug_info = {}  # 【新增】调试信息收集
        
        if not nlp_keywords or not code_keywords:
            return 0.0, debug_info
        
        # 【改进】Step 1: 从NLP关键词中筛选出"有领域知识"的词（在映射表中的词）
        mapped_nlp_keywords = [
            kw for kw in nlp_keywords 
            if kw in self.chinese_to_english_mapping
        ]
        
        debug_info["mapped_nlp_keywords"] = mapped_nlp_keywords  # 【新增】
        
        if not mapped_nlp_keywords:
            if debug:
                print(f"  [DEBUG] 没有映射的NLP关键词（原始: {nlp_keywords}）")
            return 0.0, debug_info
        
        if debug:
            print(f"  [DEBUG] 映射的NLP关键词: {mapped_nlp_keywords}")
        
        # 【改进】Step 2: 把Code标识符拆分成词汇成分
        # 例如: clk_in → {clk, in}, counter_en → {counter, en}
        code_word_parts = set()
        for identifier in code_keywords:
            # 按下划线分割
            parts = identifier.lower().split('_')
            code_word_parts.update(parts)
        
        debug_info["code_word_parts"] = list(code_word_parts)  # 【新增】
        
        if debug:
            print(f"  [DEBUG] Code标识符分割结果: {code_word_parts}")
        
        # 【改进】Step 3: 对每个NLP关键词，检查其映射的英文词是否在Code词汇中
        successful_mappings = 0
        match_details = []  # 用于调试
        
        for nlp_kw in mapped_nlp_keywords:
            # 获取该中文关键词对应的英文关键词列表
            english_equivalents = self.chinese_to_english_mapping[nlp_kw]
            
            # 检查是否至少有一个英文对应词在Code词汇部件中出现
            for en_kw in english_equivalents:
                en_kw_lower = en_kw.lower()
                # 检查完整匹配或作为词根匹配（例如 clk 在 clk_in 中）
                if en_kw_lower in code_word_parts:
                    successful_mappings += 1
                    match_details.append(f"{nlp_kw}→{en_kw_lower}✓")
                    break  # 只要找到一个匹配就算成功
            else:
                # 没有找到任何匹配
                match_details.append(f"{nlp_kw}→{english_equivalents}✗")
        
        debug_info["match_details"] = match_details  # 【新增】
        debug_info["successful_mappings"] = successful_mappings  # 【新增】
        debug_info["total_mapped_keywords"] = len(mapped_nlp_keywords)  # 【新增】
        
        if debug:
            print(f"  [DEBUG] 匹配详情: {match_details}")
            print(f"  [DEBUG] 成功匹配数: {successful_mappings}/{len(mapped_nlp_keywords)}")
        
        # 【改进】计算成功率
        mapping_confidence = successful_mappings / len(mapped_nlp_keywords)
        return min(mapping_confidence, 1.0), debug_info




class SemanticAligner:
    """【修改】增强的语义对齐器 - 支持语法库和语义映射规则"""

    def __init__(self):
        """
        【修改】初始化增强的语义对齐器

        Args:
            rules_library: 映射规则库
        """

        # 【修改】初始化语法库
        self.nlp_syntax = NLPSyntaxLibrary()
        self.code_syntax = CodeSyntaxLibrary()

        # 【修改】初始化语义映射规则库
        self.semantic_mapping_lib = SemanticMappingRulesLibrary()

        self.similarity_threshold = 0.7
        self.high_similarity_threshold = 0.8
        self.low_similarity_threshold = 0.5


    def align_requirements_to_code(
        self,
        req_id: int,
        req_elements: Dict,
        req_vector: np.ndarray,
        code_elements: Dict,
        code_vector: np.ndarray,
        code_segment: str,
        req_text: str = "",
        code_text: str = "",
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

        # 【修改】第2步：提取NLP关键字和代码构造
        # NLP返回的是tokens列表
        req_keywords = req_elements.get("keywords", [])
        
        # 代码返回的keywords可能是字典（Verilog关键字） 或列表（标识符）
        # 我们需要提取真实有意义的标识符：信号、端口、模块名等
        code_keywords_raw = code_elements.get("keywords", {})
        if isinstance(code_keywords_raw, dict):
            # 如果是字典，提取键（Verilog关键字）
            # 但这些不如实际标识符有用，所以我们改用信号和端口名
            signals_list = code_elements.get("signals", [])
            ports_list = code_elements.get("ports", [])
            modules_list = code_elements.get("modules", [])
            
            # 从这些结构中提取实际标识符
            code_keywords = []
            for sig in signals_list:
                if isinstance(sig, dict) and "name" in sig:
                    code_keywords.append(sig["name"])
            for port in ports_list:
                if isinstance(port, dict) and "name" in port:
                    code_keywords.append(port["name"])
            code_keywords.extend(modules_list)
        else:
            # 如果已经是列表，直接使用
            code_keywords = code_keywords_raw if code_keywords_raw else []

        # 【修改】第3步：使用语法库进行深层分析
        nlp_patterns = (
            self.nlp_syntax.extract_semantic_patterns(req_text) if req_text else {}
        )
        code_constructs = (
            self.code_syntax.extract_code_constructs(code_text) if code_text else {}
        )

        # 【改进】第4步：计算两个维度的对齐度
        # 维度1: 细粒度关键词匹配（改进的拆分+映射表方法）
        mapping_confidence, debug_info_mapping = self.semantic_mapping_lib.find_semantic_mappings(
            req_keywords, code_keywords, debug=True
        )
        
        # 🔍 调试输出（已关闭以简化输出）
        # print(f"  [DEBUG] req_keywords: {req_keywords[:8]}...")
        # print(f"  [DEBUG] code_keywords: {code_keywords[:8]}...")

        # 维度2: 粗粒度语义类型匹配（semantic_type重叠）
        semantic_type_match = 0.0
        if nlp_patterns and code_constructs:
            nlp_types = set(v["semantic_type"] for v in nlp_patterns.values())
            code_types = set(v["semantic_type"] for v in code_constructs.values())
            
            if nlp_types and code_types:
                overlaps = len(nlp_types & code_types)
                union = len(nlp_types | code_types)
                semantic_type_match = overlaps / union if union > 0 else 0.0


        # 【修改】第6步：综合决定对齐状态（同时考虑细粒度+粗粒度）
        status, confidence, reason = self._determine_alignment_status(
            req_keywords,
            code_keywords,
            mapping_confidence,  # 细粒度关键词匹配
            semantic_type_match=semantic_type_match,  # 粗粒度类型匹配
            nlp_patterns=nlp_patterns,
            code_constructs=code_constructs,
        )

        # ✨ 生成alignment_pairs - 用于深度学习模型推理
        alignment_pairs = self._generate_alignment_pairs(
            req_keywords, code_keywords
        )

        result = AlignmentResult(
            req_id=req_id,
            code_segment=code_segment[:100],
            status=status,
            confidence=confidence,
            reason=reason,
            mapping_confidence=mapping_confidence,  # 【修改】加入语义映射置信度
            alignment_pairs=alignment_pairs,  # ✨ 传递对齐对
            debug_info=debug_info_mapping,  # 【新增】传递调试信息
        )

        return result

    def _determine_alignment_status(
        self,
        req_keywords: List[str],
        code_keywords: List[str],
        mapping_confidence: float = 0.0,
        semantic_type_match: float = 0.0,
        nlp_patterns: Dict = None,
        code_constructs: Dict = None,
    ) -> Tuple[AlignmentStatus, float, str]:
        """
        【改进】确定对齐状态 - 结合细粒度+粗粒度匹配
        
        维度1: mapping_confidence = 细粒度关键词匹配（拆分+映射表方法）
        维度2: semantic_type_match = 粗粒度语义类型匹配
        
        Args:
            req_keywords: 需求关键字
            code_keywords: 代码关键字
            mapping_confidence: 细粒度关键词对应置信度
            semantic_type_match: 粗粒度语义类型匹配度
            nlp_patterns: NLP提取的模式（保留向后兼容）
            code_constructs: 代码提取的构造（保留向后兼容）

        Returns:
            (状态, 置信度, 原因) 元组
        """
        # 【改进】综合评分 = 细粒度关键词 60% + 粗粒度类型 40%
        composite_score = (
            mapping_confidence * 0.60 
            + semantic_type_match * 0.4
        )
        
        # 1. 高质量对齐：两个维度都不错
        if composite_score >= 0.75:
            confidence = min(composite_score, 1.0)
            reason = f"高质量对齐：关键词={mapping_confidence:.2f}, 类型={semantic_type_match:.2f}"
            return AlignmentStatus.ALIGNED, confidence, reason

        # 2. 良好对齐：综合分达标
        elif composite_score >= 0.55:
            confidence = min(composite_score * 0.9, 1.0)
            reason = f"良好对齐：综合评分={composite_score:.2f}"
            return AlignmentStatus.ALIGNED, confidence, reason

        # 3. 可疑对齐：单个维度尚可但综合偏弱
        elif composite_score >= 0.35 or mapping_confidence >= 0.3 or semantic_type_match >= 0.4:
            confidence = min(composite_score * 0.7, 1.0)
            reason = f"可疑对齐：关键词={mapping_confidence:.2f}, 类型={semantic_type_match:.2f}"
            return AlignmentStatus.SUSPICIOUS, confidence, reason

        # 4. 无有效匹配
        else:
            return AlignmentStatus.UNALIGNED, max(composite_score * 0.3, 0.0), f"对齐失败：关键词={mapping_confidence:.2f}, 类型={semantic_type_match:.2f}"
    
    def _generate_alignment_pairs(
        self,
        req_keywords: List[str],
        code_keywords: List[str],
    ) -> List[Dict]:
        """
        ✨ 生成对齐对 - 用于深度学习模型推理

        Args:
            req_keywords: 需求关键字
            code_keywords: 代码关键字

        Returns:
            对齐对列表 [{'req_idx': i, 'code_idx': j, 'score': s}, ...]
        """
        alignment_pairs = []
        
        # 基于关键词相似度生成对齐对
        for i, req_kw in enumerate(req_keywords):
            for j, code_kw in enumerate(code_keywords):
                # 简单起见：如果两个关键词的小写形式相同或有子字符串包含关系，就算匹配
                req_lower = req_kw.lower()
                code_lower = code_kw.lower()
                
                if req_lower == code_lower or req_lower in code_lower or code_lower in req_lower:
                    alignment_pairs.append(
                        {
                            "req_idx": i,
                            "code_idx": j,
                            "score": 0.9,  # 直接关键词匹配
                            "confidence": 0.9,
                        }
                    )
        
        # 如果没有关键词对齐，返回空列表（表示无法对齐）
        return alignment_pairs

    def batch_align(
        self,
        requirements: List[Dict],
        code_elements_list: List[Dict],
        code_vectors_list: List[np.ndarray],
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
                    req_id=req.get("id", i),
                    req_elements=req.get("elements", {}),
                    req_vector=req.get("vector"),
                    code_elements=code_elements_list[i],
                    code_vector=code_vectors_list[i],
                    code_segment=req.get("code_segment", ""),
                    req_text=req.get("text", ""),
                    code_text=req.get("code_text", ""),
                )
                results.append(result)

        return results


def align_semantics(
    req_data: Dict, code_data: Dict, req_text: str = "", code_text: str = ""
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
        req_id=req_data.get("id", 0),
        req_elements=req_data.get("elements", {}),
        req_vector=req_data.get("vector"),
        code_elements=code_data.get("elements", {}),
        code_vector=code_data.get("vector"),
        code_segment=code_data.get("code_segment", ""),
        req_text=req_text,
        code_text=code_text,
    )

    return {
        "alignment_result": result,
        "status": result.status.value,
        "confidence": result.confidence,
        "mapping_confidence": result.mapping_confidence,  # 关键词匹配置信度
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
        "id": 1,
        "elements": extraction_result["requirement"]["semantic_elements"],
        "vector": extraction_result["requirement"]["semantic_vector"],
        "text": eq_text,  # 【修改】加入原文本用于语法库分析
    }

    code_data_for_align = {
        "id": 1,
        "elements": extraction_result["code"]["semantic_elements"],
        "vector": extraction_result["code"]["semantic_vector"],
        "code_segment": code_text,
        "code_text": code_text,  # 【修改】加入代码文本用于构造分析
    }

    # 4. 第三步：运行增强的语义对齐
    print("🔗 正在进行【增强】语义对齐（使用语法库和语义映射规则库）...")
    aligner = SemanticAligner()

    alignment_result = aligner.align_requirements_to_code(
        req_id=req_data_for_align["id"],
        req_elements=req_data_for_align["elements"],
        req_vector=req_data_for_align["vector"],
        code_elements=code_data_for_align["elements"],
        code_vector=code_data_for_align["vector"],
        code_segment=code_data_for_align["code_segment"],
        req_text=req_data_for_align.get("text", ""),
        code_text=code_data_for_align.get("code_text", ""),
    )

    # 5. 第四步：打印美观的结果
    print("\n" + "=" * 100)
    print("✅ 【双端口RAM】需求-代码对齐最终报告（【修改】增强版本）")
    print("=" * 100)

    # 状态映射成中文，方便阅读
    status_cn_map = {
        "aligned": "✅ 完全对齐",
        "suspicious": "⚠️ 存疑对齐",
        "unaligned": "❌ 未对齐",
        "indirect": "🔗 间接对齐",
    }

    print(f"1. 对齐状态: {status_cn_map.get(alignment_result.status.value, '未知')}")
    print(f"2. 关键词匹配置信度: {alignment_result.mapping_confidence:.4f}")
    print(f"3. 综合置信度: {alignment_result.confidence:.4f}")
    print(f"4. 判定原因: {alignment_result.reason}")
    print(f"5. 代码片段预览: {alignment_result.code_segment[:80]}...")

    # 额外打印匹配到的FPGA术语（如果有）
    fpga_terms = extraction_result["requirement"]["semantic_elements"].get(
        "fpga_terms", []
    )
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
    ast_info = extraction_result["code"]["semantic_elements"].get("ast_nodes", {})
    if ast_info:
        print(f"  - AST根节点: {ast_info.get('type')}")
        print(f"  - AST子节点统计: {ast_info.get('children_summary', {})}")

    code_complexity = extraction_result["code"]["semantic_elements"].get(
        "code_complexity", "N/A"
    )
    print(f"  - 代码复杂度: {code_complexity}")

    print("=" * 100 + "\n")

# 测试
# eq_text = "FPGA双端口RAM模块，数据位宽固定为8比特；采用单总线时钟实现双端口RAM逻辑；端口A与总线绑定，端口B为通用业务端口；总线侧读写控制规则：在1个时钟周期内同时置位片选信号、8位地址信号、读写控制信号，即可执行读或写操作；写操作时序：写数据在寻址时立即被写入对应内存地址；读操作时序：读请求触发后，有效数据标志信号将延迟1个时钟周期脉冲，此时总线读数据端口输出对应内存数据；模块可配置参数：DEPTH为双端口RAM的存储深度，代表存储的8比特字长数据的个数。"
# code_text = "module Bus8_DPRAM #(DEPTH = 256)(input i_Bus_Rst_L,input i_Bus_Clk,input i_Bus_CS,input i_Bus_Wr_Rd_n,input [$clog2(DEPTH)-1:0] i_Bus_Addr8,input [7:0] i_Bus_Wr_Data,output [7:0] o_Bus_Rd_Data,output reg o_Bus_Rd_DV,input [7:0] i_PortB_Data,input [$clog2(DEPTH)-1:0] i_PortB_Addr8,input i_PortB_WE,output [7:0] o_PortB_Data);Dual_Port_RAM_Single_Clock #(.WIDTH(8),.DEPTH(DEPTH)) Bus_RAM_Inst(.i_Clk(i_Bus_Clk),.i_PortA_Data(i_Bus_Wr_Data),.i_PortA_Addr(i_Bus_Addr8),.i_PortA_WE(i_Bus_Wr_Rd_n & i_Bus_CS),.o_PortA_Data(o_Bus_Rd_Data),.i_PortB_Data(i_PortB_Data),.i_PortB_Addr(i_PortB_Addr8),.i_PortB_WE(i_PortB_WE),.o_PortB_Data(o_PortB_Data));always @(posedge i_Bus_Clk)begin o_Bus_Rd_DV <= i_Bus_CS & ~i_Bus_Wr_Rd_n;end endmodule"

# # 1. 执行双向语义提取（保留你原有的函数）
# extraction_result = extract_bidirectional_semantics(eq_text, code_text)

# # 2. 执行语义对齐（新增核心功能，无缝衔接）
# req_data = {
#     "id": 1,
#     "elements": extraction_result["requirement"]["semantic_elements"],
#     "vector": extraction_result["requirement"]["semantic_vector"],
# }
# code_data = {
#     "id": 1,
#     "elements": extraction_result["code"]["semantic_elements"],
#     "vector": extraction_result["code"]["semantic_vector"],
#     "code_segment": code_text,
# }
# alignment_result = align_semantics(req_data, code_data, eq_text, code_text)

# # 整合完整结果（语义提取+语义对齐）
# complete_result = {"双向语义提取": extraction_result, "语义对齐结果": alignment_result}

# # 严格按照你的格式打印输出
# print("=" * 100)
# print("✅ 等价文本与代码的语义提取+对齐完整Result结果：")
# print("=" * 100)
# print(complete_result)

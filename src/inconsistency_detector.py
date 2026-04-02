"""
不一致检测模块
实现显性和隐性的不一致检测
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class InconsistencyType(Enum):
    """不一致类型"""

    EXPLICIT = "explicit"  # 显性不一致
    IMPLICIT = "implicit"  # 隐性不一致
    MISSING = "missing"  # 缺失不一致
    EXTRA = "extra"  # 多余不一致
    CONFLICT = "conflict"  # 冲突不一致


class SeverityLevel(Enum):
    """严重程度"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class InconsistencyReport:
    """不一致报告"""

    req_id: int
    inconsistency_type: InconsistencyType
    severity: SeverityLevel
    description: str
    location: str
    confidence: float
    rule_id: Optional[str] = None
    suggestion: Optional[str] = None


class RulesEngine:
    """规则引擎 - 用于显性不一致检测"""

    def __init__(self):
        """初始化规则引擎"""
        self.rules = self._initialize_rules()

    def _initialize_rules(self) -> List[Dict]:
        """
        初始化显性不一致检测规则

        Returns:
            规则列表
        """
        rules = [
            # 存在性规则
            {
                "id": "rule_exist_clock",
                "name": "Clock Signal Existence",
                "type": "existence",
                "requirement": "must_contain_keyword",
                "keywords": ["clock", "clk"],
                "code_patterns": [r"\bclk\b", r"always.*@"],
                "severity": SeverityLevel.HIGH,
                "description": "需求指定时钟信号，代码中必须有相应实现",
            },
            {
                "id": "rule_exist_reset",
                "name": "Reset Signal Existence",
                "type": "existence",
                "requirement": "must_contain_keyword",
                "keywords": ["reset", "rst"],
                "code_patterns": [r"\brst\b", r"reset"],
                "severity": SeverityLevel.HIGH,
                "description": "需求指定复位信号，代码中必须有相应实现",
            },
            # 匹配性规则
            {
                "id": "rule_match_width",
                "name": "Bit Width Matching",
                "type": "matching",
                "requirement": "width_match",
                "patterns": [r"(\d+)\s*(?:bit|位)", r"\[(\d+):\d+\]"],
                "severity": SeverityLevel.HIGH,
                "description": "需求指定的位宽与代码实现不匹配",
            },
            {
                "id": "rule_match_frequency",
                "name": "Clock Frequency Matching",
                "type": "matching",
                "requirement": "frequency_match",
                "patterns": [r"(\d+)\s*(?:mhz|ghz|hz)", r"timescale"],
                "severity": SeverityLevel.MEDIUM,
                "description": "需求指定的频率与代码实现不匹配",
            },
            # 完整性规则
            {
                "id": "rule_complete_ports",
                "name": "Port Completeness",
                "type": "completeness",
                "requirement": "port_count_match",
                "severity": SeverityLevel.MEDIUM,
                "description": "需求中的端口与代码实现的端口数量不匹配",
            },
            {
                "id": "rule_complete_behavior",
                "name": "Behavioral Completeness",
                "type": "completeness",
                "requirement": "behavior_match",
                "patterns": [r"always", r"if.*else", r"case"],
                "severity": SeverityLevel.MEDIUM,
                "description": "需求描述的行为在代码中未完全实现",
            },
        ]
        return rules

    def check_existence_rules(
        self, req_keywords: List[str], code_text: str, req_text: str
    ) -> List[InconsistencyReport]:
        """
        检查存在性规则

        Args:
            req_keywords: 需求关键字
            code_text: 代码文本
            req_text: 需求文本

        Returns:
            不一致报告列表
        """
        inconsistencies = []

        for rule in self.rules:
            if rule["type"] != "existence":
                continue

            # 检查需求中是否包含关键字
            has_requirement = any(
                kw.lower() in req_text.lower() for kw in rule["keywords"]
            )

            if has_requirement:
                # 检查代码中是否有相应实现
                has_implementation = any(
                    re.search(pattern, code_text, re.IGNORECASE)
                    for pattern in rule["code_patterns"]
                )

                if not has_implementation:
                    inconsistencies.append(
                        InconsistencyReport(
                            req_id=0,
                            inconsistency_type=InconsistencyType.MISSING,
                            severity=rule["severity"],
                            description=rule["description"],
                            location=f"Code missing: {rule['name']}",
                            confidence=0.95,
                            rule_id=rule["id"],
                            suggestion=f"Add implementation for {rule['name']}",
                        )
                    )

        return inconsistencies

    def check_matching_rules(
        self, req_elements: Dict, code_elements: Dict
    ) -> List[InconsistencyReport]:
        """
        检查匹配性规则

        Args:
            req_elements: 需求语义要素
            code_elements: 代码语义要素

        Returns:
            不一致报告列表
        """
        inconsistencies = []

        # 检查端口数量匹配
        if "port_count" in req_elements and "port_count" in code_elements:
            if req_elements["port_count"] != code_elements["port_count"]:
                inconsistencies.append(
                    InconsistencyReport(
                        req_id=0,
                        inconsistency_type=InconsistencyType.CONFLICT,
                        severity=SeverityLevel.HIGH,
                        description="Port count mismatch between requirement and code",
                        location=f"Requirement: {req_elements['port_count']} ports, "
                        f"Code: {code_elements['port_count']} ports",
                        confidence=0.9,
                        rule_id="rule_complete_ports",
                        suggestion="Update code to match requirement port count",
                    )
                )

        # 检查模块数量
        if "modules" in req_elements and "modules" in code_elements:
            req_modules = set(req_elements.get("modules", []))
            code_modules = set(code_elements.get("modules", []))

            missing_modules = req_modules - code_modules
            if missing_modules:
                inconsistencies.append(
                    InconsistencyReport(
                        req_id=0,
                        inconsistency_type=InconsistencyType.MISSING,
                        severity=SeverityLevel.HIGH,
                        description=f"Missing modules: {missing_modules}",
                        location="Module definition",
                        confidence=0.85,
                        rule_id="rule_complete_behavior",
                        suggestion=f"Add missing modules: {missing_modules}",
                    )
                )

        return inconsistencies

    def check_completeness_rules(
        self, req_elements: Dict, code_elements: Dict
    ) -> List[InconsistencyReport]:
        """
        检查完整性规则

        Args:
            req_elements: 需求语义要素
            code_elements: 代码语义要素

        Returns:
            不一致报告列表
        """
        inconsistencies = []

        # 检查信号完整性
        req_signals = set(req_elements.get("keywords", []))
        code_signals = set(code_elements.get("keywords", []))

        missing_signals = req_signals - code_signals
        if missing_signals:
            inconsistencies.append(
                InconsistencyReport(
                    req_id=0,
                    inconsistency_type=InconsistencyType.MISSING,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Missing signal implementations: {missing_signals}",
                    location="Signal definition",
                    confidence=0.75,
                    rule_id="rule_complete_behavior",
                    suggestion=f"Add missing signals: {missing_signals}",
                )
            )

        return inconsistencies


class ImplicitInconsistencyDetector:
    """隐性不一致检测器 - 基于GAT+Bi-GRU模型的检测"""

    def __init__(self):
        """初始化隐性不一致检测器"""
        self.similarity_threshold = 0.5
        self.high_deviation_threshold = 0.3

    def detect_semantic_gap(
        self, req_vector: np.ndarray, code_vector: np.ndarray
    ) -> Tuple[float, SeverityLevel]:
        """
        检测语义间隙

        Args:
            req_vector: 需求语义向量
            code_vector: 代码语义向量

        Returns:
            (间隙度, 严重程度) 元组
        """
        if req_vector is None or code_vector is None:
            return 0.0, SeverityLevel.INFO

        # 计算欧氏距离
        distance = np.linalg.norm(req_vector - code_vector)
        gap_score = min(distance, 2.0) / 2.0  # 归一化到 [0, 1]

        if gap_score > 0.7:
            severity = SeverityLevel.CRITICAL
        elif gap_score > 0.5:
            severity = SeverityLevel.HIGH
        elif gap_score > 0.3:
            severity = SeverityLevel.MEDIUM
        elif gap_score > 0.1:
            severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.INFO

        return gap_score, severity

    def detect_context_inconsistency(
        self, req_elements: Dict, code_elements: Dict
    ) -> List[InconsistencyReport]:
        """
        检测上下文不一致

        Args:
            req_elements: 需求语义要素
            code_elements: 代码语义要素

        Returns:
            不一致报告列表
        """
        inconsistencies = []

        # 【修改】检测FPGA特定特征的不一致 - 添加防守性处理
        req_features = set(
            elem.get("type")
            for elem in req_elements.get("fpga_terms", [])
            if elem.get("type")  # 过滤None值
        )
        code_features = set(
            feat.get("type")  # 【修改】改为使用.get()以免KeyError
            for feat in code_elements.get("fpga_features", [])
            if feat.get("type")  # 过滤None值
        )

        missing_features = req_features - code_features

        for feature in missing_features:
            inconsistencies.append(
                InconsistencyReport(
                    req_id=0,
                    inconsistency_type=InconsistencyType.IMPLICIT,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Missing FPGA feature: {feature}",
                    location="Design feature",
                    confidence=0.7,
                    suggestion=f"Add {feature} implementation",
                )
            )

        return inconsistencies

    def detect_behavior_inconsistency(
        self, req_keywords: List[str], code_structure: Dict
    ) -> List[InconsistencyReport]:
        """
        检测行为不一致

        Args:
            req_keywords: 需求关键字
            code_structure: 代码结构

        Returns:
            不一致报告列表
        """
        inconsistencies = []

        # 检测时序逻辑
        if any(kw in req_keywords for kw in ["synchronous", "sequential"]):
            if code_structure.get("behavior_count", 0) == 0:
                inconsistencies.append(
                    InconsistencyReport(
                        req_id=0,
                        inconsistency_type=InconsistencyType.IMPLICIT,
                        severity=SeverityLevel.HIGH,
                        description="Synchronous behavior required but not found in code",
                        location="Always block / Sequential logic",
                        confidence=0.85,
                        suggestion="Add always @(posedge clk) block",
                    )
                )

        return inconsistencies


class InconsistencyDetector:
    """统一的不一致检测器"""

    def __init__(self):
        """初始化不一致检测器"""
        self.rules_engine = RulesEngine()
        self.implicit_detector = ImplicitInconsistencyDetector()

    def detect_all_inconsistencies(
        self,
        req_id: int,
        req_text: str,
        req_elements: Dict,
        req_vector: np.ndarray,
        code_text: str,
        code_elements: Dict,
        code_vector: np.ndarray,
    ) -> Dict:
        """
        检测所有类型的不一致

        Args:
            req_id: 需求ID
            req_text: 需求文本
            req_elements: 需求语义要素
            req_vector: 需求向量
            code_text: 代码文本
            code_elements: 代码语义要素
            code_vector: 代码向量

        Returns:
            包含所有不一致的字典
        """
        result = {
            "req_id": req_id,
            "explicit_inconsistencies": [],
            "implicit_inconsistencies": [],
            "total_issues": 0,
            "severity_distribution": {},
        }

        # 检测显性不一致
        explicit = []
        explicit.extend(
            self.rules_engine.check_existence_rules(
                req_elements.get("keywords", []), code_text, req_text
            )
        )
        explicit.extend(
            self.rules_engine.check_matching_rules(req_elements, code_elements)
        )
        explicit.extend(
            self.rules_engine.check_completeness_rules(req_elements, code_elements)
        )

        # 检测隐性不一致
        implicit = []

        # 语义间隙检测
        gap_score, severity = self.implicit_detector.detect_semantic_gap(
            req_vector, code_vector
        )
        if gap_score > 0.1:
            implicit.append(
                InconsistencyReport(
                    req_id=req_id,
                    inconsistency_type=InconsistencyType.IMPLICIT,
                    severity=severity,
                    description=f"Semantic gap detected (score: {gap_score:.2f})",
                    location="Overall semantic alignment",
                    confidence=gap_score,
                )
            )

        # 上下文不一致检测
        implicit.extend(
            self.implicit_detector.detect_context_inconsistency(
                req_elements, code_elements
            )
        )

        # 行为不一致检测
        implicit.extend(
            self.implicit_detector.detect_behavior_inconsistency(
                req_elements.get("keywords", []), code_elements
            )
        )

        # 更新结果
        result["explicit_inconsistencies"] = [
            {
                "type": inc.inconsistency_type.value,
                "severity": inc.severity.value,
                "description": inc.description,
                "location": inc.location,
                "confidence": inc.confidence,
                "rule_id": inc.rule_id,
                "suggestion": inc.suggestion,
            }
            for inc in explicit
        ]

        result["implicit_inconsistencies"] = [
            {
                "type": inc.inconsistency_type.value,
                "severity": inc.severity.value,
                "description": inc.description,
                "location": inc.location,
                "confidence": inc.confidence,
                "suggestion": inc.suggestion,
            }
            for inc in implicit
        ]

        result["total_issues"] = len(explicit) + len(implicit)

        # 统计严重程度分布
        all_issues = explicit + implicit
        severity_counts = {}
        for issue in all_issues:
            key = issue.severity.value
            severity_counts[key] = severity_counts.get(key, 0) + 1
        result["severity_distribution"] = severity_counts

        return result

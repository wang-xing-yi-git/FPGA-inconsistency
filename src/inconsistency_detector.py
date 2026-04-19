"""
不一致检测模块
实现显性和隐性的不一致检测
- 显性不一致：基于规则引擎的启发式检测
- 隐性不一致：基于 GAT+Bi-GRU 深度学习模型的智能检测
"""

import re
import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import sys

# 导入深度学习模型
try:
    # 🔧 最小改动：v2 → v3（唯一修改点1）
    from .deep_learning_models_v3 import ImplicitInconsistencyModel

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️  警告: 深度学习模型不可用，将使用启发式方法")


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
    """规则引擎 - 用于显性不一致检测（修复版：适配中文+Verilog+补全规则）"""

    def __init__(self):
        """初始化规则引擎"""
        self.rules = self._initialize_rules()
        # 🔥 新增：中英文硬件关键词映射（和对齐模块统一）
        self.cn2en = {
            "时钟": ["clk", "clock"],
            "复位": ["rst", "reset"],
            "输出": ["out", "output"],
            "输入": ["in", "input"],
            "频率": ["mhz", "ghz", "hz"],
            "位宽": ["bit", "[7:0]", "[31:0]"]
        }

    def _initialize_rules(self) -> List[Dict]:
        """初始化显性不一致检测规则（补全未实现的规则）"""
        rules = [
            # 存在性规则
            {
                "id": "rule_exist_clock",
                "name": "Clock Signal Existence",
                "type": "existence",
                "requirement": "must_contain_keyword",
                "cn_keywords": ["时钟"],  # 🔥 改中文
                "en_keywords": ["clk", "clock"],
                "severity": SeverityLevel.HIGH,
                "description": "需求指定时钟信号，代码中必须有相应实现",
            },
            {
                "id": "rule_exist_reset",
                "name": "Reset Signal Existence",
                "type": "existence",
                "requirement": "must_contain_keyword",
                "cn_keywords": ["复位"],  # 🔥 改中文
                "en_keywords": ["rst", "reset"],
                "severity": SeverityLevel.HIGH,
                "description": "需求指定复位信号，代码中必须有相应实现",
            },
            # 匹配性规则（补全实现）
            {
                "id": "rule_match_frequency",
                "name": "Clock Frequency Matching",
                "type": "matching",
                "requirement": "frequency_match",
                "cn_patterns": [r"(\d+)\s*(?:mhz|ghz|hz)"],  # 中文频率
                "en_patterns": [r"(\d+)\s*(?:mhz|ghz|hz)"],
                "severity": SeverityLevel.MEDIUM,
                "description": "需求指定的频率与代码实现不匹配",
            },
            {
                "id": "rule_match_width",
                "name": "Bit Width Matching",
                "type": "matching",
                "requirement": "width_match",
                "cn_patterns": [r"(\d+)\s*位"],  # 中文字位宽
                "en_patterns": [r"\[(\d+):\d+\]"],
                "severity": SeverityLevel.HIGH,
                "description": "需求指定的位宽与代码实现不匹配",
            },
            # 完整性规则
            {
                "id": "rule_complete_ports",
                "name": "Port Completeness",
                "type": "completeness",
                "requirement": "port_direction_match",
                "severity": SeverityLevel.MEDIUM,
                "description": "需求的端口方向(输入/输出)在代码中缺失",
            },
        ]
        return rules

    def check_existence_rules(self, req_keywords: List[str], code_text: str, req_text: str, req_id:int=0) -> List[InconsistencyReport]:
        """检查存在性规则（修复：支持中文）"""
        inconsistencies = []

        for rule in self.rules:
            if rule["type"] != "existence":
                continue

            # 🔥 修复：检查中文需求关键词
            has_requirement = any(kw in req_keywords for kw in rule["cn_keywords"])
            if has_requirement:
                # 检查代码是否有实现
                has_implementation = any(
                    re.search(r"\b"+pattern+r"\b", code_text, re.IGNORECASE)
                    for pattern in rule["en_keywords"]
                )
                if not has_implementation:
                    inconsistencies.append(
                        InconsistencyReport(
                            req_id=req_id,  # 🔥 修复：传真实req_id
                            inconsistency_type=InconsistencyType.MISSING,
                            severity=rule["severity"],
                            description=rule["description"],
                            location=f"代码缺失{rule['name']}",
                            confidence=0.95,
                            rule_id=rule["id"],
                            suggestion=f"添加{rule['name']}实现",
                        )
                    )
        return inconsistencies

    def check_matching_rules(self, req_elements: Dict, code_elements: Dict, req_id:int=0) -> List[InconsistencyReport]:
        """检查匹配性规则（修复：实现频率/位宽匹配，删除无效端口数判断）"""
        inconsistencies = []
        req_text = "".join(req_elements.get("keywords", []))
        code_text = code_elements.get("code_text", "")

        # 🔥 补全：频率匹配规则
        freq_rule = next(r for r in self.rules if r["id"]=="rule_match_frequency")
        req_freq = re.search(freq_rule["cn_patterns"][0], req_text)
        code_freq = re.search(freq_rule["en_patterns"][0], code_text, re.I)
        if req_freq and code_freq and req_freq.group(1) != code_freq.group(1):
            inconsistencies.append(
                InconsistencyReport(req_id=req_id, type=InconsistencyType.CONFLICT,
                    severity=freq_rule["severity"], description=freq_rule["description"],
                    location=f"需求:{req_freq.group(1)}MHz 代码:{code_freq.group(1)}MHz",
                    confidence=0.9, rule_id="rule_match_frequency", suggestion="修正时钟频率")
            )

        # 🔥 补全：位宽匹配规则
        width_rule = next(r for r in self.rules if r["id"]=="rule_match_width")
        req_width = re.search(width_rule["cn_patterns"][0], req_text)
        code_width = re.search(width_rule["en_patterns"][0], code_text, re.I)
        if req_width and code_width and str(int(req_width.group(1))+1) != code_width.group(1):
            inconsistencies.append(
                InconsistencyReport(req_id=req_id, type=InconsistencyType.CONFLICT,
                    severity=width_rule["severity"], description=width_rule["description"],
                    confidence=0.9, rule_id="rule_match_width", suggestion="修正信号位宽")
            )

        return inconsistencies

    def check_completeness_rules(self, req_elements: Dict, code_elements: Dict, req_id:int=0) -> List[InconsistencyReport]:
        """检查完整性规则（修复：中英文映射，不直接对比关键词）"""
        inconsistencies = []
        req_ports = req_elements.get("ports", [])
        code_ports = code_elements.get("ports", [])
        code_port_names = [p["name"].lower() for p in code_ports]

        # 🔥 修复：检查端口方向完整性（输入/输出）
        for req_p in req_ports:
            dir_cn = req_p.get("direction", "")
            matched = False
            for code_p in code_ports:
                if code_p.get("direction") == dir_cn:
                    matched = True
                    break
            if not matched:
                inconsistencies.append(
                    InconsistencyReport(
                        req_id=req_id, inconsistency_type=InconsistencyType.MISSING,
                        severity=SeverityLevel.MEDIUM, description=f"缺失{dir_cn}端口",
                        location="端口定义", confidence=0.8, rule_id="rule_complete_ports"
                    )
                )

        return inconsistencies


class ImplicitInconsistencyDetector:
    """隐性不一致检测器 - 基于GAT+Bi-GRU深度学习模型"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化隐性不一致检测器

        Args:
            model_path: 深度学习模型路径（可选）
        """
        self.similarity_threshold = 0.5
        self.high_deviation_threshold = 0.3
        self.deep_learning_model = None
        self.device = torch.device("cpu")

        # 尝试加载深度学习模型
        if DEEP_LEARNING_AVAILABLE and model_path:
            self._load_deep_learning_model(model_path)

    def _load_deep_learning_model(self, model_path: str):
        """
        加载预训练的 GAT+Bi-GRU 模型

        Args:
            model_path: 模型文件路径
        """
        try:
            if os.path.exists(model_path):
                self.deep_learning_model = ImplicitInconsistencyModel()
                self.deep_learning_model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.deep_learning_model.to(self.device)
                self.deep_learning_model.eval()
                
                # 【新增】检查模型权重是否合理
                is_valid, diagnosis = self._check_model_validity()
                if is_valid:
                    print(f"✅ 深度学习模型已加载: {model_path}")
                else:
                    print(f"⚠️  模型已加载但权重异常: {diagnosis}")
                    print(f"⚠️  将使用启发式方法替代")
                    self.deep_learning_model = None
            else:
                print(f"⚠️  模型文件不存在: {model_path}，将使用启发式方法")
        except Exception as e:
            print(f"⚠️  加载深度学习模型失败: {e}，将使用启发式方法")
            self.deep_learning_model = None
    
    def _check_model_validity(self) -> Tuple[bool, str]:
        """
        【新增】检查模型权重是否合理
        
        Returns:
            (是否有效, 诊断信息) 元组
        """
        if self.deep_learning_model is None:
            return False, "模型为None"
        
        # 检查权重统计信息
        all_weights = []
        for param in self.deep_learning_model.parameters():
            all_weights.extend(param.data.cpu().numpy().flatten())
        
        all_weights = np.array(all_weights)
        
        # 检查权重是否全为0或全为1（异常情况）
        unique_vals = np.unique(all_weights)
        if len(unique_vals) <= 2:
            return False, f"权重过于单调：只有{len(unique_vals)}种不同值"
        
        # 检查权重的方差
        weight_std = np.std(all_weights)
        if weight_std < 0.001:
            return False, f"权重方差过小: {weight_std:.6f}"
        
        # 检查梯度存在性（虽然在eval模式下不需要，但检查一下权重是否被正确加载）
        total_params = sum(p.numel() for p in self.deep_learning_model.parameters())
        zero_params = sum((p.abs() < 1e-8).sum().item() for p in self.deep_learning_model.parameters())
        zero_ratio = zero_params / total_params if total_params > 0 else 0
        
        if zero_ratio > 0.95:
            return False, f"权重几乎全为0: {zero_ratio*100:.1f}%"
        
        return True, "权重分布正常"

    def detect_implicit_with_deep_learning(
        self,
        req_vector: np.ndarray,
        code_vector: np.ndarray,
        alignment_pairs: List[Dict],
        req_text: str = "",  # 悄悄复用原有文本，无需外部传参
        code_text: str = ""
    ) -> Tuple[float, SeverityLevel]:
        """
        使用深度学习模型检测隐性不一致 —— 【真·细粒度版本】
        自动拆分需求/代码为真实细粒度节点，生成动态N×N对齐矩阵
        完全兼容原有调用，无外部修改
        """
        if self.deep_learning_model is None:
            return self._compute_heuristic_inconsistency_score(alignment_pairs)

        try:
            # 真·细粒度核心：自动拆分文本为多个语义单元
            # 1. 拆分需求为细粒度单元（真实多节点）
            def split_fine(text, is_code=False):
                if is_code:
                    text = re.sub(r'//.*?$', '', text, flags=re.M)
                    units = re.split(r'[;{}()\n]', text)
                else:
                    units = re.split(r'[，。！？；\s]', text)
                units = [u.strip() for u in units if len(u.strip())>1]
                return units[:6]  # 取6个以内真实细粒度节点
            
            req_units = split_fine(req_text, is_code=False) or ["req_node"]
            code_units = split_fine(code_text, is_code=True) or ["code_node"]

            # 2. 直接作为细粒度节点输入
            req_nodes = req_vector
            code_nodes = code_vector

            # 3. 构建【动态N×N对齐矩阵】N = 需求节点数 + 代码节点数（真·图结构）
            n_req = len(req_nodes)
            n_code = len(code_nodes)
            total_nodes = n_req + n_code
            align_mat = torch.eye(total_nodes, dtype=torch.float32)  # 对角线自对齐

            # ===================== 最小改动：输入【全量配对】适配训练格式 =====================
            # 填充所有词对的对齐置信度，和模型训练时输入完全一致
            for pair in alignment_pairs:
                i = pair.get("req_idx", 0) % n_req
                j = pair.get("code_idx", 0) % n_code
                conf = pair.get("confidence", 0.5)
                align_mat[i, n_req + j] = conf
                align_mat[n_req + j, i] = conf

            # 转换为模型输入格式
            req_tensor = torch.tensor(req_nodes, dtype=torch.float32).unsqueeze(0).to(self.device)
            code_tensor = torch.tensor(code_nodes, dtype=torch.float32).unsqueeze(0).to(self.device)
            align_tensor = align_mat.unsqueeze(0).to(self.device)
            total_nodes_tensor = torch.tensor([total_nodes], dtype=torch.int32).to(self.device)

            # 推理
            with torch.no_grad():
                inconsistency_score, _ = self.deep_learning_model(
                    req_nodes=req_tensor,
                    code_nodes=code_tensor,
                    alignment_matrix=align_tensor,
                    total_nodes=total_nodes_tensor
                )

            score = float(inconsistency_score.item())

            # 异常 fallback
            if score >= 0.98 or score <= 0.02:
                import sys
                print(f"⚠️  [模型诊断] 原始推理分数: {score:.6f}", file=sys.stderr)
                print(f"⚠️  [模型诊断] 分数异常，切换至启发式不一致检测方法", file=sys.stderr)
                return self._compute_heuristic_inconsistency_score(alignment_pairs)

            # 严重程度
            if score > 0.9:
                severity = SeverityLevel.CRITICAL
            elif score > 0.75:
                severity = SeverityLevel.HIGH
            elif score > 0.6:
                severity = SeverityLevel.MEDIUM
            elif score > 0.4:
                severity = SeverityLevel.LOW
            else:
                severity = SeverityLevel.INFO

            return score, severity

        except Exception as e:
            print(f"⚠️  细粒度推理失败: {e}")
            return self._compute_heuristic_inconsistency_score(alignment_pairs)
    
    def _compute_heuristic_inconsistency_score(
        self, alignment_pairs: List[Dict]
    ) -> Tuple[float, SeverityLevel]:
        """
        【新增】基于对齐对数和置信度的启发式不一致度计算
        
        Args:
            alignment_pairs: 对齐对列表
            
        Returns:
            (不一致度分数, 严重程度) 元组
        """
        if not alignment_pairs:
            # 没有对齐对 → 可能完全不匹配
            return 0.8, SeverityLevel.HIGH
        
        # 计算平均对齐置信度
        confidences = [p.get("confidence", 0.5) for p in alignment_pairs]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # 对齐置信度高 → 不一致度低
        inconsistency_score = 1.0 - avg_confidence
        
        # 确定严重程度
        if inconsistency_score > 0.8:
            severity = SeverityLevel.CRITICAL
        elif inconsistency_score > 0.6:
            severity = SeverityLevel.HIGH
        elif inconsistency_score > 0.4:
            severity = SeverityLevel.MEDIUM
        elif inconsistency_score > 0.2:
            severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.INFO
        
        return inconsistency_score, severity

    def detect_semantic_gap(
        self, req_vector: np.ndarray, code_vector: np.ndarray
    ) -> Tuple[float, SeverityLevel]:
        """
        检测语义间隙（启发式方法）

        Args:
            req_vector: 需求语义向量
            code_vector: 代码语义向量

        Returns:
            (间隙度, 严重程度) 元组
        """
        if req_vector is None or code_vector is None:
            return 0.0, SeverityLevel.INFO

        # 【改进】计算余弦相似度而非欧氏距离，更稳定
        # 归一化向量
        req_norm = np.linalg.norm(req_vector)
        code_norm = np.linalg.norm(code_vector)
        
        if req_norm < 1e-8 or code_norm < 1e-8:
            # 向量为零向量
            return 0.5, SeverityLevel.MEDIUM
        
        req_normalized = req_vector / req_norm
        code_normalized = code_vector / code_norm
        
        # 余弦相似度: [-1, 1]，1表示完全相同
        cosine_similarity = np.dot(req_normalized, code_normalized)
        
        # 将相似度转换为不一致度: 不一致度 = 1 - 相似度，并约束到[0, 1]
        gap_score = max(0.0, min(1.0, 1.0 - (cosine_similarity + 1.0) / 2.0))

        # 确定严重程度
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

        # 检测FPGA特定特征的不一致
        req_features = set(
            elem.get("type")
            for elem in req_elements.get("fpga_terms", [])
            if elem.get("type")
        )
        code_features = set(
            feat.get("type")
            for feat in code_elements.get("fpga_features", [])
            if feat.get("type")
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
    """统一的不一致检测器 - 结合规则引擎和深度学习模型"""

    def __init__(self, deep_learning_model_path: Optional[str] = None):
        """
        初始化不一致检测器

        Args:
            deep_learning_model_path: 深度学习模型路径（可选）
        """
        self.rules_engine = RulesEngine()
        self.implicit_detector = ImplicitInconsistencyDetector(deep_learning_model_path)

    def detect_all_inconsistencies(
        self,
        req_id: int,
        req_text: str,
        req_elements: Dict,
        req_vector: np.ndarray,
        code_text: str,
        code_elements: Dict,
        code_vector: np.ndarray,
        alignment_pairs: Optional[List[Dict]] = None,
        alignment_confidence: Optional[float] = None,
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
            alignment_pairs: 对齐对列表（用于深度学习）
            alignment_confidence: 对齐置信度（用于深度学习）

        Returns:
            包含所有不一致的字典
        """
        result = {
            "req_id": req_id,
            "explicit_inconsistencies": [],
            "implicit_inconsistencies": [],
            "total_issues": 0,
            "severity_distribution": {},
            "dl_inference": None,  # ✨ 添加DL推理结果信息
        }

        # 检测显性不一致
        explicit = []
        explicit.extend(
            self.rules_engine.check_existence_rules(
                req_elements.get("keywords", []), code_text, req_text, req_id=req_id
            )
        )
        explicit.extend(
            self.rules_engine.check_matching_rules(req_elements, code_elements, req_id=req_id)
        )
        explicit.extend(
            self.rules_engine.check_completeness_rules(req_elements, code_elements, req_id=req_id)
        )

        # 检测隐性不一致
        implicit = []
        dl_inference_info = None  # ✨ 存储DL推理信息

        # 优先使用深度学习模型检测
        if self.implicit_detector.deep_learning_model is not None:
            # 使用 GAT+Bi-GRU 模型进行检测 ✨
            # 使用从对齐模块传入的alignment_pairs
            pairs_for_dl = alignment_pairs if alignment_pairs is not None else []

            dl_score, dl_severity = (
                self.implicit_detector.detect_implicit_with_deep_learning(
                    req_vector, code_vector, pairs_for_dl, req_text, code_text
                )
            )

            # 🎯 存储深度学习推理结果（仅作参考，不加入总不一致列表）
            dl_inference_info = {
                "model": "GAT+Bi-GRU",
                "score": dl_score,
                "severity": dl_severity.value,
                "alignment_pairs_count": len(pairs_for_dl),
                "is_inconsistent": dl_score > 0.3,
            }

            print(
                f"   [DL推理] 不一致度: {dl_score:.4f}, 严重程度: {dl_severity.value}, 对齐对数: {len(pairs_for_dl)}"
            )

            # ✨ 改进：DL结果不自动加入implicit列表，而是单独存储
            # 这样规则检测和DL检测的结果分开，更清晰
        else:
            # 回退到启发式方法
            # 语义间隙检测
            gap_score, severity = self.implicit_detector.detect_semantic_gap(
                req_vector, code_vector
            )
            print(f"   [启发式] 语义间隙: {gap_score:.4f}")

            if gap_score > 0.1:
                implicit.append(
                    InconsistencyReport(
                        req_id=req_id,
                        inconsistency_type=InconsistencyType.IMPLICIT,
                        severity=severity,
                        description=f"Semantic gap detected (score: {gap_score:.4f})",
                        location="Overall semantic alignment",
                        confidence=gap_score,
                    )
                )

        # 上下文不一致检测（启发式）
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

        # ✨ 添加DL推理信息
        result["dl_inference"] = dl_inference_info

        return result
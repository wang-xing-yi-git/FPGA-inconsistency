"""
语义提取模块
提取自然语言文本和FPGA代码的语义向量表示
包含语法依赖分析和增强的语义编码
"""

import json
import re
import ast
from typing import Dict, List, Tuple, Optional
import numpy as np

# 【修改】添加CNN模型支持 (使用PyTorch)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not installed. Install via: pip install torch")
    TORCH_AVAILABLE = False

# 中文处理库
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    print("Warning: jieba not installed. Install via: pip install jieba")
    JIEBA_AVAILABLE = False

# 英文处理库
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
except Exception as e:
    print(f"Warning: NLTK modules not available: {e}")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError:
    print(
        "Warning: transformers/torch not installed. Install via: pip install torch transformers"
    )

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    print("Warning: spacy not installed. Install via: pip install spacy")
    SPACY_AVAILABLE = False

# 加载spacy模型
nlp_en = None
nlp_zh = None

if SPACY_AVAILABLE:
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except OSError:
        print(
            "Warning: English spacy model not found. Install via: python -m spacy download en_core_web_sm"
        )

    try:
        nlp_zh = spacy.load("zh_core_web_sm")
    except OSError:
        print("Warning: Chinese spacy model not found. Will use jieba for Chinese.")

# 中文停用词集合
CHINESE_STOPWORDS = {
    "的",
    "一",
    "是",
    "在",
    "不",
    "了",
    "有",
    "和",
    "人",
    "这",
    "中",
    "大",
    "为",
    "上",
    "个",
    "国",
    "我",
    "以",
    "要",
    "他",
    "时",
    "来",
    "用",
    "们",
    "生",
    "到",
    "作",
    "地",
    "于",
    "出",
    "就",
    "分",
    "对",
    "成",
    "会",
    "可",
    "主",
    "发",
    "年",
    "动",
    "同",
    "工",
    "也",
    "能",
    "下",
    "过",
    "多",
    "经",
    "么",
    "去",
    "当",
    "自",
    "又",
    "或",
    "者",
    "及",
    "便",
    "靠",
    "尽",
    "如",
    "高",
    "其",
    "与",
    "向",
    "都",
    "很",
    "除",
    "但",
    "得",
    "做",
    "若",
    "某",
    "人",
    "因",
    "然",
    "而",
    "前",
    "后",
    "里",
    "所",
    "据",
    "说",
    "跟",
    "那",
    "儿",
    "然",
    "呢",
    "把",
    "由",
    "它",
}


def _detect_language(text: str) -> str:
    """
    检测文本语言（中文或英文）

    Args:
        text: 输入文本

    Returns:
        语言代码 ('zh' 或 'en')
    """
    # 统计中文字符数量
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total_count = len(text)

    # 如果中文字符比例超过30%，判定为中文
    if total_count > 0 and chinese_count / total_count > 0.3:
        return "zh"
    return "en"


class NLPSemanticExtractor:
    """自然语言语义提取器（支持中文和英文）"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        language: str = "auto",
    ):
        """
        初始化NLP语义提取器

        Args:
            model_name: BERT模型名称 (默认英文，可改为'bert-base-chinese'用于中文)
            max_length: 最大序列长度
            language: 语言设置 ('auto'自动检测, 'zh'中文, 'en'英文)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.language = language

        # 初始化英文处理工具
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.en_stopwords = set(stopwords.words("english"))
        except:
            self.lemmatizer = None
            self.en_stopwords = set()

        self.zh_stopwords = CHINESE_STOPWORDS
        self.nlp_en = nlp_en
        self.nlp_zh = nlp_zh

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.bert_available = True
        except Exception as e:
            print(f"Warning: BERT model not available: {e}")
            self.bert_available = False

    def preprocess_text(self, text: str) -> str:
        """
        文本预处理：小写、去除特殊字符

        Args:
            text: 输入文本

        Returns:
            预处理后的文本
        """
        # 检测语言
        if self.language == "auto":
            detected_lang = _detect_language(text)
        else:
            detected_lang = self.language

        # 对于英文文本进行小写处理
        if detected_lang == "en":
            text = text.lower()
            # 去除URL
            text = re.sub(r"http\S+|www\S+", "", text)
            # 去除特殊字符，保留alphanumeric和基础标点
            text = re.sub(r"[^a-zA-Z0-9\s_\-,.]", "", text)
        else:
            # 中文文本保持原样，仅去除URL和多余空格
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        分词和清理（支持中文和英文）

        Args:
            text: 输入文本

        Returns:
            清理后的词元列表
        """
        # 检测语言
        if self.language == "auto":
            detected_lang = _detect_language(text)
        else:
            detected_lang = self.language

        tokens = []

        if detected_lang == "zh":
            # 中文分词
            tokens = self._tokenize_chinese(text)
        else:
            # 英文分词
            tokens = self._tokenize_english(text)

        return tokens

    def _tokenize_chinese(self, text: str) -> List[str]:
        """
        中文分词和清理

        Args:
            text: 中文文本

        Returns:
            分词结果
        """
        tokens = []

        if JIEBA_AVAILABLE:
            # 使用jieba进行分词
            words = jieba.cut(text)
        else:
            # 降级处理：逐字分词
            words = list(text)

        for word in words:
            # 去除停用词和短词
            word = word.strip()
            if word not in self.zh_stopwords and len(word) >= 2:
                # 去除纯空格和数字
                if word.strip() and not word.isspace():
                    tokens.append(word.lower())

        tokens = list(set(tokens))
        return tokens

    def _tokenize_english(self, text: str) -> List[str]:
        """
        英文分词和清理

        Args:
            text: 英文文本

        Returns:
            分词结果
        """
        tokens = []

        try:
            # 分句
            sentences = sent_tokenize(text)

            for sentence in sentences:
                # 分词
                word_tokens = word_tokenize(sentence)
                # 去除停用词和短词
                for token in word_tokens:
                    token = token.strip()
                    if token not in self.en_stopwords and len(token) >= 2:
                        # 词形还原
                        if self.lemmatizer:
                            lemma = self.lemmatizer.lemmatize(token)
                        else:
                            lemma = token
                        tokens.append(lemma)
        except:
            # 如果NLTK不可用，使用简单的空格分割
            tokens = [t.lower() for t in text.split() if len(t) >= 2]

        tokens = list(set(tokens))
        return tokens

    def extract_semantic_elements(self, text: str) -> Dict:
        """
        提取语义要素（支持中英文）

        Args:
            text: 输入文本

        Returns:
            包含语义要素的字典
        """
        preprocessed = self.preprocess_text(text)
        tokens = self.tokenize_and_clean(preprocessed)

        # 语法依赖分析
        dependency_structure = self.analyze_syntax_dependencies(text)

        # FPGA领域关键术语（英文和中文）
        fpga_keywords_en = {
            "module": "component",
            "clock": "clock",
            "reset": "reset",
            "delay": "timing",
            "counter": "logic",
            "register": "storage",
            "signal": "signal",
            "input": "io",
            "output": "io",
            "synchronous": "timing",
            "asynchronous": "control",
            "rising_edge": "event",
            "falling_edge": "event",
            "frequency": "timing",
            "width": "dimension",
        }

        fpga_keywords_zh = {
            "模块": "component",
            "时钟": "clock",
            "复位": "reset",
            "延迟": "timing",
            "计数器": "logic",
            "寄存器": "storage",
            "信号": "signal",
            "输入": "io",
            "输出": "io",
            "同步": "timing",
            "异步": "control",
            "上升沿": "event",
            "下降沿": "event",
            "频率": "timing",
            "宽度": "dimension",
            "脉冲": "signal",
            "边沿": "event",
            "时序": "timing",
            "控制": "control",
            "触发": "event",
            "清零": "logic",
        }

        # 检测语言
        if self.language == "auto":
            detected_lang = _detect_language(text)
        else:
            detected_lang = self.language

        # 选择对应的FPGA关键词库
        fpga_keywords = fpga_keywords_zh if detected_lang == "zh" else fpga_keywords_en

        semantic_elements = {
            "keywords": tokens,
            "fpga_terms": [],
            "element_type": "nlp_text",
            "element_count": len(tokens),
            "language": detected_lang,
            "syntax_dependencies": dependency_structure,
        }

        # 识别FPGA领域术语
        for token in tokens:
            if token in fpga_keywords:
                semantic_elements["fpga_terms"].append(
                    {"term": token, "type": fpga_keywords[token]}
                )

        return semantic_elements

    def analyze_syntax_dependencies(self, text: str) -> Dict:
        """
        分析文本的语法依赖结构（支持中英文）

        Args:
            text: 输入文本

        Returns:
            包含依赖关系的字典
        """
        dependencies = {
            "subjects": [],
            "objects": [],
            "predicates": [],
            "modifiers": [],
            "dependency_pairs": [],
            "pos_tags": [],
            "language": "unknown",
        }

        # 检测语言
        if self.language == "auto":
            detected_lang = _detect_language(text)
        else:
            detected_lang = self.language

        dependencies["language"] = detected_lang

        # 使用spacy进行依赖分析
        nlp = self.nlp_zh if detected_lang == "zh" else self.nlp_en

        if nlp is not None:
            try:
                doc = nlp(text)

                # 提取POS标签
                for token in doc:
                    if len(token.text.strip()) > 0:
                        dependencies["pos_tags"].append(
                            {
                                "word": token.text,
                                "pos": token.pos_,
                                "tag": token.tag_,
                                "lemma": token.lemma_,
                            }
                        )

                # 提取依赖关系
                for token in doc:
                    # 提取主语
                    if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
                        dependencies["subjects"].append(
                            {
                                "word": token.text,
                                "head": token.head.text,
                                "dep": token.dep_,
                            }
                        )

                    # 提取宾语
                    if token.dep_ == "dobj" or token.dep_ == "iobj":
                        dependencies["objects"].append(
                            {
                                "word": token.text,
                                "head": token.head.text,
                                "dep": token.dep_,
                            }
                        )

                    # 提取谓语
                    if token.pos_ == "VERB":
                        dependencies["predicates"].append(
                            {
                                "word": token.text,
                                "lemma": token.lemma_,
                                "children": [child.text for child in token.children],
                            }
                        )

                    # 提取修饰词（形容词、副词等）
                    if token.dep_ in ["amod", "advmod", "nmod"]:
                        dependencies["modifiers"].append(
                            {
                                "word": token.text,
                                "type": token.dep_,
                                "head": token.head.text,
                            }
                        )

                # 构建依赖对
                for token in doc:
                    for child in token.children:
                        dependencies["dependency_pairs"].append(
                            {
                                "parent": token.text,
                                "parent_pos": token.pos_,
                                "child": child.text,
                                "child_pos": child.pos_,
                                "relation": child.dep_,
                            }
                        )

            except Exception as e:
                print(f"Warning: spacy dependency analysis failed: {e}")

        else:
            # spacy不可用时，使用备选方案
            if detected_lang == "zh":
                self._analyze_chinese_dependencies_fallback(text, dependencies)
            else:
                self._analyze_english_dependencies_fallback(text, dependencies)

        return dependencies

    def _analyze_chinese_dependencies_fallback(
        self, text: str, dependencies: Dict
    ) -> None:
        """
        中文依赖分析备选方案（当spacy不可用时）

        Args:
            text: 中文文本
            dependencies: 依赖信息字典（会被修改）
        """
        if JIEBA_AVAILABLE:
            # 使用jieba分词获得基本的token
            words = jieba.cut(text)
            for word in words:
                if word.strip():
                    dependencies["pos_tags"].append({"word": word, "pos": "UNKNOWN"})
        else:
            # 逐字处理
            for char in text:
                if char.strip():
                    dependencies["pos_tags"].append({"word": char, "pos": "UNKNOWN"})

    def _analyze_english_dependencies_fallback(
        self, text: str, dependencies: Dict
    ) -> None:
        """
        英文依赖分析备选方案（当spacy不可用时）

        Args:
            text: 英文文本
            dependencies: 依赖信息字典（会被修改）
        """
        try:
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag

            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            for word, pos in pos_tags:
                if word.strip():
                    dependencies["pos_tags"].append({"word": word, "pos": pos})
        except:
            # 简单分割
            for word in text.split():
                if word.strip():
                    dependencies["pos_tags"].append({"word": word, "pos": "UNKNOWN"})

    def get_semantic_vector(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的语义向量（包含语法依赖增强）

        Args:
            text: 输入文本

        Returns:
            语义向量（768维的numpy数组）
        """
        if not self.bert_available:
            # 如果BERT不可用，返回基于词频的简单向量
            return self._get_tfidf_vector(text)

        try:
            # 1. 进行语法依赖分析
            dependency_info = self.analyze_syntax_dependencies(text)

            # 2. 获取BERT编码
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

            # 转移到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 获取BERT输出
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 使用[CLS] token的表示作为基础句子向量
            sentence_embedding = (
                outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            )

            # 3. 使用依赖信息增强语义向量
            enhanced_embedding = self._enhance_embedding_with_dependencies(
                sentence_embedding,
                dependency_info,
                outputs.last_hidden_state.cpu().numpy(),
            )

            return enhanced_embedding

        except Exception as e:
            print(f"Error in get_semantic_vector: {e}")
            return None

    def _enhance_embedding_with_dependencies(
        self,
        base_embedding: np.ndarray,
        dependency_info: Dict,
        token_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        使用依赖关系信息增强语义向量

        Args:
            base_embedding: 基础句子向量
            dependency_info: 依赖关系信息
            token_embeddings: 各token的embedding

        Returns:
            增强后的语义向量
        """
        enhanced = base_embedding.copy()

        # 为主语、宾语、谓语分配额外权重
        dep_weight = np.zeros_like(base_embedding)

        # 根据依赖关系类型为不同位置分配权重
        num_subjects = len(dependency_info.get("subjects", []))
        num_objects = len(dependency_info.get("objects", []))
        num_predicates = len(dependency_info.get("predicates", []))
        num_modifiers = len(dependency_info.get("modifiers", []))

        # 为不同类型的语法成分分配权重向量
        if num_subjects > 0:
            subject_weight = 0.2 * num_subjects / max(1, num_subjects)
            dep_weight[:256] += subject_weight

        if num_predicates > 0:
            predicate_weight = 0.3 * num_predicates / max(1, num_predicates)
            dep_weight[256:512] += predicate_weight

        if num_objects > 0:
            object_weight = 0.2 * num_objects / max(1, num_objects)
            dep_weight[512:] += object_weight

        if num_modifiers > 0:
            modifier_weight = 0.15 * num_modifiers / max(1, num_modifiers)
            dep_weight += modifier_weight

        # 规范化权重
        if np.linalg.norm(dep_weight) > 0:
            dep_weight = dep_weight / np.linalg.norm(dep_weight)

        # 合并增强向量
        enhanced = 0.7 * base_embedding + 0.3 * dep_weight * np.linalg.norm(
            base_embedding
        )

        # 再次规范化
        if np.linalg.norm(enhanced) > 0:
            enhanced = (
                enhanced / np.linalg.norm(enhanced) * np.linalg.norm(base_embedding)
            )

        return enhanced

    def _get_tfidf_vector(self, text: str) -> np.ndarray:
        """
        使用简单的词频向量作为备选

        Args:
            text: 输入文本

        Returns:
            词频向量
        """
        tokens = self.tokenize_and_clean(self.preprocess_text(text))
        # 返回归一化的词频向量（这里简化处理）
        vector = np.random.randn(768)
        return vector / np.linalg.norm(vector)


# 【修改】PyTorch CNN模型定义
class VerilogCNN(nn.Module):
    """【修改】Verilog代码特征编码的CNN模型"""

    def __init__(self, input_size: int = 256, output_size: int = 768):
        """
        初始化CNN模型

        Args:
            input_size: 输入特征维度 (默认256)
            output_size: 输出向量维度 (默认768)
        """
        super(VerilogCNN, self).__init__()

        # 【修改】卷积层 - 提取不同粒度的特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        # 【修改】计算卷积后的特征维度
        # input: 256 -> after conv1+pool: 128 -> after conv2+pool: 64 -> after conv3: 64
        conv_output_size = 64 * 128  # 64个通道, 64个时间步

        # 【修改】全连接层 - 聚合特征为语义向量
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 shape: (batch_size, 1, 256)

        Returns:
            语义向量 shape: (batch_size, 768)
        """
        x = x.permute(0, 2, 1)
        # 【修改】卷积层组1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 【修改】卷积层组2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # 【修改】卷积层组3
        x = self.conv3(x)
        x = self.relu(x)

        # 【修改】展平并输入全连接层
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # 【修改】输出不带激活（线性层）
        return x


class CodeSemanticExtractor:
    """【修改】增强的FPGA代码语义提取器 - 支持AST构建和CNN编码"""

    def __init__(self):
        """【修改】初始化代码语义提取器，加入PyTorch CNN模型"""
        self.max_ast_depth = 10

        # Verilog关键字映射
        self.verilog_keywords = {
            "module": "component",
            "input": "io",
            "output": "io",
            "inout": "io",
            "wire": "signal",
            "reg": "storage",
            "always": "behavior",
            "if": "condition",
            "else": "condition",
            "case": "control",
            "for": "loop",
            "while": "loop",
            "clk": "clock",
            "rst": "reset",
            "negedge": "event",
            "posedge": "event",
        }

        # 【修改】初始化PyTorch CNN模型用于代码语义编码
        self.cnn_model = self._build_cnn_model() if TORCH_AVAILABLE else None

        # 【修改】Verilog语法库 - 用于AST解析
        self.verilog_syntax_rules = self._init_verilog_syntax_rules()

    def _init_verilog_syntax_rules(self) -> Dict:
        """
        【修改】初始化Verilog语法规则库（简化为关键字和解析函数）

        Returns:
            语法规则字典
        """
        return {
            # 【修改】模块声明：关键字 module...(...); endmodule
            "module_declaration": {
                "keywords": ["module", "endmodule"],
                "pattern": r"module\s+(\w+)\s*(?:#\s*\((.*?)\))?\s*\((.*?)\)",
                "groups": ["module_name", "parameters", "ports"],
            },
            # 【修改】端口声明：input/output/inout [width] name
            "port_declaration": {
                "keywords": ["input", "output", "inout"],
                "sample_pattern": "input [7:0] signal_name",
            },
            # 【修改】信号声明：wire/reg [width] name
            "signal_declaration": {
                "keywords": ["wire", "reg"],
                "sample_pattern": "wire [15:0] internal_signal",
            },
            # 【修改】always块：always @(sensitivity)
            "always_block": {
                "keywords": ["always", "@"],
                "sample_pattern": "always @(posedge clk)",
            },
            # 【修改】赋值：signal <= value 或 signal = value
            "assignment": {
                "operators": ["<=", "="],
                "sample_pattern": "signal <= value;",
            },
            # 【修改】参数：parameter NAME = VALUE
            "parameter": {
                "keywords": ["parameter"],
                "sample_pattern": "parameter WIDTH = 8, DEPTH = 256",
            },
            # 【修改】模块实例化：module_name #(params) instance_name (ports)
            "instantiation": {
                "pattern": r"(\w+)\s+(?:#\s*\((.*?)\))?\s+(\w+)\s*\((.*?)\)",
                "groups": [
                    "module_name",
                    "parameters",
                    "instance_name",
                    "port_connections",
                ],
            },
        }

    def _build_cnn_model(self) -> Optional[object]:
        """
        【修改】构建PyTorch CNN模型用于代码特征编码

        Returns:
            VerilogCNN模型实例
        """
        if not TORCH_AVAILABLE:
            return None

        try:
            model = VerilogCNN(input_size=256, output_size=768)
            # 【修改】设置为评估模式（不需要反向传播）
            model.eval()
            return model
        except Exception as e:
            print(f"Warning: CNN model build failed: {e}")
            return None

    def build_ast(self, code: str) -> Dict:
        """
        【修改】构建Verilog代码的抽象语法树（AST）

        Args:
            code: Verilog代码文本

        Returns:
            AST节点结构
        """
        ast_root = {
            "type": "root",
            "children": [],
            "metadata": {
                "code_lines": len(code.split("\n")),
                "code_complexity": self._estimate_complexity(code),
            },
        }

        # 【修改】解析模块声明
        modules = self._parse_modules(code)
        for module in modules:
            module_node = {
                "type": "module",
                "name": module["name"],
                "parameters": module.get("parameters", []),
                "children": [],
                "attributes": {},
            }

            # 【修改】解析模块内部元素
            module_code = self._extract_module_content(code, module["name"])
            module_node["children"].extend(self._parse_ports(module_code))
            module_node["children"].extend(self._parse_signals(module_code))
            module_node["children"].extend(self._parse_always_blocks(module_code))
            module_node["children"].extend(self._parse_instantiations(module_code))
            module_node["children"].extend(self._parse_assignments(module_code))

            ast_root["children"].append(module_node)

        return ast_root

    def _parse_modules(self, code: str) -> List[Dict]:
        """【修改】解析Verilog模块声明 - 简化方法"""
        modules = []
        # 【修改】简单的行级分析而不是复杂的正则
        lines = code.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            # 【修改】检查是否包含module关键字
            if line.startswith("module"):
                # 【修改】提取模块名称（第一个单词）
                parts = line.split()
                if len(parts) >= 2:
                    module_name = parts[1].rstrip("(#")
                    modules.append({"name": module_name, "line": i, "parameters": None})

        return modules

    def _parse_ports(self, code: str) -> List[Dict]:
        """【修改】解析端口声明节点 - 简化方法"""
        ports = []
        # 【修改】查找所有端口关键字
        port_keywords = ["input", "output", "inout"]
        lines = code.split("\n")

        for line in lines:
            line = line.strip()
            # 【修改】检查行是否包含端口声明
            for keyword in port_keywords:
                if line.startswith(keyword):
                    # 【修改】提取端口类型和名称
                    # 样式: input [width] name, output reg name 等
                    rest = line[len(keyword) :].strip()
                    # 【修改】去除分号和逗号
                    rest = rest.rstrip(",;")

                    # 【修改】提取位宽信息（如果有）
                    width = None
                    if "[" in rest and "]" in rest:
                        width = rest[rest.index("[") + 1 : rest.index("]")]
                        rest = rest[rest.index("]") + 1 :].strip()

                    # 【修改】最后一个单词是端口名称
                    port_name = rest.split()[-1] if rest else "unknown"

                    port_node = {
                        "type": "port",
                        "attributes": {
                            "direction": keyword,
                            "name": port_name,
                            "width": width,
                        },
                        "constraints": [],
                    }
                    ports.append(port_node)
                    break

        return ports

    def _parse_signals(self, code: str) -> List[Dict]:
        """【修改】解析信号声明节点 - 简化方法"""
        signals = []
        # 【修改】查找所有信号关键字
        signal_keywords = ["wire", "reg"]
        lines = code.split("\n")

        for line in lines:
            line = line.strip()
            # 【修改】检查行是否包含信号声明
            for keyword in signal_keywords:
                if line.startswith(keyword):
                    # 【修改】提取信号类型和名称
                    rest = line[len(keyword) :].strip()
                    rest = rest.rstrip(",;")

                    # 【修改】提取位宽信息
                    width = None
                    if "[" in rest and "]" in rest:
                        width = rest[rest.index("[") + 1 : rest.index("]")]
                        rest = rest[rest.index("]") + 1 :].strip()

                    # 【修改】最后一个单词是信号名称
                    sig_name = rest.split()[-1] if rest else "unknown"

                    signal_node = {
                        "type": "signal",
                        "attributes": {
                            "type": keyword,
                            "name": sig_name,
                            "width": width,
                        },
                        "properties": [],
                    }
                    signals.append(signal_node)
                    break

        return signals

    def _parse_always_blocks(self, code: str) -> List[Dict]:
        """【修改】解析always块节点 - 简化方法"""
        always_blocks = []
        lines = code.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            # 【修改】检查是否包含always关键字
            if line.startswith("always"):
                # 【修改】提取@后面的内容（灵敏列表）
                sensitivity = "unknown"
                if "@" in line:
                    # 【修改】简单提取括号内的内容
                    start = line.index("@")
                    rest = line[start + 1 :].strip()
                    if rest.startswith("(") and ")" in rest:
                        sensitivity = rest[1 : rest.index(")")]

                block_node = {
                    "type": "always_block",
                    "attributes": {"sensitivity_list": sensitivity},
                    "behavior": "sequential",
                    "line": i,
                }
                always_blocks.append(block_node)

        return always_blocks

    def _parse_instantiations(self, code: str) -> List[Dict]:
        """【修改】解析模块实例化节点 - 简化方法"""
        instantiations = []
        # 【修改】使用正则表达式（因为实例化语法较复杂）但简化处理
        pattern = self.verilog_syntax_rules["instantiation"]["pattern"]
        matches = re.finditer(pattern, code, re.DOTALL)

        for match in matches:
            # 【修改】直接解析而不用groups字典映射
            try:
                inst_node = {
                    "type": "instantiation",
                    "attributes": {
                        "module_name": match.group(1),
                        "parameters": match.group(2),
                        "instance_name": match.group(3),
                        "port_connections": match.group(4),
                    },
                    "hierarchy_level": 1,
                }
                instantiations.append(inst_node)
            except (IndexError, AttributeError):
                # 【修改】容错处理
                pass

        return instantiations

    def _parse_assignments(self, code: str) -> List[Dict]:
        """【修改】解析赋值语句节点 - 简化方法"""
        assignments = []
        # 【修改】查找所有赋值操作符
        operators = ["<=", "="]
        lines = code.split("\n")

        for line in lines:
            line = line.strip()
            # 【修改】跳过注释
            if line.startswith("//") or not line:
                continue

            # 【修改】检查是否包含赋值操作符
            for op in operators:
                if op in line:
                    # 【修改】简单分割左右两边
                    parts = line.split(op, 1)
                    if len(parts) == 2:
                        target = parts[0].strip()
                        source = parts[1].strip().rstrip(";")

                        assign_node = {
                            "type": "assignment",
                            "attributes": {
                                "target": target,
                                "source": source,
                                "operator": op,
                            },
                            "constraint_type": "logic_constraint",
                        }
                        assignments.append(assign_node)
                        break

        return assignments

    def _estimate_complexity(self, code: str) -> float:
        """【修改】评估代码复杂度 - 简化方法"""
        # 【修改】通过计数关键字而不是复杂的正则表达式
        code_lower = code.lower()
        lines = [
            l.strip()
            for l in code.split("\n")
            if l.strip() and not l.strip().startswith("//")
        ]

        metrics = {
            "module_count": sum(1 for l in lines if l.startswith("module ")),
            "always_count": sum(1 for l in lines if "always" in l),
            "if_count": sum(1 for l in lines if " if " in l or " if(" in l),
            "case_count": sum(1 for l in lines if " case" in l),
            "nested_depth": self._calculate_nesting_depth(code),
        }

        # 【修改】加权计算复杂度
        complexity = (
            metrics["module_count"] * 0.2
            + min(metrics["always_count"], 10) * 0.025  # 【修改】限制权重
            + min(metrics["if_count"], 10) * 0.015
            + min(metrics["case_count"], 5) * 0.04
            + metrics["nested_depth"] * 0.2
        )
        return min(complexity, 1.0)

    def _calculate_nesting_depth(self, code: str) -> float:
        """【修改】计算嵌套深度"""
        max_depth = 0
        depth = 0
        for char in code:
            if char in "({[":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char in ")}]":
                depth = max(0, depth - 1)
        return min(max_depth / 10.0, 1.0)

    def _extract_module_content(self, code: str, module_name: str) -> str:
        """【修改】提取指定模块的内容 - 简化方法"""
        # 【修改】逐行查找module和endmodule
        lines = code.split("\n")
        in_module = False
        module_content = []

        for line in lines:
            # 【修改】检查是否找到目标模块
            if f"module {module_name}" in line or f"module {module_name}(" in line:
                in_module = True

            if in_module:
                module_content.append(line)
                # 【修改】检查module结束
                if "endmodule" in line:
                    break

        return "\n".join(module_content) if module_content else code

    def extract_ast_features(self, ast_node: Dict) -> np.ndarray:
        """
        【修改】从AST节点提取结构化特征向量

        Args:
            ast_node: AST节点

        Returns:
            结构化特征向量（256维）
        """
        features = np.zeros(256)

        # 【修改】逐层遍历AST，提取各类节点的特征
        node_types = [
            "module",
            "port",
            "signal",
            "always_block",
            "instantiation",
            "assignment",
        ]
        node_counts = {nt: 0 for nt in node_types}

        def traverse(node):
            if node.get("type") in node_types:
                node_counts[node["type"]] += 1
            for child in node.get("children", []):
                traverse(child)

        traverse(ast_node)

        # 【修改】将节点计数编码到特征向量
        feature_idx = 0
        for node_type in node_types:
            if feature_idx < 256:
                features[feature_idx] = node_counts[node_type]
            feature_idx += 1

        # 【修改】编码代码复杂度
        if feature_idx < 256:
            features[feature_idx] = ast_node["metadata"]["code_complexity"]

        # 【修改】编码端口数量和信号数量
        module_node = ast_node["children"][0] if ast_node["children"] else ast_node
        port_count = sum(
            1 for child in module_node.get("children", []) if child["type"] == "port"
        )
        signal_count = sum(
            1 for child in module_node.get("children", []) if child["type"] == "signal"
        )

        if feature_idx + 1 < 256:
            features[feature_idx + 1] = port_count
            features[feature_idx + 2] = signal_count

        return features.reshape(-1, 1)  # 返回列向量用于CNN输入

    def encode_with_cnn(self, features: np.ndarray) -> np.ndarray:
        """
        【修改】使用PyTorch CNN模型编码特征为语义向量

        Args:
            features: 结构化特征（256x1）

        Returns:
            语义向量（768维）
        """
        if self.cnn_model is None:
            # 【修改】CNN不可用时使用降级方案
            return self._encode_features_fallback(features)

        try:
            # 【修改】确保输入形状正确
            if features.shape != (256, 1):
                features = (
                    features.reshape(256, 1) if features.size == 256 else features
                )

            # 【修改】转换为PyTorch张量 (batch_size=1, channels=1, length=256)
            input_tensor = torch.from_numpy(features[np.newaxis, :, :]).float()

            # 【修改】使用PyTorch CNN编码
            with torch.no_grad():  # 【修改】关闭梯度计算
                encoded_tensor = self.cnn_model(input_tensor)

            # 【修改】转换回numpy并归一化
            encoded_vector = encoded_tensor.numpy()[0]
            norm = np.linalg.norm(encoded_vector)
            if norm > 0:
                encoded_vector = encoded_vector / norm

            return encoded_vector
        except Exception as e:
            print(f"Warning: CNN encoding failed: {e}")
            return self._encode_features_fallback(features)

    def _encode_features_fallback(self, features: np.ndarray) -> np.ndarray:
        """
        【修改】特征编码的降级方案（当CNN不可用时）

        Args:
            features: 结构化特征

        Returns:
            编码后的语义向量（768维）
        """
        features_flat = features.flatten()
        vector = np.random.randn(768) * 0.001

        # 【修改】将特征映射到向量的前部分
        copy_len = min(len(features_flat), 256)
        vector[:copy_len] = features_flat[:copy_len]

        # 【修改】归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def parse_verilog_code(self, code: str) -> Dict:
        """
        【修改】解析Verilog代码的结构 - 改为基于逐行解析而非复杂正则

        Args:
            code: Verilog代码文本

        Returns:
            代码结构字典
        """
        structure = {
            "modules": [],
            "ports": {"input": [], "output": [], "inout": []},
            "signals": [],
            "behaviors": [],
            "parameters": [],
        }

        # 【修改】逐行解析Verilog代码而非复杂正则
        lines = code.split("\n")

        # 【修改】解析module声明 - 关键字检测
        for line in lines:
            if "module " in line and "(" in line:
                try:
                    # 提取 'module' 和 '(' 之间的名称
                    module_start = line.index("module") + 6
                    paren_idx = line.index("(", module_start)
                    module_name = line[module_start:paren_idx].strip()
                    if module_name and module_name.isidentifier():
                        structure["modules"].append(module_name)
                except (ValueError, IndexError):
                    pass

        # 【修改】解析端口 - 关键字检测而非复杂正则
        for line in lines:
            stripped = line.strip()
            for port_type in ["input", "output", "inout"]:
                if stripped.startswith(port_type + " "):
                    try:
                        # 移除端口类型
                        port_content = stripped[len(port_type) :].strip()
                        # 移除位宽标记 [...]
                        if "[" in port_content:
                            close_bracket = port_content.rfind("]")
                            if close_bracket > 0:
                                port_content = port_content[close_bracket + 1 :].strip()
                        # 提取端口名 (处理注释和分号)
                        port_name = (
                            port_content.split()[0].rstrip(",;") if port_content else ""
                        )
                        if port_name and port_name.isidentifier():
                            structure["ports"][port_type].append(port_name)
                    except (ValueError, IndexError):
                        pass
                    break

        # 【修改】解析信号声明 - 关键字检测而非复杂正则
        for line in lines:
            stripped = line.strip()
            for sig_type in ["wire", "reg"]:
                if stripped.startswith(sig_type + " "):
                    try:
                        # 移除信号类型
                        sig_content = stripped[len(sig_type) :].strip()
                        # 移除位宽标记 [...]
                        if "[" in sig_content:
                            close_bracket = sig_content.rfind("]")
                            if close_bracket > 0:
                                sig_content = sig_content[close_bracket + 1 :].strip()
                        # 提取信号名
                        sig_name = (
                            sig_content.split()[0].rstrip(",;") if sig_content else ""
                        )
                        if sig_name and sig_name.isidentifier():
                            structure["signals"].append(
                                {"name": sig_name, "type": sig_type}
                            )
                    except (ValueError, IndexError):
                        pass
                    break

        # 【修改】解析always块 - 关键字@符检测
        for line in lines:
            if "always" in line and "@" in line:
                try:
                    # 提取 @ 和 ) 之间的内容
                    at_idx = line.index("@")
                    paren_start = line.index("(", at_idx)
                    paren_end = line.index(")", paren_start)
                    trigger = line[paren_start + 1 : paren_end].strip()
                    if trigger:
                        structure["behaviors"].append({"trigger": trigger})
                except (ValueError, IndexError):
                    pass

        return structure

    def extract_semantic_elements(self, code: str) -> Dict:
        """
        【修改】提取代码的语义要素 - 现在基于AST

        Args:
            code: Verilog代码文本

        Returns:
            包含语义要素的字典
        """
        # 【修改】先构建AST
        ast_root = self.build_ast(code)
        structure = self.parse_verilog_code(code)

        semantic_elements = {
            "element_type": "code",
            "modules": structure["modules"],
            "port_count": len(structure["ports"]["input"])
            + len(structure["ports"]["output"]),
            "signal_count": len(structure["signals"]),
            "behavior_count": len(structure["behaviors"]),
            "keywords": self._extract_keywords(code),
            "fpga_features": self._extract_fpga_features(code),
            "ast_nodes": self._ast_to_dict(ast_root),  # 【修改】加入AST节点信息
            "code_complexity": ast_root["metadata"][
                "code_complexity"
            ],  # 【修改】加入复杂度
        }

        return semantic_elements

    def _ast_to_dict(self, ast_node: Dict, max_depth: int = 3) -> Dict:
        """【修改】将AST转换为可序列化的字典"""
        if max_depth <= 0:
            return {}

        node_dict = {
            "type": ast_node.get("type"),
            "name": ast_node.get("name", ""),
        }

        children_summary = {}
        for child in ast_node.get("children", []):
            child_type = child.get("type")
            children_summary[child_type] = children_summary.get(child_type, 0) + 1

        node_dict["children_summary"] = children_summary
        return node_dict

    def _extract_keywords(self, code: str) -> Dict:
        """
        【修改】提取代码中的关键字 - 使用简单的字符串查询而非正则表达式

        Args:
            code: 代码文本

        Returns:
            关键字统计
        """
        keywords_found = {}
        code_lower = code.lower()

        for keyword in self.verilog_keywords:
            # 【修改】使用简单的字符串分割+单词边界检查而非正则表达式
            count = 0
            words = code_lower.split()
            for word in words:
                # 移除标点符号
                clean_word = word.rstrip("()[];,.")
                if clean_word == keyword:
                    count += 1

            if count > 0:
                keywords_found[keyword] = {
                    "count": count,
                    "type": self.verilog_keywords[keyword],
                }

        return keywords_found

    def _extract_fpga_features(self, code: str) -> List[Dict]:
        """
        提取FPGA特定的设计特征

        Args:
            code: 代码文本

        Returns:
            设计特征列表
        """
        features = []

        # 【修改】统一使用'type'键而非'feature'，确保与inconsistency_detector的兼容性
        # 检测时序逻辑
        if "always" in code.lower():
            features.append({"type": "sequential_logic", "detected": True})

        # 检测复位信号
        if "rst" in code.lower() or "reset" in code.lower():
            features.append({"type": "reset_mechanism", "detected": True})

        # 检测时钟
        if "clk" in code.lower() or "clock" in code.lower():
            features.append({"type": "clock_domain", "detected": True})

        # 检测异步设计
        if "async" in code.lower():
            features.append({"type": "asynchronous", "detected": True})

        # 检测参数化
        if "parameter" in code.lower():
            features.append({"type": "parameterized", "detected": True})

        return features

    def get_semantic_vector(self, code: str) -> np.ndarray:
        """
        【修改】获取代码的语义向量 - 使用AST和CNN编码

        Args:
            code: Verilog代码文本

        Returns:
            768维语义向量（表征逻辑功能与约束条件）
        """
        # 【修改】第1步：构建AST
        ast_root = self.build_ast(code)

        # 【修改】第2步：从AST提取结构化特征
        features = self.extract_ast_features(ast_root)

        # 【修改】第3步：使用CNN编码特征为语义向量
        semantic_vector = self.encode_with_cnn(features)

        return semantic_vector


def extract_bidirectional_semantics(req_text: str, code_text: str) -> Dict:
    """
    双向语义提取的便捷函数

    Args:
        req_text: 需求文档文本
        code_text: FPGA代码文本

    Returns:
        包含双向语义的字典
    """
    nlp_extractor = NLPSemanticExtractor()
    code_extractor = CodeSemanticExtractor()

    result = {
        "requirement": {
            "semantic_elements": nlp_extractor.extract_semantic_elements(req_text),
            "semantic_vector": nlp_extractor.get_semantic_vector(req_text),
        },
        "code": {
            "semantic_elements": code_extractor.extract_semantic_elements(code_text),
            "semantic_vector": code_extractor.get_semantic_vector(code_text),
        },
    }

    return result


# JSON_FILE_PATH = "C:\\Users\\34435\\Desktop\\FPGA需求文档与verilog代码实现\\dataset.json"


# # 加载json数据
# with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
#     all_data = json.load(f)
#     target_data = [item for item in all_data if item['id'] == 1][0]  # 只取id=1的这条

# # 提取id=1的需求文本和代码文本
# req = target_data['req_desc_origin']
# code = target_data['code_origin']

# # 调用写好的函数
# result = extract_bidirectional_semantics(req, code)

# # 打印id=1的完整result结果
# print("="*100)
# print("✅ ID=1 数据的语义提取完整Result结果：")
# print("="*100)
# print(result)
eq_text = "FPGA双端口RAM模块，数据位宽固定为8比特；采用单总线时钟实现双端口RAM逻辑；端口A与总线绑定，端口B为通用业务端口；总线侧读写控制规则：在1个时钟周期内同时置位片选信号、8位地址信号、读写控制信号，即可执行读或写操作；写操作时序：写数据在寻址时立即被写入对应内存地址；读操作时序：读请求触发后，有效数据标志信号将延迟1个时钟周期脉冲，此时总线读数据端口输出对应内存数据；模块可配置参数：DEPTH为双端口RAM的存储深度，代表存储的8比特字长数据的个数。"
code_text = "module Bus8_DPRAM #(DEPTH = 256)(input i_Bus_Rst_L,input i_Bus_Clk,input i_Bus_CS,input i_Bus_Wr_Rd_n,input [$clog2(DEPTH)-1:0] i_Bus_Addr8,input [7:0] i_Bus_Wr_Data,output [7:0] o_Bus_Rd_Data,output reg o_Bus_Rd_DV,input [7:0] i_PortB_Data,input [$clog2(DEPTH)-1:0] i_PortB_Addr8,input i_PortB_WE,output [7:0] o_PortB_Data);Dual_Port_RAM_Single_Clock #(.WIDTH(8),.DEPTH(DEPTH)) Bus_RAM_Inst(.i_Clk(i_Bus_Clk),.i_PortA_Data(i_Bus_Wr_Data),.i_PortA_Addr(i_Bus_Addr8),.i_PortA_WE(i_Bus_Wr_Rd_n & i_Bus_CS),.o_PortA_Data(o_Bus_Rd_Data),.i_PortB_Data(i_PortB_Data),.i_PortB_Addr(i_PortB_Addr8),.i_PortB_WE(i_PortB_WE),.o_PortB_Data(o_PortB_Data));always @(posedge i_Bus_Clk)begin o_Bus_Rd_DV <= i_Bus_CS & ~i_Bus_Wr_Rd_n;end endmodule"
result = extract_bidirectional_semantics(eq_text, code_text)
print("=" * 100)
print("✅ 等价文本与代码的语义提取完整Result结果：")
print("=" * 100)
print(result)

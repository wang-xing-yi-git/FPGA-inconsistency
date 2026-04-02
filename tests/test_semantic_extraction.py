"""
测试语义提取模块
"""

import unittest
from src.semantic_extraction import (
    NLPSemanticExtractor,
    CodeSemanticExtractor,
    extract_bidirectional_semantics,
)


class TestNLPSemanticExtractor(unittest.TestCase):
    """NLP语义提取器测试"""

    def setUp(self):
        self.extractor = NLPSemanticExtractor()

    def test_preprocess_text(self):
        """测试文本预处理"""
        text = "The FPGA Design with Clock Signal"
        processed = self.extractor.preprocess_text(text)
        self.assertEqual(processed, "the fpga design with clock signal")

    def test_tokenize_and_clean(self):
        """测试分词和清理"""
        text = "The FPGA design requires a clock signal and reset signal"
        tokens = self.extractor.tokenize_and_clean(text)
        self.assertIn("fpga", tokens)
        self.assertIn("design", tokens)
        # 停用词应该被去除
        self.assertNotIn("the", tokens)

    def test_extract_semantic_elements(self):
        """测试语义要素提取"""
        text = "A clock-driven counter with 8-bit width and asynchronous reset"
        elements = self.extractor.extract_semantic_elements(text)
        self.assertIn("keywords", elements)
        self.assertIn("fpga_terms", elements)
        self.assertEqual(elements["element_type"], "nlp_text")


class TestCodeSemanticExtractor(unittest.TestCase):
    """代码语义提取器测试"""

    def setUp(self):
        self.extractor = CodeSemanticExtractor()

    def test_parse_verilog_code(self):
        """测试Verilog代码解析"""
        code = """
        module counter (
            input clk,
            input rst_n,
            output [7:0] count
        );
        always @(posedge clk) begin
            count <= count + 1;
        end
        endmodule
        """
        structure = self.extractor.parse_verilog_code(code)
        self.assertIn("counter", structure["modules"])
        self.assertIn("clk", structure["ports"]["input"])
        self.assertIn("count", structure["ports"]["output"])

    def test_extract_semantic_elements(self):
        """测试代码语义要素提取"""
        code = """
        module test (input clk, output [7:0] data);
        always @(posedge clk) begin
            data <= data + 1;
        end
        endmodule
        """
        elements = self.extractor.extract_semantic_elements(code)
        self.assertEqual(elements["element_type"], "code")
        self.assertEqual(elements["port_count"], 1)
        self.assertGreater(elements["behavior_count"], 0)

    def test_extract_fpga_features(self):
        """测试FPGA特征提取"""
        code = "always @(posedge clk) begin end"
        features = self.extractor._extract_fpga_features(code)
        feature_types = [f["type"] for f in features]  # 【修改】改为访问'type'键
        self.assertIn("sequential_logic", feature_types)
        self.assertIn("clock_domain", feature_types)  # 也可以验证clk检测


if __name__ == "__main__":
    unittest.main()

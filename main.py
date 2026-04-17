"""
主程序入口
整合所有模块实现完整的FPGA文实不一致检测流程
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

from src.semantic_extraction import NLPSemanticExtractor, CodeSemanticExtractor
from src.semantic_alignment import SemanticAligner
from src.inconsistency_detector import InconsistencyDetector
from src.data_processor import DataProcessor, ConfigLoader, ReportGenerator


class FPGAInconsistencyDetectionSystem:
    """FPGA文实不一致检测系统"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化系统

        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigLoader.load_config(config_path)
        self.nlp_extractor = NLPSemanticExtractor(
            model_name=self.config["nlp"].get("bert_model", "bert-base-uncased"),
            max_length=self.config["nlp"].get("bert_max_length", 512),
        )
        self.code_extractor = CodeSemanticExtractor()
        self.aligner = SemanticAligner()

        # 🚀 集成深度学习模型（GAT+Bi-GRU）
        model_path = self.config.get("models", {}).get(
            "deep_learning_model", "models/implicit_model_v3.pth"
        )
        self.detector = InconsistencyDetector(deep_learning_model_path=model_path)

        self.data_processor = DataProcessor()

    def process_item(self, item: Dict) -> Dict:
        """
        处理单个数据项

        Args:
            item: 输入数据项

        Returns:
            处理结果
        """
        req_id = item.get("id")
        req_text = item.get("req_desc_origin", "")
        code_text = item.get("code_origin", "")

        print(f"\n{'='*60}")
        print(f"Processing Item {req_id}")
        print(f"{'='*60}")

        # 1. 双向语义提取
        print(f"[1/4] 双向语义提取...")

        # 需求语义提取
        req_elements = self.nlp_extractor.extract_semantic_elements(req_text)
        req_vector = self.nlp_extractor.get_semantic_vector(req_text)
     

        # 代码语义提取
        code_elements = self.code_extractor.extract_semantic_elements(code_text)
        code_vector = self.code_extractor.get_semantic_vector(code_text)

        # 2. 语义对齐
        print(f"[2/4] 语义对齐...")
        alignment_result = self.aligner.align_requirements_to_code(
            req_id=req_id,
            req_elements=req_elements,
            req_vector=req_vector,
            code_elements=code_elements,
            code_vector=code_vector,
            code_segment=code_text,
        )

        # 3. 不一致检测
        print(f"[3/4] 不一致检测...")
        # 🚀 从对齐结果中提取对齐对信息，传给深度学习模型
        alignment_pairs = getattr(alignment_result, "alignment_pairs", []) or []
        alignment_confidence = getattr(alignment_result, "confidence", 0.5)

        inconsistency_result = self.detector.detect_all_inconsistencies(
            req_id=req_id,
            req_text=req_text,
            req_elements=req_elements,
            req_vector=req_vector,
            code_text=code_text,
            code_elements=code_elements,
            code_vector=code_vector,
            alignment_pairs=alignment_pairs,  # ✨ 传递对齐对
            alignment_confidence=alignment_confidence,  # ✨ 传递对齐置信度
        )

        # 4. 结果整合
        print(f"[4/4] 结果整合...")
        result = {
            "id": req_id,
            "req_summary": {
                "text": req_text[:200] + "..." if len(req_text) > 200 else req_text,
                "keywords_count": len(req_elements.get("keywords", [])),
            },
            "code_summary": {
                "text": code_text[:200] + "..." if len(code_text) > 200 else code_text,
                "modules": code_elements.get("modules", []),
                "port_count": code_elements.get("port_count", 0),
            },
            "alignment": {
                "status": alignment_result.status.value,
                "keyword_match": alignment_result.mapping_confidence,  # 【改进】细粒度关键词匹配
                "confidence": alignment_result.confidence,  # 综合置信度
                "reason": alignment_result.reason,  # 对齐判定原因
                "debug_info": alignment_result.debug_info,  # 【新增】调试详情
            },
            "inconsistency_detection": inconsistency_result,
        }

        # 打印摘要 ✨ 改进的输出逻辑
        print(f"\n结果摘要:")
        print(f"  对齐状态: {alignment_result.status.value}")
        print(f"  细粒度关键词匹配: {alignment_result.mapping_confidence:.2f}")
        print(f"  综合置信度: {alignment_result.confidence:.2f}")
        print(f"  判定原因: {alignment_result.reason}")

        # 分别显示DL推理和规则检测结果
        explicit_count = len(inconsistency_result["explicit_inconsistencies"])
        implicit_count = len(inconsistency_result["implicit_inconsistencies"])

        print(f"\n  📊 检测结果分析:")
        print(f"     规则检测: {explicit_count} 个")
        print(f"     启发式: {implicit_count} 个")

        # ✨ 显示深度学习推理结果（独立显示）
        if inconsistency_result.get("dl_inference"):
            dl_info = inconsistency_result["dl_inference"]
            print(f"\n  🤖 深度学习推理 (GAT+Bi-GRU):")
            print(f"     不一致度: {dl_info['score']:.4f} ({dl_info['severity']})")
            print(f"     对齐对数: {dl_info['alignment_pairs_count']}")
            if dl_info["is_inconsistent"]:
                print(f"     ⚠️  模型判定: 存在不一致")
            else:
                print(f"     ✓ 模型判定: 无不一致")

        # 显示总体统计
        total = inconsistency_result["total_issues"]
        severity_dist = inconsistency_result["severity_distribution"]
        print(f"\n  📈 总统计:")
        print(f"     总问题数: {total} 个")
        print(f"     严重程度分布: {severity_dist}")

        return result

    def process_dataset(self, input_path: str) -> List[Dict]:
        """
        处理整个数据集

        Args:
            input_path: 输入文件路径

        Returns:
            处理结果列表
        """
        print(f"加载数据集: {input_path}")

        # 加载数据
        dataset = self.data_processor.load_dataset(input_path)

        # 验证数据
        is_valid, errors = self.data_processor.validate_dataset(dataset)
        if not is_valid:
            print("数据验证失败:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)

        print(f"数据验证成功，共 {len(dataset)} 项")

        # 处理每一项
        results = []
        for i, item in enumerate(dataset, 1):
            processed_item = self.data_processor.preprocess_item(item)
            result = self.process_item(processed_item)
            results.append(result)

            if i < len(dataset):
                print(f"\n进度: {i}/{len(dataset)}")

        return results

    def save_results(
        self, results: List[Dict], output_path: str, format: str = "json"
    ) -> bool:
        """
        保存检测结果

        Args:
            results: 检测结果列表
            output_path: 输出路径
            format: 输出格式 ('json' 或 'html')

        Returns:
            是否成功保存
        """
        print(f"\n{'='*60}")
        print(f"生成报告")
        print(f"{'='*60}")

        if format == "json":
            # 保存JSON格式
            json_path = (
                output_path if output_path.endswith(".json") else output_path + ".json"
            )
            success = self.data_processor.save_results(results, json_path)
            if success:
                print(f"✓ JSON报告已保存: {json_path}")
            return success

        elif format == "html":
            # 生成HTML报告
            detailed_report = ReportGenerator.generate_detailed_report(results)
            html_path = (
                output_path if output_path.endswith(".html") else output_path + ".html"
            )
            success = ReportGenerator.generate_html_report(detailed_report, html_path)
            if success:
                print(f"✓ HTML报告已保存: {html_path}")

            # 同时保存JSON
            json_path = output_path.replace(".html", ".json")
            json_success = self.data_processor.save_results(results, json_path)
            if json_success:
                print(f"✓ JSON报告已保存: {json_path}")

            return success and json_success

        else:
            print(f"不支持的格式: {format}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FPGA设计文实不一致检测系统")
    parser.add_argument(
        "--input", type=str, default="data/raw/dataset.json", help="输入数据文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/detection_report",
        help="输出报告路径（不需要扩展名）",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--format", type=str, choices=["json", "html"], default="json", help="输出格式"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FPGA设计文实不一致检测系统")
    print("=" * 60)

    # 初始化系统
    system = FPGAInconsistencyDetectionSystem(args.config)

    # 🚀 显示深度学习模型加载状态
    if system.detector.implicit_detector.deep_learning_model is not None:
        print("✅ 深度学习模型 (GAT+Bi-GRU) 已加载")
        print(
            f"   模型路径: {system.config.get('models', {}).get('deep_learning_model', 'models/implicit_model_v3.pth')}"
        )
        print("   检测方式: 优先使用深度学习 + 启发式降级")
    else:
        print("⚠️  深度学习模型未找到，将使用启发式规则检测")

    print()  # 空行分隔

    # 处理数据集
    results = system.process_dataset(args.input)

    # 保存结果
    system.save_results(results, args.output, args.format)

    # 生成汇总报告
    summary = ReportGenerator.generate_summary_report(results)
    print(f"\n{'='*60}")
    print("检测汇总")
    print(f"{'='*60}")
    print(f"总检测项: {summary['summary']['total_items']}")
    print(f"包含问题的项: {summary['summary']['items_with_issues']}")
    print(f"总不一致: {summary['summary']['total_issues']}")
    print(f"严重程度分布: {summary['severity_distribution']}")

    print(f"\n✓ 检测完成！")


if __name__ == "__main__":
    main()

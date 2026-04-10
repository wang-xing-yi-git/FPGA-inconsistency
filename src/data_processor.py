"""
数据处理模块
处理输入数据和输出格式
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持NumPy数据类型"""

    def default(self, obj):
        """处理NumPy类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def convert_numpy_types(obj):
    """递归转换对象中的NumPy类型"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class DataProcessor:
    """数据处理器"""

    @staticmethod
    def load_dataset(filepath: str) -> List[Dict]:
        """
        加载数据集

        Args:
            filepath: 数据文件路径

        Returns:
            数据列表
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else [data]

    @staticmethod
    def validate_dataset(data: List[Dict]) -> tuple[bool, List[str]]:
        """
        验证数据集格式

        Args:
            data: 数据列表

        Returns:
            (是否有效, 错误列表) 元组
        """
        errors = []
        required_fields = ["id", "req_desc_origin", "code_origin"]

        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {i}: missing field '{field}'")

        return len(errors) == 0, errors

    @staticmethod
    def preprocess_item(item: Dict) -> Dict:
        """
        预处理单个数据项

        Args:
            item: 数据项

        Returns:
            预处理后的数据项
        """
        processed = {
            "id": item.get("id"),
            "req_desc_origin": item.get("req_desc_origin", "").strip(),
            "code_origin": item.get("code_origin", "").strip(),
            "metadata": item.get("metadata", {}),
        }

        return processed

    @staticmethod
    def save_results(results: List[Dict], output_path: str) -> bool:
        """
        保存检测结果

        Args:
            results: 检测结果列表
            output_path: 输出文件路径

        Returns:
            是否成功保存
        """
        try:
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # 转换NumPy类型
            results_converted = convert_numpy_types(results)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    results_converted, f, ensure_ascii=False, indent=2, cls=NumpyEncoder
                )

            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    @staticmethod
    def load_rules(rules_path: str) -> Optional[Dict]:
        """
        加载规则库

        Args:
            rules_path: 规则文件路径

        Returns:
            规则库字典
        """
        if not os.path.exists(rules_path):
            return None

        with open(rules_path, "r", encoding="utf-8") as f:
            return json.load(f)


from typing import Tuple


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Optional[Dict]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            print("Warning: PyYAML not installed, using default config")
            return DataProcessor._get_default_config()
        except FileNotFoundError:
            print(f"Config file not found: {config_path}, using default config")
            return DataProcessor._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict:
        """
        获取默认配置

        Returns:
            默认配置字典
        """
        return {
            "nlp": {
                "bert_model": "bert-base-uncased",
                "bert_max_length": 512,
                "embedding_dim": 768,
            },
            "code": {"ast_encoding_dim": 256, "cnn_filters": [32, 64, 128]},
            "alignment": {
                "similarity_threshold": 0.7,
                "high_similarity_threshold": 0.8,
                "low_similarity_threshold": 0.5,
            },
            "inconsistency": {
                "gat": {"num_heads": 4, "hidden_dim": 128, "num_layers": 2},
                "bigru": {"hidden_dim": 64, "num_layers": 1},
                "confidence_threshold": 0.5,
            },
            "output": {
                "report_format": "json",
                "report_path": "./reports",
                "save_intermediate": True,
            },
        }


class ReportGenerator:
    """报告生成器"""

    @staticmethod
    def generate_summary_report(results: List[Dict]) -> Dict:
        """
        生成汇总报告

        Args:
            results: 检测结果列表

        Returns:
            汇总报告
        """
        total_items = len(results)

        # 初始化统计
        total_issues = 0
        items_with_issues = 0
        severity_stats = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}

        # 遍历每个结果并统计
        for result in results:
            # 获取不一致检测结果（嵌套在 inconsistency_detection 中）
            inconsistency_result = result.get("inconsistency_detection", {})

            # 获取总不一致数量
            item_total_issues = inconsistency_result.get("total_issues", 0)
            total_issues += item_total_issues

            # 累计有问题的项数
            if item_total_issues > 0:
                items_with_issues += 1

            # 累计严重程度分布
            severity_dist = inconsistency_result.get("severity_distribution", {})
            for severity, count in severity_dist.items():
                if severity in severity_stats:
                    try:
                        severity_stats[severity] += int(count)
                    except (ValueError, TypeError):
                        pass

        # 生成报告
        report = {
            "summary": {
                "total_items": total_items,
                "items_with_issues": items_with_issues,
                "items_without_issues": total_items - items_with_issues,
                "total_issues": total_issues,
                "average_issues_per_item": (
                    total_issues / total_items if total_items > 0 else 0
                ),
                "issue_percentage": (
                    (items_with_issues / total_items * 100) if total_items > 0 else 0
                ),
            },
            "severity_distribution": severity_stats,
            "severity_percentage": {
                key: (count / total_issues * 100) if total_issues > 0 else 0
                for key, count in severity_stats.items()
            },
        }

        return report

    @staticmethod
    def generate_detailed_report(results: List[Dict]) -> Dict:
        """
        生成详细报告

        Args:
            results: 检测结果列表

        Returns:
            详细报告
        """
        summary = ReportGenerator.generate_summary_report(results)

        return {
            "metadata": {
                "report_type": "FPGA Inconsistency Detection Report",
                "version": "1.0",
            },
            "summary": summary,
            "detailed_results": results,
        }

    @staticmethod
    def generate_html_report(results: Dict, output_path: str) -> bool:
        """
        生成HTML报告

        Args:
            results: 检测结果
            output_path: 输出路径

        Returns:
            是否成功生成
        """
        try:
            html_content = ReportGenerator._build_html(results)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            return True
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            return False

    @staticmethod
    def _build_html(results: Dict) -> str:
        """
        构建HTML内容

        Args:
            results: 检测结果

        Returns:
            HTML字符串
        """
        summary = results.get("summary", {})
        summary_info = summary.get("summary", {})
        severity_dist = summary.get("severity_distribution", {})

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>FPGA Inconsistency Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stat {{ display: inline-block; margin: 10px 20px; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .stat-label {{ font-size: 12px; color: #666; }}
                .severity-critical {{ color: #d32f2f; }}
                .severity-high {{ color: #f57c00; }}
                .severity-medium {{ color: #fbc02d; }}
                .severity-low {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>FPGA 设计文实不一致检测报告</h1>
            
            <div class="summary">
                <h2>检测汇总</h2>
                <div class="stat">
                    <div class="stat-value">{summary_info.get('total_items', 0)}</div>
                    <div class="stat-label">总检测项</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary_info.get('total_issues', 0)}</div>
                    <div class="stat-label">发现不一致</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{summary_info.get('issue_percentage', 0):.1f}%</div>
                    <div class="stat-label">包含问题的项</div>
                </div>
            </div>
            
            <h2>严重程度分布</h2>
            <table>
                <tr>
                    <th>严重程度</th>
                    <th>数量</th>
                    <th>比例</th>
                </tr>
                <tr>
                    <td class="severity-critical">严重</td>
                    <td>{severity_dist.get('critical', 0)}</td>
                    <td>{severity_dist.get('critical', 0)}</td>
                </tr>
                <tr>
                    <td class="severity-high">高</td>
                    <td>{severity_dist.get('high', 0)}</td>
                    <td>{severity_dist.get('high', 0)}</td>
                </tr>
                <tr>
                    <td class="severity-medium">中</td>
                    <td>{severity_dist.get('medium', 0)}</td>
                    <td>{severity_dist.get('medium', 0)}</td>
                </tr>
                <tr>
                    <td class="severity-low">低</td>
                    <td>{severity_dist.get('low', 0)}</td>
                    <td>{severity_dist.get('low', 0)}</td>
                </tr>
            </table>
        </body>
        </html>
        """

        return html

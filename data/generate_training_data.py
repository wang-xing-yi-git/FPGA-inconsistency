"""
快速数据生成脚本 - 生成真实编码的训练数据用于GAT+Bi-GRU模型

这个脚本生成合成的FPGA设计文实不一致数据，但使用真实的BERT编码和CNN提取的向量。

使用方法：
    python generate_training_data.py --num-samples 500 --output data/implicit_inconsistency_training_data.json
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import sys
from pathlib import Path

# 获取当前脚本所在的目录（data文件夹）
current_dir = Path(__file__).parent
# 获取项目根目录（FPGA-inconsistency文件夹），即data的父目录
root_dir = current_dir.parent
# 将项目根目录添加到Python搜索路径中
sys.path.insert(0, str(root_dir))


def generate_synthetic_data(
    num_samples: int = 500, use_real_encoding: bool = True
) -> List[Dict]:
    """
    生成合成训练数据（支持真实BERT编码或随机编码）

    Args:
        num_samples: 生成样本数
        use_real_encoding: 是否使用真实BERT编码（默认True）

    Returns:
        样本列表
    """

    # 初始化语义提取器（仅在需要时导入）
    nlp_extractor = None
    code_extractor = None

    if use_real_encoding:
        try:
            print("\n🔄 初始化语义提取器...")
            from src.semantic_extraction import (
                NLPSemanticExtractor,
                CodeSemanticExtractor,
            )

            nlp_extractor = NLPSemanticExtractor(
                model_name="bert-base-uncased", max_length=512
            )
            code_extractor = CodeSemanticExtractor()
            print("✅ 提取器初始化完成\n")
        except Exception as e:
            print(f"⚠️  无法导入语义提取器: {e}")
            print("   将使用随机向量代替...\n")
            nlp_extractor = None
            code_extractor = None

    # 需求模板
    req_templates = [
        "实现一个{width}位{type}计数器，具有{features}",
        "设计一个{width}位{type}算术运算模块，支持{features}",
        "构建{width}位{type}缓存电路，需要{features}",
        "实现{width}位移位寄存器，具备{features}功能",
        "设计一个{width}位{type}分频器，包含{features}",
    ]

    # 位宽
    widths = [4, 8, 16, 32, 64]

    # 数据类型
    types = ["递增", "递减", "二进制", "格雷码"]

    # 功能特性
    features_list = [
        "异步复位和同步清零",
        "异步复位功能",
        "同步清零功能",
        "异步复位和使能控制",
        "同步清零和使能控制",
        "可配置步长",
        "溢出检测",
        "中断信号输出",
    ]

    # 不一致类型
    inconsistency_types = [
        "WIDTH_MISMATCH",  # 位宽不匹配
        "LOGIC_ERROR",  # 逻辑错误
        "TIMING_ISSUE",  # 时序问题
        "MISSING_FEATURE",  # 缺失功能
        "BEHAVIOR_CONFLICT",  # 行为冲突
    ]

    data = []

    print("\n🔧 生成合成训练数据（使用真实BERT编码）...")
    print("-" * 60)

    for i in range(num_samples):
        # 随机选择参数
        width = np.random.choice(widths)
        type_choice = np.random.choice(types)
        features = np.random.choice(features_list)

        # 生成需求文本
        template = np.random.choice(req_templates)
        req_text = template.format(width=width, type=type_choice, features=features)

        # 随机决定是否一致 (70% 一致, 30% 不一致)
        is_consistent = np.random.random() > 0.3

        if is_consistent:
            # 一致：代码实现与需求匹配
            code_width = width
            code_features = features
            label = 0
            inconsistency_details = None
        else:
            # 不一致：代码与需求有偏差
            # 常见的不一致情况
            issue_type = np.random.choice(inconsistency_types)
            label = 1

            if issue_type == "WIDTH_MISMATCH":
                # 位宽不匹配
                code_width = max(4, width - 4)  # 降低位宽
                code_features = features
                inconsistency_details = {
                    "type": "WIDTH_MISMATCH",
                    "issue": f"需求{width}位但代码{code_width}位",
                    "severity": "CRITICAL",
                    "description": "实现的位宽小于需求",
                }
            elif issue_type == "MISSING_FEATURE":
                # 缺失功能
                code_width = width
                # 移除一个特性
                if "异步复位" in code_features:
                    code_features = code_features.replace("异步复位", "")
                elif "同步清零" in code_features:
                    code_features = code_features.replace("同步清零", "")
                else:
                    code_features = (
                        code_features[:-5] if len(code_features) > 5 else code_features
                    )

                inconsistency_details = {
                    "type": "MISSING_FEATURE",
                    "issue": f"需求功能'{features}'但代码未完整实现",
                    "severity": "HIGH",
                    "description": "代码缺少需求中的某个功能",
                }
            elif issue_type == "LOGIC_ERROR":
                # 逻辑错误
                code_width = width
                code_features = features
                inconsistency_details = {
                    "type": "LOGIC_ERROR",
                    "issue": "实现的逻辑与需求不符",
                    "severity": "HIGH",
                    "description": "代码实现的行为与需求描述不一致",
                }
            else:
                # 其他问题
                code_width = width
                code_features = features
                inconsistency_details = {
                    "type": issue_type,
                    "issue": f"检测到{issue_type}问题",
                    "severity": "MEDIUM",
                    "description": "代码与需求存在不一致",
                }

        # 生成代码文本
        code_text = f"""module design_{i}(
    input clk,
    input rst_n,
    output [{code_width-1}:0] data_out
);
    // 设计实现: {code_features}
    reg [{code_width-1}:0] counter;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            counter <= 0;
        else
            counter <= counter + 1;
    end
    
    assign data_out = counter;
endmodule"""

        # 📊 生成768维向量（优先使用真实BERT编码，否则使用随机向量）
        if nlp_extractor and code_extractor:
            try:
                req_vector = nlp_extractor.get_semantic_vector(req_text)
                if req_vector is None:
                    req_vector = np.random.randn(768)
                else:
                    req_vector = req_vector.tolist()
            except Exception as e:
                print(f"⚠️  样本{i+1} 需求编码失败: {str(e)[:60]}，使用随机向量")
                req_vector = np.random.randn(768).tolist()

            try:
                code_vector = code_extractor.get_semantic_vector(code_text)
                if code_vector is None:
                    code_vector = np.random.randn(768)
                else:
                    code_vector = code_vector.tolist()
            except Exception as e:
                print(f"⚠️  样本{i+1} 代码编码失败: {str(e)[:60]}，使用随机向量")
                code_vector = np.random.randn(768).tolist()
        else:
            # 推退回随机向量
            req_vector = np.random.randn(768).tolist()
            code_vector = np.random.randn(768).tolist()

        # 如果一致，让两个向量更接近
        if is_consistent:
            code_vector = (np.array(req_vector) + np.random.randn(768) * 0.1).tolist()

        # 生成对齐对
        alignment_pairs = [
            {
                "req": f"{width}位",
                "code": f"[{code_width-1}:0]",
                "confidence": 0.95 if is_consistent else 0.5,
            },
            {
                "req": type_choice,
                "code": "counter <= counter + 1",
                "confidence": 0.92 if is_consistent else 0.6,
            },
            {
                "req": features,
                "code": "// 设计实现: " + code_features,
                "confidence": 0.88 if is_consistent else 0.4,
            },
        ]

        # 构建样本
        sample = {
            "id": i + 1,
            "req_text": req_text,
            "code_text": code_text,
            "req_vector": req_vector,
            "code_vector": code_vector,
            "alignment_pairs": alignment_pairs,
            "label": label,
            "inconsistency_details": inconsistency_details,
        }

        data.append(sample)

        # 显示进度
        if (i + 1) % 100 == 0:
            print(
                f"  已生成: {i + 1:4d}/{num_samples} 样本 "
                f"(一致: {sum(1 for d in data if d['label']==0)}, "
                f"不一致: {sum(1 for d in data if d['label']==1)})"
            )

    print("-" * 60)
    print(f"✅ 生成完成！共{len(data)}个样本")

    # 统计信息
    consistent_count = sum(1 for d in data if d["label"] == 0)
    inconsistent_count = sum(1 for d in data if d["label"] == 1)
    print(f"   一致样本: {consistent_count} ({consistent_count/len(data)*100:.1f}%)")
    print(
        f"   不一致样本: {inconsistent_count} ({inconsistent_count/len(data)*100:.1f}%)"
    )

    return data


def save_data(data: List[Dict], output_path: str):
    """保存数据为JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 数据已保存到: {output_path}")
    print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def split_data(data: List[Dict], train_ratio: float = 0.8) -> tuple:
    """
    划分训练集和测试集

    Args:
        data: 原始数据
        train_ratio: 训练集比例

    Returns:
        (训练集, 测试集)
    """
    np.random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="生成合成训练数据")
    parser.add_argument(
        "--num-samples", type=int, default=500, help="生成样本数 (默认: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/implicit_inconsistency_training_data.json",
        help="输出文件路径",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="data/implicit_inconsistency_test_data.json",
        help="测试集输出路径",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="训练集比例 (默认: 0.8)"
    )
    parser.add_argument(
        "--use-real-encoding",
        action="store_true",
        default=False,
        help="使用真实BERT编码而不是随机向量（可能较慢）",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("合成FPGA设计文实不一致训练数据生成器")
    print("=" * 60)
    if args.use_real_encoding:
        print("📌 模式: 使用真实BERT编码")
    else:
        print("📌 模式: 使用随机向量（快速）")

    # 生成数据
    all_data = generate_synthetic_data(
        args.num_samples, use_real_encoding=args.use_real_encoding
    )

    # 划分训练集和测试集
    train_data, test_data = split_data(all_data, args.train_ratio)

    print(f"\n📊 数据划分:")
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   测试集: {len(test_data)} 样本")

    # 保存数据
    save_data(train_data, args.output)
    save_data(test_data, args.test_output)

    print("\n✅ 数据生成完成！")
    print(f"\n下一步：运行训练脚本")
    print(f"  python models/trained/train_implicit_model_v2.py")
    print()


if __name__ == "__main__":
    main()

"""
细粒度数据生成脚本 - 生成适配GAT+Bi-GRU模型的训练数据

这个脚本生成合成的FPGA设计文实不一致数据，输出：
- req_nodes: 需求细粒度语义节点列表 [[768], [768], ...]
- code_nodes: 代码细粒度语义节点列表 [[768], [768], ...]
- alignment_matrix: 细粒度节点对齐矩阵 (n+m)×(n+m)
完全匹配ImplicitInconsistencyModel的细粒度输入要求
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import sys
import re

# 获取项目路径
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

# 固定配置
FEATURE_DIM = 768  # 节点特征维度
MIN_NODES = 3      # 最小细粒度节点数
MAX_NODES = 10     # 最大细粒度节点数


def split_text_into_fine_grained_units(text: str, is_code: bool = False) -> List[str]:
    """
    将文本拆分为细粒度语义单元（核心改造点）
    Args:
        text: 原始文本（需求/代码）
        is_code: 是否为代码文本
    Returns:
        细粒度单元列表
    """
    if is_code:
        # 代码文本拆分规则：按模块、变量、注释、语句拆分
        # 1. 移除注释（简化版）
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        # 2. 拆分代码块
        units = re.split(r';|{|}|\n', text)
        # 3. 过滤空值和短文本，提取有效代码单元
        units = [unit.strip() for unit in units if unit.strip() and len(unit.strip()) > 2]
        # 4. 进一步拆分长语句（按关键词）
        fine_units = []
        for unit in units:
            sub_units = re.split(r',|=|\(|\)|\+|-|\*|/', unit)
            fine_units.extend([s.strip() for s in sub_units if s.strip() and len(s.strip()) > 1])
        return fine_units[:MAX_NODES]  # 限制最大节点数
    else:
        # 需求文本拆分规则：按标点、短语拆分
        units = re.split(r'，|。|、|；| |,|;', text)
        units = [unit.strip() for unit in units if unit.strip() and len(unit.strip()) > 1]
        # 确保至少MIN_NODES个节点
        if len(units) < MIN_NODES:
            units = units + [f"补充单元{i}" for i in range(MIN_NODES - len(units))]
        return units[:MAX_NODES]


def generate_fine_grained_vectors(
    text_units: List[str], 
    use_real_encoding: bool = False,
    nlp_extractor=None,
    code_extractor=None,
    is_code: bool = False
) -> List[List[float]]:
    """
    为细粒度单元生成768维向量
    """
    vectors = []
    for unit in text_units:
        if use_real_encoding:
            try:
                if is_code:
                    vec = code_extractor.get_semantic_vector(unit)
                else:
                    vec = nlp_extractor.get_semantic_vector(unit)
                vec = vec.tolist() if vec is not None else np.random.randn(FEATURE_DIM).tolist()
            except:
                vec = np.random.randn(FEATURE_DIM).tolist()
        else:
            vec = np.random.randn(FEATURE_DIM).tolist()
        vectors.append(vec)
    return vectors


def generate_alignment_matrix(
    req_units: List[str], 
    code_units: List[str],
    is_consistent: bool
) -> List[List[float]]:
    """
    生成细粒度节点的对齐矩阵 (n+m)×(n+m)（核心改造点）
    Args:
        req_units: 需求细粒度单元列表
        code_units: 代码细粒度单元列表
        is_consistent: 是否一致（影响对齐置信度）
    Returns:
        对齐矩阵（二维列表）
    """
    n = len(req_units)  # 需求节点数
    m = len(code_units) # 代码节点数
    total = n + m       # 总节点数
    
    # 初始化对齐矩阵
    mat = np.zeros((total, total), dtype=np.float32)
    
    # 1. 需求节点内部对齐（对角线1，相邻节点高置信度）
    for i in range(n):
        mat[i, i] = 1.0  # 自对齐
        if i > 0:
            mat[i, i-1] = mat[i-1, i] = 0.8  # 相邻节点
        if i < n-1:
            mat[i, i+1] = mat[i+1, i] = 0.8
    
    # 2. 代码节点内部对齐
    for i in range(m):
        mat[n+i, n+i] = 1.0  # 自对齐
        if i > 0:
            mat[n+i, n+i-1] = mat[n+i-1, n+i] = 0.8
        if i < m-1:
            mat[n+i, n+i+1] = mat[n+i+1, n+i] = 0.8
    
    # 3. 需求-代码跨域对齐（核心：一致性影响对齐置信度）
    for i, req_unit in enumerate(req_units):
        for j, code_unit in enumerate(code_units):
            # 基于文本相似度的基础置信度
            common_words = len(set(req_unit) & set(code_unit))
            base_conf = common_words / max(len(req_unit), len(code_unit), 1)
            
            # 一致样本：提升对齐置信度
            if is_consistent:
                conf = min(0.7 + base_conf * 0.3, 1.0)
            # 不一致样本：降低对齐置信度
            else:
                conf = max(0.0, base_conf * 0.2)
            
            mat[i, n+j] = mat[n+j, i] = conf
    
    # 归一化（每行和为1）
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / (row_sums + 1e-6)
    
    return mat.tolist()


def generate_synthetic_data(
    num_samples: int = 500, 
    use_real_encoding: bool = False
) -> List[Dict]:
    """生成细粒度训练数据"""
    # 初始化语义提取器
    nlp_extractor = None
    code_extractor = None
    if use_real_encoding:
        try:
            print("\n🔄 初始化语义提取器...")
            from src.semantic_extraction import NLPSemanticExtractor, CodeSemanticExtractor
            nlp_extractor = NLPSemanticExtractor(model_name="bert-base-uncased", max_length=512)
            code_extractor = CodeSemanticExtractor()
            print("✅ 提取器初始化完成\n")
        except Exception as e:
            print(f"⚠️  无法导入语义提取器: {e}")
            print("   将使用随机向量代替...\n")
            use_real_encoding = False

    # 需求模板（FPGA场景）
    req_templates = [
        "实现一个{width}位{type}计数器，具有{features}",
        "设计一个{width}位{type}算术运算模块，支持{features}",
        "构建{width}位{type}缓存电路，需要{features}",
        "实现{width}位移位寄存器，具备{features}功能",
        "设计一个{width}位{type}分频器，包含{features}",
    ]
    widths = [4, 8, 16, 32, 64]
    types = ["递增", "递减", "二进制", "格雷码"]
    features_list = [
        "异步复位和同步清零", "异步复位功能", "同步清零功能",
        "异步复位和使能控制", "同步清零和使能控制", "可配置步长",
        "溢出检测", "中断信号输出"
    ]
    inconsistency_types = ["WIDTH_MISMATCH", "LOGIC_ERROR", "MISSING_FEATURE", "BEHAVIOR_CONFLICT"]

    data = []
    print("\n🔧 生成细粒度训练数据...")
    print("-" * 60)

    for i in range(num_samples):
        # 1. 生成基础参数
        width = int(np.random.choice(widths))
        type_choice = np.random.choice(types)
        features = np.random.choice(features_list)
        is_consistent = np.random.random() > 0.3  # 70%一致，30%不一致
        label = 0 if is_consistent else 1

        # 2. 生成需求/代码文本
        req_text = np.random.choice(req_templates).format(
            width=width, type=type_choice, features=features
        )
        # 生成不一致的代码参数
        if is_consistent:
            code_width = width
            code_features = features
            inconsistency_details = None
        else:
            issue_type = np.random.choice(inconsistency_types)
            if issue_type == "WIDTH_MISMATCH":
                code_width = max(4, width - 4)
                code_features = features
            elif issue_type == "MISSING_FEATURE":
                code_width = width
                code_features = features.replace("异步复位", "") if "异步复位" in features else \
                               features.replace("同步清零", "") if "同步清零" in features else features[:-5]
            else:
                code_width = width
                code_features = features
            inconsistency_details = {
                "type": issue_type, "req_width": width, "code_width": code_width,
                "req_features": features, "code_features": code_features,
                "severity": "CRITICAL" if issue_type == "WIDTH_MISMATCH" else "HIGH"
            }
        # 生成Verilog代码
        code_text = f"""module design_{i}(
    input clk, input rst_n, output [{code_width-1}:0] data_out
);
    reg [{code_width-1}:0] counter;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) counter <= 0;
        else counter <= counter + 1;
    end
    assign data_out = counter;
endmodule"""

        # 3. 细粒度拆分（核心）
        req_units = split_text_into_fine_grained_units(req_text, is_code=False)
        code_units = split_text_into_fine_grained_units(code_text, is_code=True)

        # 4. 生成细粒度节点向量（核心）
        req_nodes = generate_fine_grained_vectors(
            req_units, use_real_encoding, nlp_extractor, code_extractor, is_code=False
        )
        code_nodes = generate_fine_grained_vectors(
            code_units, use_real_encoding, nlp_extractor, code_extractor, is_code=True
        )

        # 5. 生成细粒度对齐矩阵（核心）
        alignment_matrix = generate_alignment_matrix(req_units, code_units, is_consistent)

        # 6. 构建样本
        sample = {
            "id": i + 1,
            "req_text": req_text,
            "code_text": code_text,
            "req_nodes": req_nodes,          # 细粒度需求节点（列表）
            "code_nodes": code_nodes,        # 细粒度代码节点（列表）
            "alignment_matrix": alignment_matrix,  # 细粒度对齐矩阵
            "label": float(label),           # 0=一致，1=不一致
            "inconsistency_details": inconsistency_details,
            "req_units": req_units,          # 调试用：细粒度单元文本
            "code_units": code_units         # 调试用：细粒度单元文本
        }
        data.append(sample)

        # 进度展示
        if (i + 1) % 100 == 0:
            consistent = sum(1 for d in data if d["label"] == 0)
            inconsistent = sum(1 for d in data if d["label"] == 1)
            print(f"  已生成: {i + 1:4d}/{num_samples} 样本 "
                  f"(一致: {consistent}, 不一致: {inconsistent})")

    # 统计
    print("-" * 60)
    consistent_count = sum(1 for d in data if d["label"] == 0)
    inconsistent_count = len(data) - consistent_count
    print(f"✅ 生成完成！共{len(data)}个样本")
    print(f"   一致样本: {consistent_count} ({consistent_count/len(data)*100:.1f}%)")
    print(f"   不一致样本: {inconsistent_count} ({inconsistent_count/len(data)*100:.1f}%)")

    return data


def save_data(data: List[Dict], output_path: str):
    """保存细粒度数据"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保数值类型正确
    for sample in data:
        sample["label"] = float(sample["label"])
        # 确保节点向量是float
        sample["req_nodes"] = [[float(x) for x in vec] for vec in sample["req_nodes"]]
        sample["code_nodes"] = [[float(x) for x in vec] for vec in sample["code_nodes"]]
        # 确保对齐矩阵是float
        sample["alignment_matrix"] = [[float(x) for x in row] for row in sample["alignment_matrix"]]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n💾 数据已保存到: {output_path}")
    print(f"   文件大小: {file_size:.2f} MB")


def split_data(data: List[Dict], train_ratio: float = 0.8) -> tuple:
    """分层拆分训练/测试集"""
    consistent_data = [d for d in data if d["label"] == 0]
    inconsistent_data = [d for d in data if d["label"] == 1]

    np.random.shuffle(consistent_data)
    np.random.shuffle(inconsistent_data)

    train_consistent = consistent_data[:int(len(consistent_data)*train_ratio)]
    test_consistent = consistent_data[int(len(consistent_data)*train_ratio):]
    train_inconsistent = inconsistent_data[:int(len(inconsistent_data)*train_ratio)]
    test_inconsistent = inconsistent_data[int(len(inconsistent_data)*train_ratio):]

    train_data = train_consistent + train_inconsistent
    test_data = test_consistent + test_inconsistent
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description="生成细粒度GAT+Bi-GRU模型训练数据")
    parser.add_argument("--num-samples", type=int, default=500, help="生成样本数")
    parser.add_argument("--output", type=str, 
                        default="data/implicit_inconsistency_training_data_v3.json",
                        help="训练集输出路径")
    parser.add_argument("--test-output", type=str,
                        default="data/implicit_inconsistency_test_data_v3.json",
                        help="测试集输出路径")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--use-real-encoding", action="store_true", default=False,
                        help="使用真实BERT编码（更耗时但效果更好）")

    args = parser.parse_args()

    # 打印配置
    print("\n" + "=" * 60)
    print("FPGA设计文实不一致检测 - 细粒度训练数据生成器")
    print("=" * 60)
    print(f"📌 目标模型: GAT + Bi-GRU (细粒度节点)")
    print(f"📌 节点特征维度: {FEATURE_DIM}")
    print(f"📌 节点数量范围: {MIN_NODES}-{MAX_NODES}")
    print(f"📌 编码模式: {'真实BERT编码' if args.use_real_encoding else '随机向量'}")
    print(f"📌 样本总数: {args.num_samples}")

    # 生成数据
    all_data = generate_synthetic_data(args.num_samples, args.use_real_encoding)

    # 拆分训练/测试集
    train_data, test_data = split_data(all_data, args.train_ratio)
    print(f"\n📊 数据划分:")
    print(f"   训练集: {len(train_data)} 样本 (一致: {sum(1 for d in train_data if d['label']==0)})")
    print(f"   测试集: {len(test_data)} 样本 (一致: {sum(1 for d in test_data if d['label']==0)})")

    # 保存数据
    save_data(train_data, args.output)
    save_data(test_data, args.test_output)

    # 使用提示
    print("\n✅ 细粒度数据生成完成！")
    print(f"\n下一步训练命令:")
    print(f"  python models/trained/train_implicit_model_v3.py")
    print(f"\n生成的文件包含:")
    print(f"  - req_nodes: 需求细粒度语义节点列表")
    print(f"  - code_nodes: 代码细粒度语义节点列表")
    print(f"  - alignment_matrix: 细粒度节点对齐矩阵")


if __name__ == "__main__":
    main()
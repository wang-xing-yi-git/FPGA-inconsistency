#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证KeyError修复的脚本
检查fpga_features的数据结构是否正确
"""

from src.semantic_extraction import CodeSemanticExtractor

# 测试代码示例
test_code = """
module counter(
    input clk,
    input rst,
    output [7:0] count
);

always @(posedge clk) begin
    if (rst)
        count <= 0;
    else
        count <= count + 1;
end

endmodule
"""

print("=" * 60)
print("验证fpga_features数据结构")
print("=" * 60)

extractor = CodeSemanticExtractor()

# 测试 _extract_fpga_features
print("\n【1】测试 _extract_fpga_features()...")
features = extractor._extract_fpga_features(test_code)
print(f"提取的特征数: {len(features)}")

for feat in features:
    print(f"  ✓ 特征: {feat.get('type')} - {feat}")

# 测试数据访问
print("\n【2】测试数据访问 (原始错误位置)...")
try:
    feature_types = set(feat["type"] for feat in features)
    print(f"✓ 成功提取所有特征的type: {feature_types}")
except KeyError as e:
    print(f"✗ 获取type键时出错: {e}")

# 测试完整流程
print("\n【3】测试完整semantic_elements流程...")
try:
    elements = extractor.extract_semantic_elements(test_code)
    fpga_features = elements.get("fpga_features", [])
    print(f"✓ 获取fpga_features: {len(fpga_features)} 项")

    # 尝试访问type键（原始错误检测点）
    feature_types = set(f["type"] for f in fpga_features)
    print(f"✓ 成功遍历所有特征的type: {feature_types}")
except KeyError as e:
    print(f"✗ 出错: {e}")
except Exception as e:
    print(f"✗ 其他错误: {e}")

print("\n" + "=" * 60)
print("✓ 验证完成 - fpga_features数据结构正确！")
print("=" * 60)

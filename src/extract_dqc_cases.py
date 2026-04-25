"""
从DQC Test Suite提取有标签的XBRL错误案例
"""

import os
import xml.etree.ElementTree as ET
import json
import re
from collections import defaultdict

def parse_dqc_output_file(filepath):
    """解析DQC输出文件，提取错误信息"""

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        errors = []

        for entry in root.findall('.//entry'):
            code = entry.get('code', '')
            level = entry.get('level', '')

            if level == 'error' and code.startswith('DQC'):
                message_elem = entry.find('message')
                if message_elem is not None:
                    message = message_elem.text

                    # 提取错误类型
                    error_type = classify_error_type(code, message)

                    # 提取涉及的fact
                    ref = entry.find('ref')
                    involved_facts = []
                    if ref is not None:
                        for prop in ref.findall('.//property'):
                            name = prop.get('name', '')
                            value = prop.text or ''
                            involved_facts.append({'name': name, 'value': value})

                    errors.append({
                        'code': code,
                        'error_type': error_type,
                        'message': message[:500],  # 截断message
                        'involved_facts': involved_facts,
                        'source_file': os.path.basename(filepath)
                    })

        return errors

    except Exception as e:
        return None

def classify_error_type(code, message):
    """分类DQC错误类型"""

    # DQC错误类型分类
    error_categories = {
        'calculation': ['0044', '0045', '0046', '0047', '0048',  # 计算不一致
                        '0227', '0228', '0230', '0231'],  # 计算网络
        'value_sign': ['0001', '0002', '0003'],  # 值/符号问题
        'dimension': ['0003', '0004', '0005'],  # 维度问题
        'unit': ['0004', '0006', '0007'],  # 单位问题
        'element_selection': ['0044', '0085', '0086'],  # 元素选择错误
    }

    # 从code提取规则号
    rule_num = code.split('.')[2] if '.' in code else ''

    for category, rule_nums in error_categories.items():
        if rule_num in rule_nums:
            return category

    # 检查message关键词
    if 'calculation' in message.lower() or 'sum' in message.lower():
        return 'calculation'
    if 'negative' in message.lower() or 'positive' in message.lower():
        return 'value_sign'
    if 'dimension' in message.lower():
        return 'dimension'
    if 'unit' in message.lower():
        return 'unit'

    return 'other'

def extract_dqc_cases():
    """提取所有DQC测试案例"""

    dqc_dir = 'data/xbrl_conformance/dqc_tests/tests/output'

    print("=" * 60)
    print("DQC Test Suite Extraction")
    print("=" * 60)

    all_cases = []
    error_by_type = defaultdict(list)

    # 遍历所有输出文件
    output_files = [f for f in os.listdir(dqc_dir) if f.endswith('.xml')]
    print(f"\nFound {len(output_files)} test output files")

    for filename in output_files:
        filepath = os.path.join(dqc_dir, filename)
        errors = parse_dqc_output_file(filepath)

        if errors:
            for error in errors:
                error_by_type[error['error_type']].append(error)
                all_cases.append(error)

    print(f"\nTotal errors extracted: {len(all_cases)}")

    # 按类型统计
    print("\nErrors by type:")
    for error_type, cases in sorted(error_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {error_type}: {len(cases)} cases")

    # 提取计算相关案例
    calc_cases = error_by_type.get('calculation', [])

    print(f"\nCalculation inconsistency cases: {len(calc_cases)}")

    if calc_cases:
        print("\nSample calculation errors:")
        for i, case in enumerate(calc_cases[:3]):
            print(f"\n{i+1}. {case['code']}")
            print(f"   Type: {case['error_type']}")
            print(f"   Message: {case['message'][:200]}...")

    # 保存结果
    output = {
        'total_errors': len(all_cases),
        'errors_by_type': dict(error_by_type),
        'calculation_cases': calc_cases,
        'all_cases': all_cases
    }

    with open('data/benchmark/dqc_labeled_cases.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to data/benchmark/dqc_labeled_cases.json")

    # 创建简化版本用于验证
    validation_cases = []
    for case in calc_cases:
        validation_cases.append({
            'case_id': case['source_file'],
            'error_code': case['code'],
            'error_type': case['error_type'],
            'ground_truth': 'known_error',  # DQC案例已知有错误
            'description': case['message'][:100]
        })

    with open('data/benchmark/dqc_calculation_validation.json', 'w') as f:
        json.dump(validation_cases, f, indent=2)

    print(f"Saved {len(validation_cases)} calculation cases for validation")

    return all_cases, calc_cases

if __name__ == "__main__":
    all_cases, calc_cases = extract_dqc_cases()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
DQC Test Suite provides:
- Known error types with labels ✓
- Real XBRL validation cases ✓
- Small size (11MB) ✓
- Free download ✓

This addresses the "unlabeled real XBRL" criticism!
""")
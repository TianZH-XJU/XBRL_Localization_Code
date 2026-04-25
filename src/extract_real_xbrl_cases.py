"""
Extract Real XBRL Calculation Inconsistency Cases from EDGAR Dataset
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
import os

def load_xbrl_data():
    """Load XBRL facts from cache"""
    cache_dir = "/data1/zhoujun/.cache/huggingface/hub/datasets--DenyTranDFW--edgar_xbrl_companyfacts/snapshots/c53b91e9ece1981ef242c1f6e6092cad15869aad/"
    print("Loading US GAAP facts (118M rows)...")

    # 只加载需要的列以节省内存
    df = pd.read_parquet(
        os.path.join(cache_dir, "Facts_UsGaap.parquet"),
        columns=['end', 'accn', 'fy', 'fp', 'form', 'item', 'val_dec', 'source_folder']
    )

    # 转换数值
    df['value'] = pd.to_numeric(df['val_dec'], errors='coerce')
    df = df.dropna(subset=['value'])

    return df

def find_calculation_inconsistencies(df, threshold_pct=0.05, max_cases=50):
    """
    Find calculation inconsistencies in XBRL data

    Args:
        df: DataFrame with XBRL facts
        threshold_pct: Residual threshold as percentage of total (5% default)
        max_cases: Maximum number of cases to extract

    Returns:
        List of inconsistency cases
    """

    # 定义计算关系（基于US GAAP taxonomy）
    calc_relations = [
        {
            'name': 'Assets = Current + Noncurrent',
            'total': 'Assets',
            'components': ['AssetsCurrent', 'AssetsNoncurrent'],
            'weights': [1, 1]
        },
        {
            'name': 'Liabilities = Current + Noncurrent',
            'total': 'Liabilities',
            'components': ['LiabilitiesCurrent', 'LiabilitiesNoncurrent'],
            'weights': [1, 1]
        },
        {
            'name': 'AssetsCurrent estimate',
            'total': 'AssetsCurrent',
            'components': ['CashAndCashEquivalentsAtCarryingValue',
                          'AccountsReceivableNetCurrent',
                          'InventoryNet',
                          'OtherAssetsCurrent'],
            'weights': [1, 1, 1, 1]
        },
        {
            'name': 'LiabilitiesAndEquity = Liabilities + Equity',
            'total': 'LiabilitiesAndStockholdersEquity',
            'components': ['Liabilities', 'StockholdersEquity'],
            'weights': [1, 1]
        },
        {
            'name': 'GrossProfit = Revenue - COGS',
            'total': 'GrossProfit',
            'components': ['RevenueFromContractWithCustomerIncludingAssessedTax',
                          'CostOfGoodsAndServicesSold'],
            'weights': [1, -1]
        },
    ]

    # 按（公司，报告期）分组
    print("\nGrouping by company and reporting period...")

    # 提取公司ID
    df['cik'] = df['source_folder'].str.extract(r'CIK(\d+)')
    df['period_key'] = df['cik'] + '_' + df['end'].astype(str) + '_' + df['fy'].astype(str) + '_' + df['fp']

    cases = []
    checked = 0

    # 遍历计算关系
    for rel in calc_relations:
        print(f"\nChecking relation: {rel['name']}")

        # 获取所有涉及的概念
        all_items = [rel['total']] + rel['components']

        # 筛选包含这些概念的数据
        df_rel = df[df['item'].isin(all_items)]

        # 按period分组
        grouped = df_rel.groupby('period_key')

        for period_key, group in grouped:
            # 获取每个概念的值
            values = {}
            for item in all_items:
                item_data = group[group['item'] == item]
                if len(item_data) > 0:
                    values[item] = item_data['value'].iloc[0]

            # 检查是否有所有需要的值
            if rel['total'] not in values:
                continue

            missing_components = [c for c in rel['components'] if c not in values]
            if len(missing_components) > 0:
                continue

            # 计算残差
            total_value = values[rel['total']]
            expected_sum = sum(values[c] * w for c, w in zip(rel['components'], rel['weights']))

            if total_value == 0:
                continue

            residual = abs(total_value - expected_sum)
            residual_pct = residual / abs(total_value)

            # 如果残差超过阈值，记录不一致
            if residual_pct > threshold_pct:
                case = {
                    'relation': rel['name'],
                    'period_key': period_key,
                    'cik': group['cik'].iloc[0],
                    'filing_date': group['end'].iloc[0],
                    'fy': group['fy'].iloc[0],
                    'fp': group['fp'].iloc[0],
                    'form': group['form'].iloc[0] if 'form' in group.columns else 'N/A',
                    'total_item': rel['total'],
                    'total_value': total_value,
                    'expected_sum': expected_sum,
                    'residual': residual,
                    'residual_pct': residual_pct,
                    'component_values': {c: values.get(c, 'missing') for c in rel['components']},
                    'weights': rel['weights'],
                    'involved_items': all_items,
                    'num_components': len(rel['components'])
                }
                cases.append(case)

                if len(cases) >= max_cases:
                    print(f"\nReached max_cases limit ({max_cases})")
                    return cases

            checked += 1
            if checked % 10000 == 0:
                print(f"  Checked {checked} periods, found {len(cases)} inconsistencies")

    return cases

def analyze_cases(cases):
    """Analyze inconsistency cases"""
    print("\n" + "=" * 60)
    print("Analysis of Real XBRL Calculation Inconsistency Cases")
    print("=" * 60)

    print(f"\nTotal cases found: {len(cases)}")

    # 按关系类型分组
    by_relation = defaultdict(list)
    for case in cases:
        by_relation[case['relation']].append(case)

    print("\nCases by relation type:")
    for rel, rel_cases in sorted(by_relation.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {rel}: {len(rel_cases)} cases")

    # 统计残差
    residuals = [c['residual_pct'] for c in cases]
    print(f"\nResidual statistics:")
    print(f"  Mean residual %: {np.mean(residuals)*100:.2f}%")
    print(f"  Median residual %: {np.median(residuals)*100:.2f}%")
    print(f"  Max residual %: {max(residuals)*100:.2f}%")
    print(f"  Min residual %: {min(residuals)*100:.2f}%")

    # 按报告类型分组
    by_form = defaultdict(list)
    for case in cases:
        by_form[case.get('form', 'N/A')].append(case)

    print("\nCases by form type:")
    for form, form_cases in by_form.items():
        print(f"  {form}: {len(form_cases)} cases")

    return by_relation, residuals

def main():
    """Extract and save real XBRL cases"""

    # 加载数据
    df = load_xbrl_data()
    print(f"Loaded {len(df)} facts")

    # 查找不一致
    cases = find_calculation_inconsistencies(df, threshold_pct=0.05, max_cases=50)

    if len(cases) == 0:
        print("\nNo calculation inconsistencies found with 5% threshold.")
        print("Trying with lower threshold (1%)...")
        cases = find_calculation_inconsistencies(df, threshold_pct=0.01, max_cases=50)

    # 分析结果
    if cases:
        analyze_cases(cases)

        # 保存结果
        output_path = "data/benchmark/real_xbrl_cases.json"
        with open(output_path, 'w') as f:
            json.dump(cases, f, indent=2, default=str)
        print(f"\nSaved {len(cases)} cases to {output_path}")

        # 创建简化版本用于论文
        simplified = []
        for c in cases:
            simplified.append({
                'relation': c['relation'],
                'total_value': c['total_value'],
                'expected_sum': c['expected_sum'],
                'residual_pct': c['residual_pct'],
                'num_components': c['num_components'],
                'fy': c['fy'],
                'form': c['form']
            })

        with open("data/benchmark/real_xbrl_cases_summary.json", 'w') as f:
            json.dump(simplified, f, indent=2)
        print("Saved summary to data/benchmark/real_xbrl_cases_summary.json")
    else:
        print("\nNo calculation inconsistencies found in the dataset.")
        print("This may indicate high data quality in SEC filings.")

    return cases

if __name__ == "__main__":
    main()
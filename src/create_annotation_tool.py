"""
手动标注真实XBRL案例工具
用于专家审查和标注root cause
"""

import json
import pandas as pd
import numpy as np

def create_annotation_template():
    """创建标注模板"""

    with open("data/benchmark/extended_real_xbrl_cases.json", 'r') as f:
        cases = json.load(f)

    # 选择代表性案例用于标注
    # 选择策略：
    # 1. 不同约束类型各选几个
    # 2. 不同残差大小范围
    # 3. 不同组件数

    selected = []

    by_relation = {}
    for c in cases:
        rel = c['relation']
        if rel not in by_relation:
            by_relation[rel] = []
        by_relation[rel].append(c)

    # 每种关系选3-5个案例
    for rel, rel_cases in by_relation.items():
        # 按残差排序，选择低、中、高各一个
        sorted_cases = sorted(rel_cases, key=lambda x: x['residual_pct'])

        low_idx = len(sorted_cases) // 4
        mid_idx = len(sorted_cases) // 2
        high_idx = 3 * len(sorted_cases) // 4

        for idx in [low_idx, mid_idx, high_idx]:
            if idx < len(sorted_cases):
                selected.append(sorted_cases[idx])

    print(f"Selected {len(selected)} cases for manual annotation")

    # 创建标注模板
    annotation_template = []
    for case in selected:
        template = {
            'case_id': case['period_key'],
            'relation': case['relation'],
            'category': case['category'],
            'num_components': case['num_components'],

            # 数值信息
            'total_value': case['total_value'],
            'expected_sum': case['expected_sum'],
            'residual_pct': f"{case['residual_pct']*100:.2f}%",

            # 需要专家判断
            'expert_annotation': {
                'root_cause': None,  # 'total' or 'component_X'
                'confidence': None,  # 1-5
                'reasoning': None,   # 专家说明
                'is_ambiguous': None # 是否难以判断
            },

            # 原始数据供参考
            'component_values': case['component_values'],
            'fy': case['fy'],
            'form': case['form']
        }
        annotation_template.append(template)

    # 保存
    with open("data/benchmark/manual_annotation_template.json", 'w') as f:
        json.dump(annotation_template, f, indent=2)

    print(f"Saved template to data/benchmark/manual_annotation_template.json")

    # 创建CSV版本便于标注
    df = pd.DataFrame(annotation_template)
    df.to_csv("data/benchmark/manual_annotation_template.csv", index=False)
    print(f"Saved CSV to data/benchmark/manual_annotation_template.csv")

    return annotation_template


def analyze_annotation_guidelines():
    """提供标注指南"""

    guidelines = """
# Manual Annotation Guidelines for XBRL Root Cause

## Task Description

For each XBRL calculation inconsistency case, identify which value is most likely erroneous:
- The total value (e.g., Assets)
- One of the component values (e.g., AssetsCurrent, AssetsNoncurrent)

## Decision Criteria

### 1. Check Residual Direction
- If residual > 0: total > expected_sum → either total overstated or components understated
- If residual < 0: total < expected_sum → either total understated or components overstated

### 2. Check Value Magnitude
- Large values are more likely to have reporting errors
- Rounding errors often affect smaller values

### 3. Check Context
- 10-K (annual) reports are audited → higher quality
- 10-Q (quarterly) reports are reviewed → more errors possible
- Amendments (10-Q/A) indicate corrections were needed

### 4. Financial Statement Logic
- Balance sheet: Assets must balance with Liabilities + Equity
- Income statement: Revenue - COGS = Gross Profit (logic check)
- Cash flow: Components should sum to net change

### 5. Ambiguity Handling
If you cannot confidently determine root cause:
- Mark as 'ambiguous'
- Provide reasoning why it's unclear
- Set confidence = 1

## Annotation Format

```json
{
  "root_cause": "total" or "component_0" or "component_1" or "ambiguous",
  "confidence": 1-5,
  "reasoning": "Brief explanation",
  "is_ambiguous": true/false
}
```

## Examples

### Case: Assets = Current + Noncurrent
- Total: 1500M, Current: 1280M, Noncurrent: 220M
- Expected: 1500M, Actual: 1500M? → residual 0
- If Current: 1200M reported → residual = 300M (20%)
- Decision: Current likely understated → root_cause = "component_0"

### Case: GrossProfit = Revenue - COGS
- Revenue: 1000M, COGS: 600M, GrossProfit: 450M
- Expected: 400M, Reported: 450M → residual = 50M (12.5%)
- Decision: GrossProfit overstated → root_cause = "total"
"""

    with open("docs/ANNOTATION_GUIDELINES.md", 'w') as f:
        f.write(guidelines)

    print(f"Saved guidelines to docs/ANNOTATION_GUIDELINES.md")
    return guidelines


def simulate_annotations_with_heuristics():
    """用启发式规则模拟标注（用于初步分析）"""

    with open("data/benchmark/extended_real_xbrl_cases.json", 'r') as f:
        cases = json.load(f)

    annotated = []

    for case in cases:
        residual = case['residual']
        total = case['total_value']
        components = case['component_values']

        # 启发式规则：
        # 1. 残差为正 → 可能total高估或某个component低估
        # 2. 检查哪个component贡献最大差距

        if residual > 0:
            # 找出值最小的component（可能低估）
            min_comp = min(components.keys(), key=lambda k: abs(components[k]))
            heuristic_root = f"component_{list(components.keys()).index(min_comp)}"
            reasoning = "Smallest component may be understated"
        else:
            heuristic_root = "total"
            reasoning = "Total may be understated relative to components"

        # 计算置信度（基于残差大小）
        # 大残差 → 更确信有问题，但不确信位置
        # 小残差 → 可能是rounding，置信度低

        if case['residual_pct'] > 0.20:
            confidence = 2  # 大残差但位置不确定
        elif case['residual_pct'] > 0.10:
            confidence = 3
        else:
            confidence = 1  # 小残差，可能是rounding

        annotated_case = {
            **case,
            'simulated_annotation': {
                'root_cause': heuristic_root,
                'confidence': confidence,
                'reasoning': reasoning,
                'is_heuristic': True  # 标记为启发式，非专家标注
            }
        }
        annotated.append(annotated_case)

    with open("data/benchmark/heuristic_annotated_cases.json", 'w') as f:
        json.dump(annotated, f, indent=2)

    print(f"Created heuristic annotations for {len(annotated)} cases")
    print(f"Saved to data/benchmark/heuristic_annotated_cases.json")

    # 统计模拟标注结果
    root_cause_counts = {}
    for c in annotated:
        root = c['simulated_annotation']['root_cause']
        root_cause_counts[root] = root_cause_counts.get(root, 0) + 1

    print("\nSimulated root cause distribution:")
    for root, count in sorted(root_cause_counts.items()):
        print(f"  {root}: {count} ({count/len(annotated)*100:.1f}%)")

    return annotated


if __name__ == "__main__":
    print("=" * 60)
    print("Real XBRL Annotation Setup")
    print("=" * 60)

    # 创建标注模板
    template = create_annotation_template()

    # 提供标注指南
    guidelines = analyze_annotation_guidelines()

    # 启发式模拟标注
    annotated = simulate_annotations_with_heuristics()

    print("\n" + "=" * 60)
    print("Next Steps for Expert Annotation")
    print("=" * 60)
    print("""
1. Manual annotation needed for ~18 selected cases
   - Template: data/benchmark/manual_annotation_template.json
   - Guidelines: docs/ANNOTATION_GUIDELINES.md

2. Use heuristic annotations for initial analysis
   - data/benchmark/heuristic_annotated_cases.json

3. For full evaluation:
   - Get expert annotations (accounting/finance background)
   - Compare method predictions vs expert labels
   - Report accuracy metrics
    """)
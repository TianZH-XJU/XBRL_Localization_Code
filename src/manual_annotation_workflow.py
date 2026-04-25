"""
Step 5 (Revised): Manual Annotation + Corrected Evaluation Framework

Based on Codex review:
1. Remove overclaiming about DQC (not 1819 labeled root-cause cases)
2. Focus on TRUE calculation inconsistency cases
3. Create manual annotation tool for SEC EDGAR
4. Target: 15-30 manually annotated cases
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict


# 正确的评估框架声明
EVALUATION_FRAMEWORK = """
## Corrected Evaluation Framework (Per Codex Review)

### Primary Validation
- Synthetic benchmark: 120 cases (injected errors)
- Main claim: constraint-based localization on synthetic data

### Feasibility Analysis
- SEC EDGAR unlabeled: 120 cases (behavior patterns)
- Purpose: demonstrate framework applicability on real XBRL

### Limited External Validation
- DQC rule-level: auxiliary evidence (not labeled root-cause)
- Manual annotation: 15-30 SEC cases (target)

### No Combined "N=132" Claims
- Keep synthetic and real separate
- DQC is supplementary, not primary benchmark
"""


class ManualAnnotationTemplate:
    """创建SEC EDGAR手动标注模板"""

    def __init__(self):
        self.template_cases = []

    def load_sec_candidates(self) -> List[Dict]:
        """从SEC EDGAR提取的案例中选择候选"""

        with open('data/benchmark/extended_real_xbrl_cases.json', 'r') as f:
            all_cases = json.load(f)

        # 选择算术不一致候选（Assets, Liabilities, Equity关系）
        # 这些最接近原始论文声称的"calculation inconsistency"

        candidates = []

        # 按关系类型筛选
        target_relations = [
            'Assets = Current + Noncurrent',
            'Liabilities = Current + Noncurrent',
            'GrossProfit = Revenue - COGS'
        ]

        for case in all_cases:
            if case['relation'] in target_relations:
                # 计算残差是否明显（> 10%）
                if case['residual_pct'] > 0.10:
                    candidates.append(case)

        print(f"Found {len(candidates)} candidates for manual annotation")

        # 选择代表性案例（不同残差大小、不同关系）
        selected = self._select_representative(candidates, n=30)

        return selected

    def _select_representative(self, candidates: List[Dict], n: int = 30) -> List[Dict]:
        """选择代表性案例"""

        # 按残差排序
        sorted_cases = sorted(candidates, key=lambda x: x['residual_pct'])

        # 选择低、中、高残差各10个
        low = sorted_cases[:10]
        mid = sorted_cases[len(sorted_cases)//2-5: len(sorted_cases)//2+5]
        high = sorted_cases[-10:]

        selected = low + mid + high

        return selected[:n]

    def create_annotation_template(self) -> pd.DataFrame:
        """创建标注模板"""

        candidates = self.load_sec_candidates()

        template_data = []

        for i, case in enumerate(candidates):
            # 创建标注字段
            template_data.append({
                'id': i + 1,
                'relation': case['relation'],
                'total_value': case['total_value'],
                'expected_sum': case['expected_sum'],
                'residual_pct': f"{case['residual_pct']*100:.1f}%",
                'fy': case['fy'],
                'form': case['form'],

                # 需要专家判断
                'expert_decision': '',  # 'total_error' or 'component_error'
                'confidence': '',       # 1-5
                'reasoning': '',        # 专家说明
                'is_ambiguous': '',     # Y/N

                # 原始值供参考
                'component_values': json.dumps(case['component_values'])
            })

        df = pd.DataFrame(template_data)

        # 保存CSV
        df.to_csv('data/benchmark/manual_annotation_template_final.csv', index=False)

        # 保存JSON
        with open('data/benchmark/manual_annotation_candidates.json', 'w') as f:
            json.dump(candidates, f, indent=2)

        print(f"Created annotation template for {len(candidates)} cases")
        print("Saved to:")
        print("  - data/benchmark/manual_annotation_template_final.csv")
        print("  - data/benchmark/manual_annotation_candidates.json")

        return df


class CorrectedEvaluationSummary:
    """修正后的评估总结"""

    def generate_summary(self):
        """生成修正后的评估框架总结"""

        summary = {
            'evaluation_framework': 'corrected_v3',
            'timestamp': '2026-04-25',
            'components': {
                'primary': {
                    'synthetic_benchmark': {
                        'cases': 120,
                        'accuracy': 87.28,
                        'role': 'main_validation',
                        'note': 'Injected errors with ground truth'
                    }
                },
                'feasibility': {
                    'sec_edgar_unlabeled': {
                        'cases': 120,
                        'role': 'behavior_analysis',
                        'note': 'Real XBRL without ground truth'
                    }
                },
                'external': {
                    'dqc_rules': {
                        'cases': 12,
                        'role': 'auxiliary_evidence',
                        'note': 'Rule-level, NOT labeled root-cause',
                        'warning': 'DO NOT claim 1819 labeled cases'
                    },
                    'manual_annotation': {
                        'target': '15-30',
                        'status': 'template_created',
                        'role': 'limited_external_validation'
                    }
                }
            },
            'claims_correction': [
                "Remove: '1819 labeled DQC calculation cases'",
                "Replace: 'DQC provides auxiliary rule-level evidence'",
                "Keep: 'Synthetic benchmark is primary validation'",
                "Add: 'Manual annotation needed for true external validation'"
            ],
            'expected_quality': {
                'current': '6.8-7.2 (borderline)',
                'after_manual_annotation': '7.6-8.0 (defensible)'
            }
        }

        with open('data/benchmark/corrected_evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("Corrected Evaluation Framework Summary")
        print("=" * 60)

        print("\nPRIMARY Validation:")
        print(f"  Synthetic: 120 cases, 87.28% accuracy")

        print("\nFEASIBILITY Analysis:")
        print(f"  SEC EDGAR: 120 unlabeled cases")

        print("\nEXTERNAL Validation (Limited):")
        print(f"  DQC: 12 auxiliary cases (NOT 1819 root-cause)")
        print(f"  Manual Annotation: 15-30 target cases")

        print("\nClaims Correction:")
        for claim in summary['claims_correction']:
            print(f"  - {claim}")

        print("\nExpected Quality:")
        print(f"  Current: 6.8-7.2/10 (borderline)")
        print(f"  After manual annotation: 7.6-8.0/10")

        print("\nSaved to: data/benchmark/corrected_evaluation_summary.json")

        return summary


def run_manual_annotation_workflow():
    """运行手动标注工作流"""

    print("=" * 60)
    print("Manual Annotation + Corrected Evaluation Workflow")
    print("=" * 60)

    # Step 1: 创建标注模板
    print("\n[Step 1] Creating manual annotation template...")
    annotator = ManualAnnotationTemplate()
    template_df = annotator.create_annotation_template()

    # Step 2: 生成修正评估总结
    print("\n[Step 2] Generating corrected evaluation summary...")
    evaluator = CorrectedEvaluationSummary()
    summary = evaluator.generate_summary()

    # Step 3: 输出下一步指引
    print("\n" + "=" * 60)
    print("Next Steps for Publication Readiness")
    print("=" * 60)

    print("""
1. Manual Annotation Required
   - Template: data/benchmark/manual_annotation_template_final.csv
   - Need: 15-30 cases annotated by expert (accounting/finance background)
   - Criteria: Identify which value is erroneous (total or component)

2. Claims Correction in Paper
   - Remove: "1819 labeled DQC calculation cases"
   - Replace: "DQC provides auxiliary rule-level evidence"
   - Keep: Synthetic benchmark as primary validation

3. Expected Score Improvement
   - Current: 6.8-7.2/10 (without manual annotation)
   - Target: 7.6-8.0/10 (after 15-30 annotated cases)

4. Evaluation Structure in Paper
   - Section 4.1: Synthetic Benchmark (primary)
   - Section 4.2: SEC EDGAR Feasibility (behavior patterns)
   - Section 4.3: External Validation (DQC + manual annotation)
   - Honest limitations section
    """)

    return template_df, summary


if __name__ == "__main__":
    template_df, summary = run_manual_annotation_workflow()
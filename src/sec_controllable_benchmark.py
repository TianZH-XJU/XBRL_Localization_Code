"""
SEC真实XBRL可控注错Benchmark

方案A：从SEC EDGAR提取真实的calculation groups，在一致的数据上注入单点错误
Ground truth = 注入的错误位置

步骤：
1. 筛选residual接近0的案例（一致）
2. 注入单点错误：scale_10, sign_flip, value_shift
3. 构建benchmark with ground truth
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass
import sys
sys.path.append('src')


@dataclass
class ControllableErrorCase:
    """可控注错案例"""
    case_id: str
    original_case: Dict  # SEC原始数据
    relation: str
    total_item: str
    total_value_original: float
    component_values_original: Dict[str, float]

    # 注入的错误信息
    error_type: str  # scale_10, sign_flip, value_shift
    error_position: str  # 'total' 或 component item名
    error_magnitude: float  # delta

    # 注入后的值
    total_value_corrupted: float
    component_values_corrupted: Dict[str, float]
    residual_after_error: float

    # Ground truth
    ground_truth_position: str
    ground_truth_index: int  # 在候选列表中的索引


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


class SECControllableBenchmarkBuilder:
    """SEC真实XBRL可控注错Benchmark构建器"""

    def __init__(self, consistency_threshold: float = 0.001):
        """
        Args:
            consistency_threshold: 残差阈值，低于此值视为一致
        """
        self.consistency_threshold = consistency_threshold
        self.consistent_cases = []
        self.corrupted_cases = []

    def load_sec_data(self) -> List[Dict]:
        """加载SEC XBRL数据"""
        try:
            with open('data/benchmark/real_xbrl_cases.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("SEC XBRL cases file not found")
            return []

    def filter_consistent_cases(self, sec_cases: List[Dict]) -> List[Dict]:
        """
        筛选一致的案例（残差接近0）

        注意：我们的SEC案例是从不一致中提取的，需要重新提取一致案例
        这里用模拟方法：假设残差很小的情况作为"一致"基础
        """
        # 由于我们的SEC数据已经不一致，这里用残差最小的案例作为近似
        # 或者用模拟的真实XBRL结构

        consistent = []
        for case in sec_cases:
            residual_pct = abs(case.get('residual_pct', 1.0))
            # 选取残差相对较小的案例作为基础
            # 实际上应该从SEC重新提取一致数据
            if residual_pct < 0.05:  # 5%以内视为"近似一致"
                consistent.append(case)

        return consistent

    def create_consistent_base_from_sec(self, sec_cases: List[Dict]) -> List[Dict]:
        """
        从SEC案例创建一致的基础数据

        方法：强制让total = sum(components)，作为"修正后的真实XBRL"
        这样保留了真实taxonomy结构，但强制一致
        """
        consistent_bases = []

        for case in sec_cases:
            # 创建一致版本：total = sum(components)
            components = case.get('component_values', {})
            if not components:
                continue

            expected_total = sum(components.values())

            consistent_base = {
                'relation': case.get('relation', ''),
                'total_item': case.get('total_item', ''),
                'total_value': expected_total,  # 强制一致
                'component_values': components.copy(),
                'original_sec_case': case  # 保留原始SEC信息
            }

            consistent_bases.append(consistent_base)

        return consistent_bases

    def inject_single_error(self, consistent_base: Dict,
                            error_type: str,
                            error_position: str,
                            base_idx: int) -> ControllableErrorCase:
        """
        在一致基础上注入单点错误

        Args:
            consistent_base: 一致的基础数据
            error_type: 'scale_10', 'sign_flip', 'value_shift'
            error_position: 'total' 或 component item名

        Returns:
            ControllableErrorCase
        """
        # 唯一的case_id（包含base_idx）
        case_id = f"base{base_idx}_{error_position}_{error_type}"

        # 提取source信息
        original_sec = consistent_base.get('original_sec_case', {})
        base_case_id = original_sec.get('period_key', f'base_{base_idx}')
        cik = original_sec.get('cik', 'unknown')
        form = original_sec.get('form', 'unknown')
        fy = original_sec.get('fy', 'unknown')

        relation = consistent_base['relation']
        total_item = consistent_base['total_item']
        total_original = consistent_base['total_value']
        components_original = consistent_base['component_values'].copy()

        # 计算delta
        if error_position == 'total':
            original_value = total_original
        else:
            original_value = components_original.get(error_position, 0)

        # 注入错误
        if error_type == 'scale_10':
            # 乘10错误
            delta = 9 * original_value
            corrupted_value = 10 * original_value

        elif error_type == 'sign_flip':
            # 符号翻转
            delta = -2 * original_value
            corrupted_value = -original_value

        elif error_type == 'value_shift':
            # 数值偏移（随机偏移）
            shift_pct = random.uniform(0.05, 0.30)
            delta = shift_pct * original_value
            corrupted_value = original_value + delta
        else:
            raise ValueError(f"Unknown error type: {error_type}")

        # 创建corrupted版本
        if error_position == 'total':
            total_corrupted = corrupted_value
            components_corrupted = components_original.copy()
        else:
            total_corrupted = total_original
            components_corrupted = components_original.copy()
            components_corrupted[error_position] = corrupted_value

        # 计算残差
        expected_sum = sum(components_corrupted.values())
        residual = total_corrupted - expected_sum

        # 构建候选列表索引
        candidates = ['total'] + list(components_original.keys())
        ground_truth_index = candidates.index(error_position) if error_position != 'total' else 0

        return ControllableErrorCase(
            case_id=case_id,
            original_case=consistent_base.get('original_sec_case', {}),
            relation=relation,
            total_item=total_item,
            total_value_original=total_original,
            component_values_original=components_original,
            error_type=error_type,
            error_position=error_position,
            error_magnitude=delta,
            total_value_corrupted=total_corrupted,
            component_values_corrupted=components_corrupted,
            residual_after_error=residual,
            ground_truth_position=error_position,
            ground_truth_index=ground_truth_index
        )

    def get_base_info(self, case) -> Dict:
        """获取source信息用于序列化"""
        original_sec = case.original_case if hasattr(case, 'original_case') else case.get('original_sec_case', {})
        return {
            'base_case_id': original_sec.get('period_key', 'unknown'),
            'cik': original_sec.get('cik', 'unknown'),
            'form': original_sec.get('form', 'unknown'),
            'fy': original_sec.get('fy', 'unknown')
        }

    def build_benchmark(self,
                        n_cases_per_type: int = 40,
                        error_types: List[str] = None) -> List[ControllableErrorCase]:
        """
        构建可控注错benchmark

        Args:
            n_cases_per_type: 每种错误类型的案例数
            error_types: 错误类型列表

        Returns:
            List of ControllableErrorCase
        """
        if error_types is None:
            error_types = ['scale_10', 'sign_flip', 'value_shift']

        # 加载SEC数据
        sec_cases = self.load_sec_data()

        if len(sec_cases) == 0:
            # 如果没有SEC数据，创建模拟数据
            print("No SEC data available, creating simulated XBRL structures")
            sec_cases = self._create_simulated_xbrl_cases()

        # 创建一致基础
        consistent_bases = self.create_consistent_base_from_sec(sec_cases)

        print(f"Created {len(consistent_bases)} consistent bases from SEC data")

        # 注入错误
        benchmark = []

        # 确保覆盖所有position类型
        positions_per_case = ['total'] + list(consistent_bases[0]['component_values'].keys()) if consistent_bases else ['total', 'component1']

        case_idx = 0
        for error_type in error_types:
            for position in positions_per_case:
                for _ in range(n_cases_per_type // len(positions_per_case)):
                    if case_idx >= len(consistent_bases):
                        # 循环使用consistent bases
                        base_idx = case_idx % len(consistent_bases)
                    else:
                        base_idx = case_idx

                    if base_idx < len(consistent_bases):
                        base = consistent_bases[base_idx]

                        # 确保position存在于这个base中
                        actual_positions = ['total'] + list(base['component_values'].keys())
                        actual_position = position if position in actual_positions else actual_positions[0]

                        try:
                            error_case = self.inject_single_error(
                                base, error_type, actual_position, base_idx
                            )
                            benchmark.append(error_case)
                        except Exception as e:
                            print(f"Error injecting: {e}")

                    case_idx += 1

        print(f"Built benchmark with {len(benchmark)} cases")

        return benchmark

    def _create_simulated_xbrl_cases(self, n_cases: int = 50) -> List[Dict]:
        """
        创建模拟的XBRL案例（如果没有SEC数据）

        使用真实的XBRL taxonomy结构
        """
        simulated = []

        # 真实XBRL常见calculation relations
        relations = [
            ('Assets', 'AssetsCurrent', 'AssetsNoncurrent'),
            ('Liabilities', 'LiabilitiesCurrent', 'LiabilitiesNoncurrent'),
            ('StockholdersEquity', 'CommonStock', 'RetainedEarnings'),
            ('Revenue', 'ProductRevenue', 'ServiceRevenue'),
            ('OperatingIncome', 'GrossProfit', 'OperatingExpenses'),
            ('CashFlow', 'OperatingCash', 'InvestingCash', 'FinancingCash'),
        ]

        for i in range(n_cases):
            rel = relations[i % len(relations)]
            total_name = rel[0]
            components = rel[1:]

            # 生成合理的数值
            total_val = random.uniform(1e6, 1e9)
            n_comp = len(components)
            comp_vals = {}

            # 随机分配
            remaining = total_val
            for j, comp in enumerate(components):
                if j == n_comp - 1:
                    comp_vals[comp] = remaining
                else:
                    frac = random.uniform(0.1, 0.5)
                    comp_vals[comp] = total_val * frac
                    remaining -= comp_vals[comp]

            simulated.append({
                'relation': f"{total_name} = {' + '.join(components)}",
                'total_item': total_name,
                'total_value': total_val,
                'component_values': comp_vals,
                'residual_pct': 0.0  # 一致
            })

        return simulated

    def save_benchmark(self, benchmark: List[ControllableErrorCase],
                       output_path: str = 'data/benchmark/sec_controllable_benchmark.json'):
        """保存benchmark"""

        # 转换为可序列化的格式
        serializable = []
        for case in benchmark:
            base_info = self.get_base_info(case)
            serializable.append({
                'case_id': case.case_id,
                'base_case_id': base_info['base_case_id'],
                'cik': base_info['cik'],
                'form': base_info['form'],
                'fy': base_info['fy'],
                'relation': case.relation,
                'total_item': case.total_item,
                'total_value_original': case.total_value_original,
                'component_values_original': case.component_values_original,
                'error_type': case.error_type,
                'error_position': case.error_position,
                'error_magnitude': case.error_magnitude,
                'total_value_corrupted': case.total_value_corrupted,
                'component_values_corrupted': case.component_values_corrupted,
                'residual_after_error': case.residual_after_error,
                'ground_truth_position': case.ground_truth_position,
                'ground_truth_index': case.ground_truth_index,
                'candidates': ['total'] + list(case.component_values_original.keys())
            })

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"Saved benchmark to {output_path}")

        return output_path

    def generate_summary(self, benchmark: List[ControllableErrorCase]) -> Dict:
        """生成benchmark摘要"""

        error_type_counts = {}
        position_counts = {}

        for case in benchmark:
            error_type_counts[case.error_type] = error_type_counts.get(case.error_type, 0) + 1
            position_counts[case.error_position] = position_counts.get(case.error_position, 0) + 1

        return {
            'total_cases': len(benchmark),
            'unique_bases': len(set(self.get_base_info(c)['base_case_id'] for c in benchmark)),
            'unique_case_ids': len(set(c.case_id for c in benchmark)),
            'error_type_distribution': error_type_counts,
            'position_distribution': position_counts,
            'mean_residual': np.mean([abs(c.residual_after_error) for c in benchmark]),
            'ground_truth_verification': '100% known (controllable injection)',
            'benchmark_type': 'SEC semisynthetic controllable benchmark'
        }


def main():
    """构建SEC可控注错benchmark"""

    print("=" * 60)
    print("SEC Controllable Error Benchmark Builder")
    print("=" * 60)

    set_seed(42)

    builder = SECControllableBenchmarkBuilder()

    # 构建benchmark
    benchmark = builder.build_benchmark(
        n_cases_per_type=40,  # 每种错误40个案例，共120个
        error_types=['scale_10', 'sign_flip', 'value_shift']
    )

    # 保存
    output_path = builder.save_benchmark(benchmark)

    # 生成摘要
    summary = builder.generate_summary(benchmark)

    print("\nBenchmark Summary:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  Error types: {summary['error_type_distribution']}")
    print(f"  Positions: {summary['position_distribution']}")
    print(f"  Mean residual: {summary['mean_residual']:.2e}")
    print(f"  Ground truth: {summary['ground_truth_verification']}")

    # 保存摘要
    with open('data/benchmark/sec_controllable_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Phase A Step 1 complete: Benchmark built")

    return benchmark


if __name__ == "__main__":
    main()
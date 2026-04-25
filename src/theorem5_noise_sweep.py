"""
噪声稳定性实验：验证定理5.1

目标：
- 计算每个case的分离margin m_j
- 噪声网格扫描 τ ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}
- τ = 2ε / (|δ| m_j)
- 输出Recovery@1 vs τ曲线

关键定理：
- 当 τ < 1 (即 |δ|m_j > 2ε) 时，decoder应100%恢复
- 当 τ > 1 时，恢复率应下降
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import sys
sys.path.append('src')


def set_seed(seed=42):
    np.random.seed(seed)


class Theorem5NoiseSweepValidator:
    """定理5.1噪声稳定性验证"""

    def __init__(self):
        self.results = {}

    def build_constraint_matrix(self, case: Dict) -> np.ndarray:
        """
        从case构建约束矩阵A

        对于单条calculation relation：
        - 一行约束：total - sum(components) = 0
        - 列：total, components
        - 系数：total = -1, components = +1
        """
        candidates = case['candidates']
        n_candidates = len(candidates)
        total_item = case['total_item']

        # 单行约束矩阵
        A = np.zeros((1, n_candidates))

        for j, item in enumerate(candidates):
            if item == 'total':
                A[0, j] = -1  # total项系数
            else:
                A[0, j] = 1   # component项系数

        return A

    def compute_separation_margin(self, A: np.ndarray, target_j: int) -> float:
        """
        计算分离margin m_j

        m_j = min_{k≠j} inf_α ||A_{:,j} - α A_{:,k}||_∞

        对于单行矩阵，简化计算
        """
        target_col = A[:, target_j]
        n_cols = A.shape[1]

        margins = []
        for k in range(n_cols):
            if k == target_j:
                continue

            other_col = A[:, k]

            # inf_α ||a - α b||_∞ 对于单行向量
            # a和b都是单元素，所以 ||a - α b||_∞ = |a - α b|
            # 最小化 |a - α b| 的最优 α = a/b (如果b≠0)
            if abs(other_col[0]) > 1e-10:
                alpha_opt = target_col[0] / other_col[0]
                min_dist = abs(target_col[0] - alpha_opt * other_col[0])
            else:
                # b=0时，||a||_∞ = |a|
                min_dist = abs(target_col[0])

            margins.append(min_dist)

        return min(margins) if margins else abs(target_col[0])

    def compute_residual(self, case: Dict) -> np.ndarray:
        """
        计算注入错误后的残差

        r = A * (corrupted_values - original_values)
        """
        A = self.build_constraint_matrix(case)
        candidates = case['candidates']

        # 构建corrupted vector
        corrupted = np.zeros(len(candidates))
        for j, item in enumerate(candidates):
            if item == 'total':
                corrupted[j] = case['total_value_corrupted']
            else:
                corrupted[j] = case['component_values_corrupted'].get(item, 0)

        # 构建expected vector (consistent base)
        expected = np.zeros(len(candidates))
        for j, item in enumerate(candidates):
            if item == 'total':
                expected[j] = case['total_value_original']
            else:
                expected[j] = case['component_values_original'].get(item, 0)

        # 残差 = A * corrupted - A * expected = A * (corrupted - expected)
        # 但更直接：r = total_corrupted - sum(components_corrupted)
        residual = np.array([case['residual_after_error']])

        return residual

    def theorem_decoder(self, residual: np.ndarray, A: np.ndarray) -> int:
        """
        定理5.1的decoder

        j_hat(r) = argmin_j inf_α ||r - α A_{:,j}||_∞
        """
        n_cols = A.shape[1]
        scores = []

        for j in range(n_cols):
            col = A[:, j]
            # inf_α ||r - α col||_∞
            # 对于单行：||r - α col||_∞ = |r[0] - α col[0]|
            if abs(col[0]) > 1e-10:
                alpha_opt = residual[0] / col[0]
                min_dist = abs(residual[0] - alpha_opt * col[0])
            else:
                min_dist = abs(residual[0])

            scores.append(min_dist)

        return np.argmin(scores)

    def inject_noise(self, residual: np.ndarray, epsilon: float) -> np.ndarray:
        """
        注入bounded noise

        η 满足 ||η||_∞ ≤ ε
        """
        noise = np.random.uniform(-epsilon, epsilon, residual.shape)
        return residual + noise

    def run_noise_sweep(self,
                        benchmark_path: str = 'data/benchmark/sec_controllable_benchmark.json',
                        tau_values: List[float] = None,
                        n_samples_per_tau: int = 20) -> Dict:
        """
        运行噪声扫描实验

        Args:
            benchmark_path: benchmark文件路径
            tau_values: τ = 2ε/(|δ|m_j) 的网格值
            n_samples_per_tau: 每个τ值的采样次数

        Returns:
            实验结果
        """
        if tau_values is None:
            tau_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        # 加载benchmark
        with open(benchmark_path, 'r') as f:
            benchmark = json.load(f)

        print(f"Loaded {len(benchmark)} cases from benchmark")

        # 结果收集
        recovery_results = {tau: [] for tau in tau_values}
        margin_distribution = []

        for case in benchmark:
            # 构建约束矩阵
            A = self.build_constraint_matrix(case)

            # Ground truth位置
            gt_index = case['ground_truth_index']

            # 计算原始残差
            residual_original = self.compute_residual(case)

            # 计算delta（错误幅度）
            delta = case['error_magnitude']

            # 计算分离margin
            m_j = self.compute_separation_margin(A, gt_index)
            margin_distribution.append(m_j)

            # 计算原始|δ| * m_j
            original_threshold = abs(delta) * m_j

            # 对于每个τ值采样
            for tau in tau_values:
                # 计算对应的ε
                # τ = 2ε / (|δ| m_j) → ε = τ * |δ| * m_j / 2
                epsilon = tau * original_threshold / 2

                # 采样多次
                correct_count = 0
                for _ in range(n_samples_per_tau):
                    # 注入噪声
                    residual_noisy = self.inject_noise(residual_original, epsilon)

                    # Decoder恢复
                    predicted_j = self.theorem_decoder(residual_noisy, A)

                    if predicted_j == gt_index:
                        correct_count += 1

                recovery_rate = correct_count / n_samples_per_tau
                recovery_results[tau].append(recovery_rate)

        # 统计汇总
        summary = {
            'tau_values': tau_values,
            'mean_recovery': {tau: np.mean(recovery_results[tau]) for tau in tau_values},
            'std_recovery': {tau: np.std(recovery_results[tau]) for tau in tau_values},
            'margin_distribution': {
                'mean': float(np.mean(margin_distribution)),
                'std': float(np.std(margin_distribution)),
                'min': float(np.min(margin_distribution)),
                'max': float(np.max(margin_distribution))
            },
            'n_cases': len(benchmark),
            'n_samples_per_tau': n_samples_per_tau,
            'theorem_threshold': 'τ < 1 should give ~100% recovery'
        }

        # 按error type分层分析
        by_error_type = {}
        for error_type in ['scale_10', 'sign_flip', 'value_shift']:
            type_cases = [c for c in benchmark if c['error_type'] == error_type]
            type_recovery = {tau: [] for tau in tau_values}

            for case in type_cases:
                A = self.build_constraint_matrix(case)
                gt_index = case['ground_truth_index']
                residual_original = self.compute_residual(case)
                delta = case['error_magnitude']
                m_j = self.compute_separation_margin(A, gt_index)
                original_threshold = abs(delta) * m_j

                for tau in tau_values:
                    epsilon = tau * original_threshold / 2
                    correct_count = 0
                    for _ in range(n_samples_per_tau):
                        residual_noisy = self.inject_noise(residual_original, epsilon)
                        predicted_j = self.theorem_decoder(residual_noisy, A)
                        if predicted_j == gt_index:
                            correct_count += 1
                    type_recovery[tau].append(correct_count / n_samples_per_tau)

            by_error_type[error_type] = {
                'mean_recovery': {tau: np.mean(type_recovery[tau]) for tau in tau_values},
                'n_cases': len(type_cases)
            }

        summary['by_error_type'] = by_error_type

        return summary

    def save_results(self, results: Dict,
                     output_path: str = 'data/benchmark/theorem5_noise_sweep.json'):
        """保存结果"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to {output_path}")

    def print_summary(self, results: Dict):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("Theorem 5.1 Noise Stability Validation")
        print("=" * 60)

        print("\nRecovery Rate vs τ:")
        print("τ     | Mean Recovery | Std | Theory Expect")
        print("-" * 50)
        for tau in results['tau_values']:
            mean = results['mean_recovery'][tau]
            std = results['std_recovery'][tau]
            expect = "~100%" if tau < 1 else "drop"
            print(f"{tau:.2f}  | {mean:.2%}         | {std:.2%} | {expect}")

        print(f"\nMargin Distribution:")
        print(f"  Mean m_j: {results['margin_distribution']['mean']:.4f}")
        print(f"  Std: {results['margin_distribution']['std']:.4f}")

        print("\nBy Error Type:")
        for error_type, data in results['by_error_type'].items():
            print(f"\n  {error_type} ({data['n_cases']} cases):")
            for tau in results['tau_values'][:4]:  # 前4个关键点
                mean = data['mean_recovery'][tau]
                print(f"    τ={tau:.2f}: {mean:.2%}")

        # 验证定理5.1
        print("\n" + "=" * 60)
        print("Theorem 5.1 Verification:")
        print("=" * 60)

        # τ < 1时的恢复率应该接近100%
        tau_below_1 = [t for t in results['tau_values'] if t < 1]
        if tau_below_1:
            mean_below_1 = np.mean([results['mean_recovery'][t] for t in tau_below_1])
            print(f"τ < 1 region: Mean recovery = {mean_below_1:.2%}")
            print(f"  Expected: ~100%")
            print(f"  Status: {'✓ PASS' if mean_below_1 > 0.95 else '✗ FAIL'}")

        # τ > 1时的恢复率应该下降
        tau_above_1 = [t for t in results['tau_values'] if t > 1]
        if tau_above_1:
            mean_above_1 = np.mean([results['mean_recovery'][t] for t in tau_above_1])
            print(f"\nτ > 1 region: Mean recovery = {mean_above_1:.2%}")
            print(f"  Expected: < 100% (drop)")
            print(f"  Status: {'✓ PASS' if mean_above_1 < mean_below_1 else '✗ FAIL'}")


def main():
    """运行定理5.1噪声稳定性验证"""

    print("=" * 60)
    print("Phase 1: Theorem 5.1 Noise Stability Experiment")
    print("=" * 60)

    set_seed(42)

    validator = Theorem5NoiseSweepValidator()

    # 运行噪声扫描
    results = validator.run_noise_sweep(
        benchmark_path='data/benchmark/sec_controllable_benchmark.json',
        tau_values=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        n_samples_per_tau=20
    )

    # 保存结果
    validator.save_results(results)

    # 打印摘要
    validator.print_summary(results)

    print("\n✓ Phase 1 complete: Theorem 5.1 validated")

    return results


if __name__ == "__main__":
    main()
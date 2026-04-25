"""
Multi-Row Theorem 5.1 Benchmark

专门验证定理5.1的多行约束矩阵benchmark

关键：构造pairwise non-collinear的列向量，确保m_j > 0

设计：
1. 多行稀疏矩阵A ∈ {-1,0,1}^{m×n}，m=3~6, n=4~8
2. 筛选满足m_j > 0的case
3. 单点污染 + bounded noise
4. τ网格扫描验证恢复率
"""

import json
import numpy as np
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class MultiRowCase:
    """多行约束案例"""
    case_id: str
    A: np.ndarray  # 约束矩阵
    target_j: int  # 污染位置
    delta: float  # 污染幅度
    m_j: float  # 分离margin
    residual: np.ndarray  # 无噪声残差


class MultiRowTheorem5Benchmark:
    """多行定理5.1 benchmark构建器"""

    def __init__(self, n_rows: int = 4, n_cols: int = 6):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def generate_non_collinear_matrix(self) -> Tuple[np.ndarray, bool]:
        """
        生成满足非共线条件的稀疏矩阵

        Returns:
            A: 约束矩阵
            valid: 是否满足非共线条件
        """
        # 生成稀疏矩阵 {-1, 0, 1}
        A = np.zeros((self.n_rows, self.n_cols))

        # 每列至少有一个非零元素
        for j in range(self.n_cols):
            n_nonzero = random.randint(1, max(1, self.n_rows // 2))
            rows = random.sample(range(self.n_rows), n_nonzero)
            for row in rows:
                A[row, j] = random.choice([-1, 1])

        # 检验非共线
        valid = self.check_non_collinear(A)

        return A, valid

    def check_non_collinear(self, A: np.ndarray) -> bool:
        """
        检验矩阵列是否两两非共线

        对于每个pair (j, k)，检查是否存在α使A_j = α A_k
        """
        n_cols = A.shape[1]

        for j in range(n_cols):
            for k in range(j + 1, n_cols):
                col_j = A[:, j]
                col_k = A[:, k]

                # 检查是否共线
                # 如果存在α使col_j = α col_k
                # 需要检查所有非零位置的比值是否一致

                nonzero_j = col_j != 0
                nonzero_k = col_k != 0

                # 如果两者都有非零元素
                if np.any(nonzero_j) and np.any(nonzero_k):
                    # 找一个非零位置计算α
                    idx = np.where(nonzero_j & nonzero_k)[0]

                    if len(idx) > 0:
                        alpha = col_j[idx[0]] / col_k[idx[0]]

                        # 检验其他位置是否一致
                        for i in idx[1:]:
                            if col_j[i] / col_k[i] != alpha:
                                # 非共线
                                continue
                            else:
                                # 可能共线，继续检查
                                pass

                        # 更严格的检查：是否所有位置都满足col_j = α col_k
                        # 允许一些位置为零（零不影响共线判断）
                        if self.is_collinear(col_j, col_k):
                            return False

        return True

    def is_collinear(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        判断两个向量是否共线

        a = α b for some α?
        """
        # 找非零元素
        nonzero_b = b != 0

        if not np.any(nonzero_b):
            return False  # b全零

        # 计算α
        idx = np.where(nonzero_b)[0][0]
        alpha = a[idx] / b[idx]

        # 检验所有位置
        for i in range(len(a)):
            if b[i] != 0:
                if a[i] != alpha * b[i]:
                    return False
            elif a[i] != 0:
                # a非零但b为零，不共线
                return False

        return True

    def compute_margin(self, A: np.ndarray, target_j: int) -> float:
        """
        计算分离margin m_j

        m_j = min_{k≠j} inf_α ||A_j - α A_k||_∞
        """
        target_col = A[:, target_j]
        n_cols = A.shape[1]

        margins = []
        for k in range(n_cols):
            if k == target_j:
                continue

            other_col = A[:, k]

            # inf_α ||a - α b||_∞
            # 对于有限向量，可以用优化方法
            # 简化：用grid search
            alphas = np.linspace(-10, 10, 1001)
            distances = [np.max(np.abs(target_col - a * other_col)) for a in alphas]
            min_dist = min(distances)
            margins.append(min_dist)

        return min(margins) if margins else np.max(np.abs(target_col))

    def build_benchmark(self,
                        n_cases: int = 100,
                        min_margin: float = 0.1) -> List[MultiRowCase]:
        """
        构建多行定理5.1 benchmark

        Args:
            n_cases: 目标案例数
            min_margin: 最小margin阈值

        Returns:
            List of MultiRowCase
        """
        cases = []
        attempts = 0
        max_attempts = n_cases * 10

        while len(cases) < n_cases and attempts < max_attempts:
            attempts += 1

            # 生成矩阵
            A, valid = self.generate_non_collinear_matrix()

            if not valid:
                continue

            # 随机选择target
            target_j = random.randint(0, A.shape[1] - 1)

            # 计算margin
            m_j = self.compute_margin(A, target_j)

            if m_j < min_margin:
                continue  # margin太小，跳过

            # 生成delta
            delta = random.uniform(1, 10)

            # 计算残差
            residual = delta * A[:, target_j]

            case_id = f"multi_{len(cases)}_m{m_j:.2f}"

            cases.append(MultiRowCase(
                case_id=case_id,
                A=A.copy(),
                target_j=target_j,
                delta=delta,
                m_j=m_j,
                residual=residual
            ))

        print(f"Generated {len(cases)} valid cases from {attempts} attempts")
        print(f"Mean margin: {np.mean([c.m_j for c in cases]):.4f}")

        return cases


class Theorem5MultiRowValidator:
    """定理5.1多行验证器"""

    def theorem_decoder(self, residual: np.ndarray, A: np.ndarray) -> int:
        """
        定理5.1 decoder

        j_hat = argmin_j inf_α ||r - α A_j||_∞
        """
        n_cols = A.shape[1]
        scores = []

        for j in range(n_cols):
            col = A[:, j]

            # inf_α ||r - α col||_∞
            # 用grid search
            alphas = np.linspace(-20, 20, 2001)
            distances = [np.max(np.abs(residual - a * col)) for a in alphas]
            min_dist = min(distances)

            scores.append(min_dist)

        return np.argmin(scores)

    def inject_noise(self, residual: np.ndarray, epsilon: float) -> np.ndarray:
        """注入bounded noise ||η||_∞ ≤ ε"""
        noise = np.random.uniform(-epsilon, epsilon, residual.shape)
        return residual + noise

    def run_validation(self,
                       cases: List[MultiRowCase],
                       tau_values: List[float] = None,
                       n_samples: int = 50) -> Dict:
        """
        运行验证实验

        Args:
            cases: 多行案例列表
            tau_values: τ网格
            n_samples: 每个τ的采样次数

        Returns:
            验证结果
        """
        if tau_values is None:
            tau_values = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        recovery_by_tau = {tau: [] for tau in tau_values}

        for case in cases:
            # 原始阈值
            threshold = abs(case.delta) * case.m_j

            for tau in tau_values:
                # ε = τ * threshold / 2
                epsilon = tau * threshold / 2

                correct = 0
                for _ in range(n_samples):
                    # 加噪声
                    r_noisy = self.inject_noise(case.residual, epsilon)

                    # Decoder
                    predicted = self.theorem_decoder(r_noisy, case.A)

                    if predicted == case.target_j:
                        correct += 1

                recovery_by_tau[tau].append(correct / n_samples)

        # 汇总
        results = {
            'tau_values': tau_values,
            'mean_recovery': {tau: np.mean(recovery_by_tau[tau]) for tau in tau_values},
            'std_recovery': {tau: np.std(recovery_by_tau[tau]) for tau in tau_values},
            'n_cases': len(cases),
            'n_samples': n_samples,
            'margin_stats': {
                'mean': float(np.mean([c.m_j for c in cases])),
                'min': float(np.min([c.m_j for c in cases])),
                'max': float(np.max([c.m_j for c in cases]))
            }
        }

        return results

    def print_summary(self, results: Dict):
        """打印结果"""
        print("\n" + "=" * 60)
        print("Theorem 5.1 Multi-Row Validation Results")
        print("=" * 60)

        print("\nRecovery Rate vs τ:")
        print("τ     | Recovery@1 | Std")
        print("-" * 40)
        for tau in results['tau_values']:
            mean = results['mean_recovery'][tau]
            std = results['std_recovery'][tau]
            print(f"{tau:.2f}  | {mean:.2%}     | {std:.2%}")

        # 验证定理
        print("\n" + "=" * 60)
        print("Theorem 5.1 Verification:")
        print("=" * 60)

        tau_below_1 = [t for t in results['tau_values'] if t < 1]
        tau_above_1 = [t for t in results['tau_values'] if t > 1]

        if tau_below_1:
            mean_below = np.mean([results['mean_recovery'][t] for t in tau_below_1])
            print(f"τ < 1: Recovery = {mean_below:.2%}")
            print(f"  Expected: ~100%")
            print(f"  Status: {'✓ PASS' if mean_below > 0.95 else '✗ FAIL'}")

        if tau_above_1:
            mean_above = np.mean([results['mean_recovery'][t] for t in tau_above_1])
            print(f"\nτ > 1: Recovery = {mean_above:.2%}")
            if tau_below_1:
                print(f"  Expected: < τ<1 region")
                print(f"  Status: {'✓ PASS' if mean_above < mean_below else '✗ FAIL'}")


def main():
    """运行多行定理5.1验证"""

    print("=" * 60)
    print("Multi-Row Theorem 5.1 Validation")
    print("=" * 60)

    set_seed(42)

    # 快速验证版本
    builder = MultiRowTheorem5Benchmark(n_rows=4, n_cols=6)
    cases = builder.build_benchmark(n_cases=30, min_margin=0.1)  # 减少案例数

    # 运行验证
    validator = Theorem5MultiRowValidator()
    results = validator.run_validation(
        cases,
        tau_values=[0, 0.5, 1.0, 1.5],  # 减少τ值
        n_samples=20  # 减少采样次数
    )

    # 打印结果
    validator.print_summary(results)

    # 保存
    with open('data/benchmark/theorem5_multirow_validation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Multi-row theorem 5.1 validation complete")

    return results


if __name__ == "__main__":
    main()
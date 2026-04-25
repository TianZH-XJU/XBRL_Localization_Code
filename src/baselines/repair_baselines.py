"""
XBRL Minimal Repair Baselines
PG-MR (Propagation-Graph-based Minimal Repair)
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from ortools.linear_solver import pywraplp
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2


class Filing:
    """XBRL Filing representation"""
    def __init__(self, facts: Dict[str, float], constraints: List[Dict]):
        self.facts = facts
        self.constraints = constraints
        self.n_facts = len(facts)

    def check_violations(self) -> List[Dict]:
        """Check constraint violations"""
        violations = []
        for c in self.constraints:
            if c["type"] == "balance":
                # Assets = Liabilities + Equity
                assets = self.facts.get("Assets", 0)
                liabilities = self.facts.get("Liabilities", 0)
                equity = self.facts.get("Equity", 0)
                residual = abs(assets - liabilities - equity)
                if residual > 0.01:
                    violations.append({
                        "constraint": c,
                        "residual": residual,
                        "involved_facts": ["Assets", "Liabilities", "Equity"]
                    })
        return violations


class RepairAction:
    """Repair action representation"""
    def __init__(self, fact_name: str, action_type: str, new_value: float, cost: float):
        self.fact_name = fact_name
        self.action_type = action_type  # sign_flip, scale_fix, value_edit
        self.new_value = new_value
        self.cost = cost


class GreedyBaseline:
    """Greedy heuristic for minimal repair"""

    COST_ORDER = {"sign_flip": 1, "scale_fix": 2, "value_edit": 3}

    def repair(self, filing: Filing) -> List[RepairAction]:
        """Find minimal repair using greedy heuristic"""
        repairs = []
        violations = filing.check_violations()

        while violations:
            # 找参与最多violation的fact
            fact_counts = {}
            for v in violations:
                for fact in v["involved_facts"]:
                    fact_counts[fact] = fact_counts.get(fact, 0) + 1

            target_fact = max(fact_counts, key=fact_counts.get)

            # 尝试最低成本动作
            for action_type in ["sign_flip", "scale_fix", "value_edit"]:
                repair = self._try_repair(filing, target_fact, action_type)
                if repair:
                    repairs.append(repair)
                    filing.facts[target_fact] = repair.new_value
                    violations = filing.check_violations()
                    break

            if not violations:
                break

        return repairs

    def _try_repair(self, filing: Filing, fact: str, action_type: str) -> Optional[RepairAction]:
        """尝试特定修复动作"""
        current_value = filing.facts[fact]

        if action_type == "sign_flip":
            new_value = -current_value
            return RepairAction(fact, "sign_flip", new_value, 1)

        elif action_type == "scale_fix":
            # 尝试×10或÷10
            for scale in [0.1, 10]:
                new_value = current_value * scale
                temp_facts = filing.facts.copy()
                temp_facts[fact] = new_value
                temp_filing = Filing(temp_facts, filing.constraints)
                if not temp_filing.check_violations():
                    return RepairAction(fact, "scale_fix", new_value, 2)

        elif action_type == "value_edit":
            # 计算正确值
            assets = filing.facts.get("Assets", 0)
            liabilities = filing.facts.get("Liabilities", 0)
            equity = filing.facts.get("Equity", 0)

            if fact == "Assets":
                new_value = liabilities + equity
            elif fact == "Liabilities":
                new_value = assets - equity
            elif fact == "Equity":
                new_value = assets - liabilities
            else:
                return None

            return RepairAction(fact, "value_edit", new_value, 3)

        return None


class ILPBaseline:
    """ILP solver for minimal repair"""

    def repair(self, filing: Filing) -> List[RepairAction]:
        """Find minimal repair using ILP"""
        solver = pywraplp.Solver.CreateSolver('CBC')

        facts = list(filing.facts.keys())
        n = len(facts)

        # 变量：z_i = fact i是否被修改
        z = [solver.IntVar(0, 1, f'z_{i}') for i in range(n)]

        # 变量：v'_i = 修复后值
        v_new = [solver.NumVar(-1e10, 1e10, f'v_{i}') for i in range(n)]

        # 约束：如果未修改则v'_i = v_i
        for i in range(n):
            solver.Add(v_new[i] >= filing.facts[facts[i]] - 1e-6 * z[i])
            solver.Add(v_new[i] <= filing.facts[facts[i]] + 1e-6 * z[i])

        # 约束：balance equation (Assets = Liabilities + Equity)
        try:
            assets_idx = facts.index("Assets")
            liabilities_idx = facts.index("Liabilities")
            equity_idx = facts.index("Equity")
            solver.Add(v_new[assets_idx] == v_new[liabilities_idx] + v_new[equity_idx])
        except ValueError:
            pass

        # 目标：minimize sum(z_i) + cost
        solver.Minimize(sum(z))

        status = solver.Solve()

        repairs = []
        if status == pywraplp.Solver.OPTIMAL:
            for i in range(n):
                if z[i].solution_value() > 0.5:
                    repairs.append(RepairAction(
                        facts[i],
                        "value_edit",
                        v_new[i].solution_value(),
                        3
                    ))

        return repairs


class MaxSATBaseline:
    """MaxSAT solver for minimal repair"""

    def repair(self, filing: Filing) -> List[RepairAction]:
        """Find minimal repair using MaxSAT"""
        # 这里简化为符号约束求解
        # 实际实现需要将数值问题离散化

        # 对于balance问题，直接计算最小修复
        assets = filing.facts.get("Assets", 0)
        liabilities = filing.facts.get("Liabilities", 0)
        equity = filing.facts.get("Equity", 0)

        residual = assets - liabilities - equity

        # 选择修改最小的fact
        repairs = []
        if abs(residual) > 0.01:
            # 计算修改每个fact的成本
            costs = {
                "Assets": abs(residual),
                "Liabilities": abs(residual),
                "Equity": abs(residual)
            }

            min_fact = min(costs, key=costs.get)

            if min_fact == "Assets":
                new_value = liabilities + equity
            elif min_fact == "Liabilities":
                new_value = assets - equity
            else:
                new_value = assets - liabilities

            repairs.append(RepairAction(min_fact, "value_edit", new_value, 3))

        return repairs


def evaluate_repair(filing: Filing, repairs: List[RepairAction]) -> Dict:
    """评估修复结果"""
    # 应用修复
    repaired_facts = filing.facts.copy()
    for r in repairs:
        repaired_facts[r.fact_name] = r.new_value

    repaired_filing = Filing(repaired_facts, filing.constraints)

    # 计算指标
    violations = repaired_filing.check_violations()

    return {
        "n_repairs": len(repairs),
        "constraint_satisfaction": len(violations) == 0,
        "repair_types": [r.action_type for r in repairs],
        "total_cost": sum(r.cost for r in repairs),
        "remaining_violations": len(violations)
    }


def run_benchmark(benchmark_path: str):
    """运行benchmark评估"""
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    results = {
        "greedy": [],
        "ilp": [],
        "maxsat": []
    }

    baselines = {
        "greedy": GreedyBaseline(),
        "ilp": ILPBaseline(),
        "maxsat": MaxSATBaseline()
    }

    for instance in benchmark["instances"]:
        filing = Filing(instance["original_values"], instance["constraints"])

        for name, baseline in baselines.items():
            repairs = baseline.repair(filing)
            eval_result = evaluate_repair(filing, repairs)
            results[name].append({
                "instance_id": instance["id"],
                "error_type": instance["error_type"],
                "difficulty": instance["difficulty"],
                **eval_result
            })

    return results


if __name__ == "__main__":
    print("Running baseline experiments...")
    results = run_benchmark("data/benchmark/synthetic_benchmark.json")

    # 统计结果
    for name, res in results.items():
        success_rate = sum(1 for r in res if r["constraint_satisfaction"]) / len(res)
        avg_repairs = sum(r["n_repairs"] for r in res) / len(res)
        avg_cost = sum(r["total_cost"] for r in res) / len(res)

        print(f"\n{name.upper()} Results:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg repairs: {avg_repairs:.2f}")
        print(f"  Avg cost: {avg_cost:.2f}")
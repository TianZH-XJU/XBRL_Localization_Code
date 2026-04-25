"""
Theorem-Driven Validation Experiments
Connecting theoretical results to empirical observations
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class FinancialConstraintExtractor:
    def extract_calc_constraints(self, table_data: List[List]) -> List[Dict]:
        constraints = []
        if not table_data or len(table_data) < 3:
            return constraints

        ncols = len(table_data[0]) if table_data else 0

        for i, row in enumerate(table_data):
            values = []
            for j in range(ncols):
                num = self._parse_number(row[j])
                if num is not None:
                    values.append((j, num))

            if len(values) >= 3:
                for split_idx in range(1, len(values) - 1):
                    component_sum = sum(v for _, v in values[:split_idx])
                    total = values[-1][1]
                    residual = abs(component_sum - total)

                    if residual < max(abs(total) * 0.02, 10):
                        constraints.append({
                            "type": "calc_sum",
                            "row": i,
                            "components": values[:split_idx],
                            "total": values[-1],
                            "equation": f"col_{values[-1][0]} = sum(cols_0-{split_idx-1})"
                        })
                        break

        return constraints

    def _parse_number(self, cell) -> Optional[float]:
        if cell is None or cell == '':
            return None
        cell = str(cell)
        match = re.search(r'[\(]?([\d,]+(?:\.\d+)?)[\)]?', cell)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                num = float(num_str)
                if '(' in cell:
                    num = -num
                return num
            except:
                return None
        return None


class ViolationTask:
    def __init__(self, original_table, modified_table, error_info, constraints):
        self.original_table = original_table
        self.modified_table = modified_table
        self.error_info = error_info
        self.constraints = constraints
        self.facts = self._extract_facts(modified_table)
        self.violations = self._check_violations()
        self.target_fact_id = f"fact_{error_info['target_row']}_{error_info['target_col']}"
        self.target_is_total = self._is_target_total()
        self.deg_j = self._compute_degree()  # Node degree for Theorem 4.1
        self.candidate_size = self._compute_candidate_size()  # |C(j)| for Theorem 4.1
        self.equivalence_class_size = self._compute_equiv_class_size()  # For Theorem 3.1

    def _extract_facts(self, table_data):
        facts = []
        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                num = self._parse_number(cell)
                if num is not None:
                    facts.append({"id": f"fact_{i}_{j}", "row": i, "col": j, "value": num})
        return facts

    def _parse_number(self, cell):
        if cell is None or cell == '':
            return None
        cell = str(cell)
        match = re.search(r'[\(]?([\d,]+(?:\.\d+)?)[\)]?', cell)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                num = float(num_str)
                if '(' in cell:
                    num = -num
                return num
            except:
                return None
        return None

    def _check_violations(self):
        violations = []
        for c in self.constraints:
            if c["type"] == "calc_sum":
                row = c["row"]
                component_sum = sum(self._get_cell_value(row, j) for j, _ in c["components"]
                                    if self._get_cell_value(row, j) is not None)
                total_j = c["total"][0]
                current_total = self._get_cell_value(row, total_j)

                if current_total is not None:
                    residual = abs(component_sum - current_total)
                    if residual > max(abs(current_total) * 0.01, 5):
                        involved_ids = [f"fact_{row}_{j}" for j, _ in c["components"]]
                        involved_ids.append(f"fact_{row}_{total_j}")
                        violations.append({
                            "constraint_id": c["equation"],
                            "row": row,
                            "residual": residual,
                            "involved_facts": involved_ids,
                            "reported_total": current_total,
                            "expected_sum": component_sum,
                            "components": c["components"],
                            "total_col": total_j,
                            "arity": len(involved_ids)
                        })
        return violations

    def _is_target_total(self):
        for v in self.violations:
            if f"fact_{v['row']}_{v['total_col']}" == self.target_fact_id:
                return True
        return False

    def _get_cell_value(self, row, col):
        if row < len(self.modified_table) and col < len(self.modified_table[row]):
            return self._parse_number(self.modified_table[row][col])
        return None

    def _compute_degree(self):
        """Compute deg(j) for target cell (Theorem 4.1)"""
        return len(self.violations)  # Number of constraints involving target

    def _compute_candidate_size(self):
        """Compute |C(j)| for target cell (Theorem 4.1)"""
        if not self.violations:
            return len(self.facts)

        # Union of all involved cells across violated constraints
        all_involved = set()
        for v in self.violations:
            all_involved.update(v["involved_facts"])

        return len(all_involved)

    def _compute_equiv_class_size(self):
        """Compute equivalence class size based on residual magnitude similarity"""
        if not self.violations:
            return 1

        # Cells with similar residual contribution (rough equivalence)
        residual_dict = defaultdict(float)
        for v in self.violations:
            for fid in v["involved_facts"]:
                residual_dict[fid] += v["residual"]

        # Group by similar residual (within 10%)
        target_residual = residual_dict.get(self.target_fact_id, 0)
        equiv_size = 0
        for fid, res in residual_dict.items():
            if abs(res - target_residual) < 0.1 * max(abs(target_residual), 1):
                equiv_size += 1

        return max(equiv_size, 1)

    def has_violation(self):
        return len(self.violations) > 0

    def evaluate(self, predicted_ids):
        hit_1 = predicted_ids[0] == self.target_fact_id if predicted_ids else False
        hit_3 = self.target_fact_id in predicted_ids[:3] if len(predicted_ids) >= 3 else False
        mrr = 0
        for i, id in enumerate(predicted_ids):
            if id == self.target_fact_id:
                mrr = 1.0 / (i + 1)
                break
        return {"hit_1": hit_1, "hit_3": hit_3, "mrr": mrr}


def inject_violation_error(original_table, constraints):
    if not constraints:
        return None, None

    modified = [row[:] for row in original_table]
    constraint = random.choice(constraints)

    row = constraint["row"]
    error_type = random.choice(["component", "total"])
    sub_type = random.choice(["sign_flip", "scale_10", "value_shift"])

    if error_type == "component":
        target_j, orig_val = constraint["components"][0]
        target_col = target_j
        original_value = orig_val
    else:
        target_j, orig_val = constraint["total"]
        target_col = target_j
        original_value = orig_val

    if sub_type == "sign_flip":
        new_val = -original_value
    elif sub_type == "scale_10":
        new_val = original_value * 10
    else:
        new_val = original_value + random.uniform(-0.3, 0.3) * abs(original_value)

    if new_val < 0:
        modified[row][target_col] = f"({abs(int(new_val))})"
    else:
        modified[row][target_col] = str(int(new_val))

    error_info = {
        "target_row": row,
        "target_col": target_col,
        "original_value": original_value,
        "error_value": new_val,
        "error_type": sub_type,
        "is_total_error": error_type == "total"
    }

    return modified, error_info


def build_benchmark():
    with open("data/benchmark/inconsistency_repair_benchmark.json") as f:
        base = json.load(f)

    extractor = FinancialConstraintExtractor()
    tasks = []

    for inst in base['instances']:
        original = inst['modified_table']
        constraints = extractor.extract_calc_constraints(original)

        if not constraints:
            continue

        modified, error_info = inject_violation_error(original, constraints)
        if modified is None:
            continue

        task = ViolationTask(original, modified, error_info, constraints)

        if task.has_violation():
            tasks.append(task)

    return tasks


class GreedyBaseline:
    def localize(self, task):
        if not task.violations:
            return [f["id"] for f in task.facts]

        fact_scores = defaultdict(float)
        for v in task.violations:
            for fid in v["involved_facts"]:
                fact_scores[fid] += v["residual"]

        ranked = sorted(fact_scores.keys(), key=lambda x: fact_scores[x], reverse=True)

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        return ranked


def run_theorem_experiments():
    """Run theorem-driven validation experiments"""
    print("=" * 60)
    print("Theorem-Driven Validation Experiments")
    print("Connecting theoretical results to empirical observations")
    print("=" * 60)

    set_seed(42)
    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    # ============================================================
    # Experiment 1: Accuracy vs deg(j) (Theorem 4.1)
    # ============================================================
    print("\n[Experiment 1] Accuracy vs Node Degree deg(j)...")

    greedy = GreedyBaseline()

    deg_groups = defaultdict(list)
    for task in tasks:
        pred = greedy.localize(task)
        result = task.evaluate(pred)
        deg_groups[task.deg_j].append(result["hit_1"])

    print("\nGreedy accuracy by degree:")
    for deg in sorted(deg_groups.keys()):
        acc = np.mean(deg_groups[deg])
        n = len(deg_groups[deg])
        print(f"  deg={deg} (n={n}): {acc:.2%}")

    # ============================================================
    # Experiment 2: Accuracy vs Candidate Size |C(j)|
    # ============================================================
    print("\n[Experiment 2] Accuracy vs Candidate Size |C(j)|...")

    size_groups = defaultdict(list)
    for task in tasks:
        pred = greedy.localize(task)
        result = task.evaluate(pred)
        size_groups[task.candidate_size].append(result["hit_1"])

    print("\nGreedy accuracy by candidate size:")
    for size in sorted(size_groups.keys())[:10]:  # Top 10 sizes
        acc = np.mean(size_groups[size])
        n = len(size_groups[size])
        print(f"  |C|={size} (n={n}): {acc:.2%}")

    # ============================================================
    # Experiment 3: Degree/Candidate Distribution by Role
    # ============================================================
    print("\n[Experiment 3] Degree/Candidate Distribution by Role...")

    total_tasks = [t for t in tasks if t.target_is_total]
    comp_tasks = [t for t in tasks if not t.target_is_total]

    total_deg = [t.deg_j for t in total_tasks]
    comp_deg = [t.deg_j for t in comp_tasks]

    total_cand = [t.candidate_size for t in total_tasks]
    comp_cand = [t.candidate_size for t in comp_tasks]

    print(f"\nTotal-target (n={len(total_tasks)}):")
    print(f"  Mean deg: {np.mean(total_deg):.2f}, Std: {np.std(total_deg):.2f}")
    print(f"  Mean |C|: {np.mean(total_cand):.2f}, Std: {np.std(total_cand):.2f}")

    print(f"\nComponent-target (n={len(comp_tasks)}):")
    print(f"  Mean deg: {np.mean(comp_deg):.2f}, Std: {np.std(comp_deg):.2f}")
    print(f"  Mean |C|: {np.mean(comp_cand):.2f}, Std: {np.std(comp_cand):.2f}")

    # Statistical test
    if len(total_deg) > 1 and len(comp_deg) > 1:
        from scipy import stats
        t_deg, p_deg = stats.ttest_ind(total_deg, comp_deg)
        t_cand, p_cand = stats.ttest_ind(total_cand, comp_cand)
        print(f"\nStatistical tests:")
        print(f"  Degree difference: t={t_deg:.2f}, p={p_deg:.4f}")
        print(f"  Candidate size difference: t={t_cand:.2f}, p={p_cand:.4f}")

    # ============================================================
    # Experiment 4: Baseline vs Impossibility Bound
    # ============================================================
    print("\n[Experiment 4] Baseline vs Impossibility Bound (Theorem 3.1)...")

    # Compute theoretical impossibility bound
    equiv_sizes = [t.equivalence_class_size for t in tasks]
    theoretical_bound = np.mean([1/s for s in equiv_sizes])

    greedy_results = []
    for task in tasks:
        pred = greedy.localize(task)
        greedy_results.append(task.evaluate(pred)["hit_1"])

    greedy_acc = np.mean(greedy_results)

    print(f"\nTheoretical impossibility bound: {theoretical_bound:.2%}")
    print(f"Greedy baseline accuracy: {greedy_acc:.2%}")
    print(f"Gap (greedy - bound): {greedy_acc - theoretical_bound:.2%}")

    if greedy_acc <= theoretical_bound + 0.05:
        print("  ✓ Greedy approaches theoretical ceiling")
    else:
        print("  ⚠ Greedy exceeds bound - may use features beyond residual")

    # ============================================================
    # Experiment 5: Accuracy by Equivalence Class Size
    # ============================================================
    print("\n[Experiment 5] Accuracy by Equivalence Class Size...")

    equiv_groups = defaultdict(list)
    for task in tasks:
        pred = greedy.localize(task)
        result = task.evaluate(pred)
        equiv_groups[task.equivalence_class_size].append(result["hit_1"])

    print("\nGreedy accuracy by equivalence class size:")
    for equiv in sorted(equiv_groups.keys())[:8]:
        acc = np.mean(equiv_groups[equiv])
        n = len(equiv_groups[equiv])
        theoretical = 1.0 / equiv
        print(f"  equiv={equiv} (n={n}): acc={acc:.2%}, theoretical≤{theoretical:.2%}")

    # ============================================================
    # Experiment 6: Correlation Analysis
    # ============================================================
    print("\n[Experiment 6] Correlation Analysis...")

    # Collect all metrics
    degs = [t.deg_j for t in tasks]
    cands = [t.candidate_size for t in tasks]
    equivs = [t.equivalence_class_size for t in tasks]
    greedy_hits = []

    for task in tasks:
        pred = greedy.localize(task)
        greedy_hits.append(1 if task.evaluate(pred)["hit_1"] else 0)

    # Correlations
    corr_deg = np.corrcoef(degs, greedy_hits)[0, 1]
    corr_cand = np.corrcoef(cands, greedy_hits)[0, 1]
    corr_equiv = np.corrcoef(equivs, greedy_hits)[0, 1]

    print(f"\nCorrelation with accuracy:")
    print(f"  deg(j): r = {corr_deg:.3f}")
    print(f"  |C(j)|: r = {corr_cand:.3f}")
    print(f"  equiv_class: r = {corr_equiv:.3f}")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "experiment": "theorem_validation",
        "degree_analysis": {
            "total_mean": float(np.mean(total_deg)),
            "total_std": float(np.std(total_deg)),
            "component_mean": float(np.mean(comp_deg)),
            "component_std": float(np.std(comp_deg)),
            "p_value": float(p_deg) if 'p_deg' in dir() else 1.0
        },
        "candidate_analysis": {
            "total_mean": float(np.mean(total_cand)),
            "component_mean": float(np.mean(comp_cand)),
            "p_value": float(p_cand) if 'p_cand' in dir() else 1.0
        },
        "impossibility_bound": {
            "theoretical": float(theoretical_bound),
            "greedy_actual": float(greedy_acc)
        },
        "correlations": {
            "deg_accuracy": float(corr_deg),
            "cand_accuracy": float(corr_cand),
            "equiv_accuracy": float(corr_equiv)
        }
    }

    with open("data/benchmark/theorem_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/theorem_validation_results.json")

    return results


if __name__ == "__main__":
    run_theorem_experiments()
    print("\n" + "=" * 60)
    print("Theorem Validation Complete")
    print("=" * 60)
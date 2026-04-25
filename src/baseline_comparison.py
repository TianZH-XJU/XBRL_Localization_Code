"""
XBRL Minimal Repair - Baseline Comparison + Failure Analysis
Iteration 3: Address Codex Round 5 feedback
1. Compare against greedy, leave-one-out, minimal residual
2. Report candidate recall by role × error type
3. Failure case studies for value_shift-total
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from scipy import stats


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
                            "n_involved": len(involved_ids)
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

    def has_violation(self):
        return len(self.violations) > 0

    def evaluate(self, predicted_ids):
        hit_1 = predicted_ids[0] == self.target_fact_id if predicted_ids else False
        hit_3 = self.target_fact_id in predicted_ids[:3] if len(predicted_ids) >= 3 else False
        hit_5 = self.target_fact_id in predicted_ids[:5] if len(predicted_ids) >= 5 else False

        mrr = 0
        for i, id in enumerate(predicted_ids):
            if id == self.target_fact_id:
                mrr = 1.0 / (i + 1)
                break

        return {"hit_1": hit_1, "hit_3": hit_3, "hit_5": hit_5, "mrr": mrr}


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


# === Baseline Methods ===

class GreedyViolationBaseline:
    """Rank by violation involvement score (sum of residuals)"""

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


class LeaveOneOutBaseline:
    """Rank by how much fixing each cell would reduce residual"""

    def localize(self, task):
        if not task.violations:
            return [f["id"] for f in task.facts]

        fact_scores = {}

        for fact in task.facts:
            repair_impact = 0.0

            for v in task.violations:
                if fact["id"] in v["involved_facts"]:
                    repair_impact += v["residual"]

                # Check if this is a component
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fact["id"]:
                        other_sum = sum(task._get_cell_value(v["row"], j2) or 0
                                       for j2, _ in v.get("components", [])
                                       if j2 != j)
                        expected_val = v["reported_total"] - other_sum
                        current_val = fact["value"]
                        deviation = abs(current_val - expected_val)
                        repair_impact = max(repair_impact, deviation)

            fact_scores[fact["id"]] = repair_impact

        ranked = sorted(fact_scores.keys(), key=lambda x: fact_scores[x], reverse=True)

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        return ranked


class MinimalResidualBaseline:
    """Rank by residual reduction if cell is corrected"""

    def localize(self, task):
        if not task.violations:
            return [f["id"] for f in task.facts]

        fact_scores = {}
        total_residual = sum(v["residual"] for v in task.violations)

        for fact in task.facts:
            residual_after_fix = total_residual

            for v in task.violations:
                if fact["id"] in v["involved_facts"]:
                    residual_after_fix -= v["residual"]

            fact_scores[fact["id"]] = total_residual - residual_after_fix

        ranked = sorted(fact_scores.keys(), key=lambda x: fact_scores[x], reverse=True)

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        return ranked


def get_candidates(task, top_k=10):
    """Get candidates from Stage 1"""
    if not task.violations:
        return [f["id"] for f in task.facts[:top_k]], {}

    fact_scores = defaultdict(float)
    for v in task.violations:
        for fid in v["involved_facts"]:
            fact_scores[fid] += v["residual"]

    ranked = sorted(fact_scores.keys(), key=lambda x: fact_scores[x], reverse=True)

    for f in task.facts:
        if f["id"] not in ranked:
            ranked.append(f["id"])

    return ranked[:top_k], fact_scores


def run_baseline_comparison():
    """Compare unified ranker against baselines"""
    print("=" * 60)
    print("XBRL Minimal Repair - Baseline Comparison + Failure Analysis")
    print("Iteration 3: Address Codex Round 5 feedback")
    print("=" * 60)

    set_seed(42)
    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    # Split
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    print(f"Train: {len(train)}, Test: {len(test)}")

    # ============================================================
    # Candidate Recall Analysis (Stage 1)
    # ============================================================
    print("\n[Step 1] Candidate Recall by Role × Error Type...")

    candidates_in_top5 = 0
    candidates_in_top10 = 0

    # By role
    total_test = [t for t in test if t.target_is_total]
    comp_test = [t for t in test if not t.target_is_total]

    total_recall_5 = sum(1 for t in total_test if t.target_fact_id in get_candidates(t, 5)[0])
    comp_recall_5 = sum(1 for t in comp_test if t.target_fact_id in get_candidates(t, 5)[0])

    print(f"\nCandidate Recall@5:")
    print(f"  Total-target (n={len(total_test)}): {total_recall_5/len(total_test):.2%} ({total_recall_5}/{len(total_test)})")
    print(f"  Component-target (n={len(comp_test)}): {comp_recall_5/len(comp_test):.2%} ({comp_recall_5}/{len(comp_test)})")

    # By role × error type
    print(f"\nCandidate Recall@5 by Role × Error Type:")
    for role in ["total", "component"]:
        for error_type in ["sign_flip", "scale_10", "value_shift"]:
            subset = [t for t in test
                     if (t.target_is_total if role == "total" else not t.target_is_total)
                     and t.error_info["error_type"] == error_type]

            if len(subset) == 0:
                continue

            recall_5 = sum(1 for t in subset if t.target_fact_id in get_candidates(t, 5)[0])
            print(f"  {role}-{error_type} (n={len(subset)}): {recall_5/len(subset):.2%} ({recall_5}/{len(subset)})")

    # ============================================================
    # Baseline Comparison
    # ============================================================
    print("\n[Step 2] Baseline Comparison...")

    baselines = {
        "greedy_violation": GreedyViolationBaseline(),
        "leave_one_out": LeaveOneOutBaseline(),
        "minimal_residual": MinimalResidualBaseline()
    }

    baseline_results = {}

    for name, baseline in baselines.items():
        results = []
        for task in test:
            pred = baseline.localize(task)
            results.append(task.evaluate(pred))

        top1 = sum(r["hit_1"] for r in results) / len(results)
        top3 = sum(r["hit_3"] for r in results) / len(results)
        top5 = sum(r["hit_5"] for r in results) / len(results)
        mrr = sum(r["mrr"] for r in results) / len(results)

        baseline_results[name] = {"top1": top1, "top3": top3, "top5": top5, "mrr": mrr}
        print(f"  {name}: Top-1={top1:.2%}, Top-3={top3:.2%}, MRR={mrr:.4f}")

    # ============================================================
    # Oracle Upper Bound
    # ============================================================
    print("\n[Step 3] Oracle Upper Bound (reranking top-5 perfectly)...")

    oracle_results = []
    for task in test:
        candidates, _ = get_candidates(task, 5)
        # Oracle: if target in top-5, rank it first; else fail
        if task.target_fact_id in candidates:
            pred = [task.target_fact_id] + [c for c in candidates if c != task.target_fact_id]
        else:
            pred = candidates
        oracle_results.append(task.evaluate(pred))

    oracle_top1 = sum(r["hit_1"] for r in oracle_results) / len(oracle_results)
    oracle_top3 = sum(r["hit_3"] for r in oracle_results) / len(oracle_results)
    print(f"  Oracle Top-1: {oracle_top1:.2%}")
    print(f"  Oracle Top-3: {oracle_top3:.2%}")

    # ============================================================
    # Failure Case Studies
    # ============================================================
    print("\n[Step 4] Failure Case Studies (value_shift-total)...")

    value_shift_total = [t for t in test
                        if t.target_is_total and t.error_info["error_type"] == "value_shift"]

    print(f"  value_shift-total tasks: {len(value_shift_total)}")

    for i, task in enumerate(value_shift_total[:3]):  # Analyze first 3
        candidates, scores = get_candidates(task, 10)
        greedy_pred = baselines["greedy_violation"].localize(task)

        print(f"\n  Case {i+1}:")
        print(f"    Target: {task.target_fact_id} (value={task.error_info['error_value']:.2f})")
        print(f"    Original: {task.error_info['original_value']:.2f}")
        print(f"    Target in candidates: {task.target_fact_id in candidates}")
        print(f"    Greedy Top-1: {greedy_pred[0]}")
        print(f"    Greedy correct: {greedy_pred[0] == task.target_fact_id}")

        # Show candidate scores
        if task.target_fact_id in candidates:
            target_idx = candidates.index(task.target_fact_id)
            target_score = scores.get(task.target_fact_id, 0)
            print(f"    Target rank in candidates: #{target_idx+1}, score={target_score:.2f}")
            print(f"    Top candidate: {candidates[0]}, score={scores.get(candidates[0], 0):.2f}")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "iteration": 3,
        "candidate_recall": {
            "total_target": {"recall_5": total_recall_5/len(total_test) if total_test else 0, "n": len(total_test)},
            "component_target": {"recall_5": comp_recall_5/len(comp_test) if comp_test else 0, "n": len(comp_test)}
        },
        "baselines": baseline_results,
        "oracle": {"top1": oracle_top1, "top3": oracle_top3},
        "value_shift_total_count": len(value_shift_total)
    }

    with open("data/benchmark/baseline_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/baseline_comparison_results.json")

    return results


if __name__ == "__main__":
    run_baseline_comparison()
    print("\n" + "=" * 60)
    print("Baseline Comparison + Failure Analysis Complete")
    print("=" * 60)
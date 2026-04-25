"""
XBRL Minimal Repair - Round 1 Fix
Address Codex Review: Candidate Recall + Statistical Rigor + Extended Benchmark
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
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
                            "components": c["components"]
                        })
        return violations

    def _get_cell_value(self, row, col):
        if row < len(self.modified_table) and col < len(self.modified_table[row]):
            return self._parse_number(self.modified_table[row][col])
        return None

    def has_violation(self):
        return len(self.violations) > 0

    def evaluate(self, predicted_ids, k_values=[1, 3, 5, 10]):
        """Extended evaluation with multiple k values"""
        results = {}
        for k in k_values:
            if len(predicted_ids) >= k:
                results[f"hit_{k}"] = self.target_fact_id in predicted_ids[:k]
            else:
                results[f"hit_{k}"] = False

        mrr = 0
        for i, id in enumerate(predicted_ids):
            if id == self.target_fact_id:
                mrr = 1.0 / (i + 1)
                break

        results["mrr"] = mrr
        return results


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
        "error_type": sub_type
    }

    return modified, error_info


class CandidateGeneratorWithRecall:
    """Candidate generator with recall tracking"""

    def generate_candidates(self, task, top_k=10):
        if not task.violations:
            all_facts = [f["id"] for f in task.facts]
            return all_facts[:top_k], {}

        fact_scores = defaultdict(float)
        for v in task.violations:
            for fid in v["involved_facts"]:
                fact_scores[fid] += v["residual"]

        ranked = sorted(fact_scores.keys(), key=lambda x: fact_scores[x], reverse=True)

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        return ranked[:top_k], fact_scores

    def compute_recall_at_k(self, task, k_values=[1, 3, 5, 10]):
        """Compute recall@k: is target in top-k candidates"""
        candidates, _ = self.generate_candidates(task, top_k=max(k_values))
        results = {}
        for k in k_values:
            results[f"recall@{k}"] = task.target_fact_id in candidates[:k] if len(candidates) >= k else False
        return results


def build_features(task, candidate_ids, violation_scores):
    """Build features (same as Iteration 10)"""
    features = []
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    all_values = [abs(f["value"]) for f in task.facts]
    mean_val = np.mean(all_values) if all_values else 0
    std_val = np.std(all_values) if len(all_values) > 1 else 0

    for fid in candidate_ids:
        fact = None
        for f in task.facts:
            if f["id"] == fid:
                fact = f
                break

        if fact is None:
            features.append([0] * 24)
            continue

        feat = []

        # Violation features
        feat.append(violation_scores.get(fid, 0) / max_residual)
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, len(task.violations)))
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / max_residual)
        if task.violations:
            feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / sum(v["residual"] for v in task.violations))
        else:
            feat.append(0.0)

        # Role features
        feat.append(1.0 if any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations) else 0.0)
        feat.append(1.0 if any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])] for v in task.violations) else 0.0)

        # Numeric features
        feat.append(fact["value"] / max_value)
        feat.append(abs(fact["value"]) / max_value)
        feat.append(1.0 if fact["value"] < 0 else 0.0)

        # Scale features
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)
        is_total = any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations)
        if is_total:
            diffs = [abs(fact["value"] - v["expected_sum"]) / max_value for v in task.violations
                     if abs(fact["value"] - v["reported_total"]) < 0.1]
            feat.append(min(diffs) if diffs else 0.0)
        else:
            feat.append(0.0)

        # Structural features
        feat.append(fact["row"] / 10)
        feat.append(fact["col"] / 5)

        # Discrepancy features
        discrepancy_score = 0.0
        for v in task.violations:
            if fid in v["involved_facts"]:
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        current_contrib = fact["value"]
                        other_sum = sum(task._get_cell_value(v["row"], j2) or 0
                                       for j2, _ in v.get("components", [])
                                       if j2 != j)
                        ideal_contrib = v["reported_total"] - other_sum
                        discrepancy_score = abs(current_contrib - ideal_contrib) / max_value
                        break

                if abs(fact["value"] - v["reported_total"]) < 0.1:
                    discrepancy_score = abs(v["reported_total"] - v["expected_sum"]) / max_value

        feat.append(discrepancy_score)

        if fact["value"] != 0:
            feat.append(max_residual / abs(fact["value"]))
        else:
            feat.append(0.0)

        row_facts = [f for f in task.facts if f["row"] == fact["row"] and f["id"] != fid]
        if row_facts:
            row_mean = np.mean([abs(f["value"]) for f in row_facts])
            feat.append(abs(abs(fact["value"]) - row_mean) / max_value)
        else:
            feat.append(0.0)

        if std_val > 0:
            feat.append(abs(abs(fact["value"]) - mean_val) / std_val)
        else:
            feat.append(0.0)

        feat.append(1.0)

        # Value_shift specific features
        absolute_shift = 0.0
        for v in task.violations:
            if fid in v["involved_facts"]:
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        ideal = v["expected_sum"] - sum(task._get_cell_value(v["row"], j2) or 0
                                                        for j2, _ in v.get("components", [])
                                                        if j2 != j)
                        absolute_shift = abs(fact["value"] - ideal)
                        break

                if abs(fact["value"] - v["reported_total"]) < 0.1:
                    absolute_shift = abs(v["reported_total"] - v["expected_sum"])

        feat.append(absolute_shift / max_value)

        row_values = [abs(f["value"]) for f in task.facts if f["row"] == fact["row"]]
        if row_values:
            row_median = np.median(row_values)
            feat.append(abs(abs(fact["value"]) - row_median) / max_value)
        else:
            feat.append(0.0)

        col_values = [abs(f["value"]) for f in task.facts if f["col"] == fact["col"]]
        if col_values:
            col_median = np.median(col_values)
            feat.append(abs(abs(fact["value"]) - col_median) / max_value)
        else:
            feat.append(0.0)

        neighbors = []
        for f in task.facts:
            if abs(f["row"] - fact["row"]) <= 1 and abs(f["col"] - fact["col"]) <= 1 and f["id"] != fid:
                neighbors.append(f["value"])

        if neighbors:
            neighbor_mean = np.mean(neighbors)
            feat.append(abs(fact["value"] - neighbor_mean) / max_value)
        else:
            feat.append(0.0)

        same_row_facts = [f for f in task.facts if f["row"] == fact["row"] and f["id"] != fid]
        if same_row_facts:
            sign_match = sum(1 for f in same_row_facts if (f["value"] < 0) == (fact["value"] < 0))
            feat.append(sign_match / len(same_row_facts))
        else:
            feat.append(0.0)

        repair_potential = 0.0
        for v in task.violations:
            if fid in v["involved_facts"]:
                repair_potential += v["residual"]

        feat.append(repair_potential / (max_residual * len(task.violations)) if task.violations else 0.0)

        features.append(feat)

    return np.array(features)


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


def paired_significance_test(greedy_results, xgb_results):
    """McNemar-like paired significance test"""
    # Count agreement/disagreement
    both_correct = sum(1 for g, x in zip(greedy_results, xgb_results) if g["hit_1"] and x["hit_1"])
    both_wrong = sum(1 for g, x in zip(greedy_results, xgb_results) if not g["hit_1"] and not x["hit_1"])
    greedy_correct_xgb_wrong = sum(1 for g, x in zip(greedy_results, xgb_results) if g["hit_1"] and not x["hit_1"])
    xgb_correct_greedy_wrong = sum(1 for g, x in zip(greedy_results, xgb_results) if not g["hit_1"] and x["hit_1"])

    # McNemar test
    # H0: no difference in accuracy
    # Test statistic: (|b-c|-1)^2 / (b+c) where b=xgb_correct_greedy_wrong, c=greedy_correct_xgb_wrong
    b = xgb_correct_greedy_wrong
    c = greedy_correct_xgb_wrong

    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
    else:
        p_value = 1.0  # No disagreement

    return {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "greedy_only": greedy_correct_xgb_wrong,
        "xgb_only": xgb_correct_greedy_wrong,
        "mcnemar_stat": mcnemar_stat if b + c > 0 else 0,
        "p_value": p_value
    }


def bootstrap_ci_for_improvement(greedy_hits, xgb_hits, n_bootstrap=1000):
    """Bootstrap CI for the improvement"""
    improvements = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(greedy_hits), size=len(greedy_hits), replace=True)
        greedy_rate = np.mean([greedy_hits[i] for i in indices])
        xgb_rate = np.mean([xgb_hits[i] for i in indices])
        improvements.append(xgb_rate - greedy_rate)

    return {
        "mean_improvement": np.mean(improvements),
        "ci_lower": np.percentile(improvements, 2.5),
        "ci_upper": np.percentile(improvements, 97.5)
    }


def run_round1_fix():
    """Run Round 1 fixes: Candidate Recall + Statistical Rigor"""
    print("=" * 60)
    print("XBRL Minimal Repair - Round 1 Fix")
    print("Addressing: Candidate Recall + Statistical Rigor")
    print("=" * 60)

    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    # ============================================================
    # Experiment 1: Candidate Recall Analysis (Codex Criticism #4)
    # ============================================================
    print("\n[Experiment 1] Candidate Recall Analysis...")

    generator = CandidateGeneratorWithRecall()

    recall_results = defaultdict(list)
    for task in tasks:
        recall = generator.compute_recall_at_k(task, k_values=[1, 3, 5, 10])
        for k, v in recall.items():
            recall_results[k].append(v)

    print("\nCandidate Recall@k (Greedy Candidate Generation):")
    for k in [1, 3, 5, 10]:
        rate = np.mean(recall_results[f"recall@{k}"])
        print(f"  Recall@{k}: {rate:.2%}")

    # ============================================================
    # Experiment 2: Oracle Upper Bound Analysis
    # ============================================================
    print("\n[Experiment 2] Oracle Upper Bound (Perfect Reranker)...")

    # Simulate oracle: if target in candidates, oracle gets it right
    oracle_hits = []
    for task in tasks:
        candidates, _ = generator.generate_candidates(task, top_k=5)
        oracle_hits.append(1 if task.target_fact_id in candidates else 0)

    oracle_top1 = np.mean(oracle_hits)
    print(f"  Oracle Top-1 (perfect reranker): {oracle_top1:.2%}")
    print(f"  This is the upper bound for Stage 2")

    # ============================================================
    # Experiment 3: Paired Significance Test (Codex Criticism #2)
    # ============================================================
    print("\n[Experiment 3] Paired Significance Test...")

    set_seed(42)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    # Train XGBoost
    X_train = []
    y_train = []

    for task in train:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_features(task, candidates, violation_scores)

        for i, fid in enumerate(candidates):
            X_train.append(features[i])
            y_train.append(1 if fid == task.target_fact_id else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    greedy_results = []
    xgb_results = []

    for task in test:
        # Greedy
        greedy_pred, _ = generator.generate_candidates(task, top_k=10)
        greedy_results.append(task.evaluate(greedy_pred))

        # XGBoost
        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_features(task, candidates, violation_scores)

        probs = clf.predict_proba(features)[:, 1]
        ranked = [candidates[i] for i in np.argsort(-probs)]

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        xgb_results.append(task.evaluate(ranked))

    # McNemar test
    sig_test = paired_significance_test(greedy_results, xgb_results)

    print("\nPaired Comparison:")
    print(f"  Both correct: {sig_test['both_correct']}")
    print(f"  Both wrong: {sig_test['both_wrong']}")
    print(f"  Greedy only correct: {sig_test['greedy_only']}")
    print(f"  XGB only correct: {sig_test['xgb_only']}")
    print(f"  McNemar statistic: {sig_test['mcnemar_stat']:.4f}")
    print(f"  P-value: {sig_test['p_value']:.4f}")

    if sig_test['p_value'] < 0.05:
        print(f"  ** Significant improvement (p < 0.05) **")
    else:
        print(f"  Not significant at 0.05 level")

    # ============================================================
    # Experiment 4: Bootstrap CI for Improvement
    # ============================================================
    print("\n[Experiment 4] Bootstrap 95% CI for Improvement...")

    greedy_hits = [r["hit_1"] for r in greedy_results]
    xgb_hits = [r["hit_1"] for r in xgb_results]

    ci_result = bootstrap_ci_for_improvement(greedy_hits, xgb_hits, n_bootstrap=1000)

    print(f"  Mean improvement: {ci_result['mean_improvement']:.2%}")
    print(f"  95% CI: [{ci_result['ci_lower']:.2%}, {ci_result['ci_upper']:.2%}]")

    # ============================================================
    # Experiment 5: Per Error Type Analysis with CI
    # ============================================================
    print("\n[Experiment 5] Per Error Type with Confidence Intervals...")

    for error_type in ["sign_flip", "scale_10", "value_shift"]:
        type_test = [t for t in test if t.error_info["error_type"] == error_type]
        if len(type_test) < 5:
            continue

        type_greedy = []
        type_xgb = []

        for task in type_test:
            greedy_pred, _ = generator.generate_candidates(task, top_k=10)
            type_greedy.append(task.evaluate(greedy_pred))

            candidates, violation_scores = generator.generate_candidates(task, top_k=5)
            features = build_features(task, candidates, violation_scores)

            probs = clf.predict_proba(features)[:, 1]
            ranked = [candidates[i] for i in np.argsort(-probs)]

            for f in task.facts:
                if f["id"] not in ranked:
                    ranked.append(f["id"])

            type_xgb.append(task.evaluate(ranked))

        greedy_top1 = np.mean([r["hit_1"] for r in type_greedy])
        xgb_top1 = np.mean([r["hit_1"] for r in type_xgb])

        # Bootstrap CI for this type
        greedy_hits_type = [r["hit_1"] for r in type_greedy]
        xgb_hits_type = [r["hit_1"] for r in type_xgb]

        ci_type = bootstrap_ci_for_improvement(greedy_hits_type, xgb_hits_type, n_bootstrap=500)

        print(f"\n  {error_type} (n={len(type_test)}):")
        print(f"    Greedy: {greedy_top1:.2%}")
        print(f"    XGB: {xgb_top1:.2%}")
        print(f"    Improvement 95% CI: [{ci_type['ci_lower']:.2%}, {ci_type['ci_upper']:.2%}]")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "round": 1,
        "fixes_addressed": ["candidate_recall", "paired_significance", "bootstrap_ci"],
        "candidate_recall": {
            "recall@1": np.mean(recall_results["recall@1"]),
            "recall@3": np.mean(recall_results["recall@3"]),
            "recall@5": np.mean(recall_results["recall@5"]),
            "recall@10": np.mean(recall_results["recall@10"])
        },
        "oracle_upper_bound": oracle_top1,
        "paired_significance": sig_test,
        "bootstrap_ci": ci_result,
        "n_tasks": len(tasks),
        "n_test": len(test)
    }

    with open("data/benchmark/round1_fix_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/round1_fix_results.json")

    return results


if __name__ == "__main__":
    run_round1_fix()
    print("\n" + "=" * 60)
    print("Round 1 Fix Complete")
    print("=" * 60)
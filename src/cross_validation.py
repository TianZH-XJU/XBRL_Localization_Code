"""
XBRL Minimal Repair - Cross-Validation for Robust Estimates
Iteration 7: 5-fold CV to address small test set fragility
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.model_selection import KFold
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


def build_features(task, candidate_ids, violation_scores):
    """Build unified features (38 features, no leakage)"""
    features = []
    epsilon = 1e-6
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    all_values = [abs(f["value"]) for f in task.facts]
    mean_val = np.mean(all_values) if all_values else 0
    std_val = np.std(all_values) if len(all_values) > 1 else 0

    n_violations = len(task.violations)
    total_residual_sum = sum(v["residual"] for v in task.violations)

    for fid in candidate_ids:
        residual_ratio = 0.0
        fact = None
        for f in task.facts:
            if f["id"] == fid:
                fact = f
                break

        if fact is None:
            features.append([0] * 38)
            continue

        feat = []

        is_total_pos = any(f"fact_{v['row']}_{v['total_col']}" == fid for v in task.violations)
        is_component_pos = any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])]
                               for v in task.violations)

        feat.append(1.0 if is_total_pos else 0.0)
        feat.append(1.0 if is_component_pos else 0.0)
        feat.append(1.0 if (is_total_pos or is_component_pos) else 0.0)

        if is_total_pos:
            for v in task.violations:
                if f"fact_{v['row']}_{v['total_col']}" == fid:
                    feat.append(v["residual"] / (max_residual + epsilon))
                    feat.append(abs(fact["value"] - v["expected_sum"]) / (max_value + epsilon))
                    signed_residual = (v["reported_total"] - v["expected_sum"]) / (max_value + epsilon)
                    feat.append(signed_residual)
                    feat.append(1.0)
                    break
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

        if is_component_pos:
            for v in task.violations:
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        other_sum = sum(task._get_cell_value(v["row"], j2) or 0
                                       for j2, _ in v.get("components", [])
                                       if j2 != j)
                        expected_val = v["reported_total"] - other_sum
                        component_discrepancy = abs(fact["value"] - expected_val) / (max_value + epsilon)
                        feat.append(component_discrepancy)
                        feat.append(v["residual"] / (max_residual + epsilon) / len(v["components"]))
                        break
                else:
                    continue
                break
            else:
                feat.extend([0.0, 0.0])
        else:
            feat.extend([0.0, 0.0])

        # Observable patterns
        if is_total_pos:
            for v in task.violations:
                if f"fact_{v['row']}_{v['total_col']}" == fid:
                    residual_ratio = v["residual"] / (abs(fact["value"]) + epsilon)
                    feat.append(residual_ratio)
                    feat.append(1.0 if residual_ratio > 5 else 0.0)
                    break
            else:
                feat.extend([0.0, 0.0])
        elif is_component_pos:
            for v in task.violations:
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        residual_ratio = v["residual"] / (abs(fact["value"]) + epsilon)
                        feat.append(residual_ratio)
                        feat.append(1.0 if residual_ratio > 5 else 0.0)
                        break
                else:
                    continue
                break
            else:
                feat.extend([0.0, 0.0])
        else:
            feat.extend([0.0, 0.0])

        # Sign agreement
        if is_total_pos:
            for v in task.violations:
                if f"fact_{v['row']}_{v['total_col']}" == fid:
                    sign_agreement = 1.0 if (v["reported_total"] * v["expected_sum"] >= 0) else 0.0
                    feat.append(sign_agreement)
                    break
            else:
                feat.append(0.0)
        else:
            feat.append(0.0)

        # Magnitude anomaly
        if is_total_pos:
            for v in task.violations:
                if f"fact_{v['row']}_{v['total_col']}" == fid:
                    magnitude_ratio = abs(fact["value"]) / (abs(v["expected_sum"]) + epsilon)
                    feat.append(magnitude_ratio)
                    feat.append(1.0 if magnitude_ratio > 5 else 0.0)
                    break
            else:
                feat.extend([0.0, 0.0])
        elif is_component_pos:
            for v in task.violations:
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        other_sum = sum(task._get_cell_value(v["row"], j2) or 0
                                       for j2, _ in v.get("components", [])
                                       if j2 != j)
                        expected_val = v["reported_total"] - other_sum
                        magnitude_ratio = abs(fact["value"]) / (abs(expected_val) + epsilon)
                        feat.append(magnitude_ratio)
                        feat.append(1.0 if magnitude_ratio > 5 else 0.0)
                        break
                else:
                    continue
                break
            else:
                feat.extend([0.0, 0.0])
        else:
            feat.extend([0.0, 0.0])

        feat.append(1.0 if is_total_pos and residual_ratio > 5 else 0.0)
        feat.append(1.0 if is_component_pos and residual_ratio > 5 else 0.0)

        feat.append(violation_scores.get(fid, 0) / (max_residual + epsilon))
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, n_violations))
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / (max_residual + epsilon))

        feat.append(fact["value"] / (max_value + epsilon))
        feat.append(abs(fact["value"]) / (max_value + epsilon))
        feat.append(1.0 if fact["value"] < 0 else 0.0)
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)

        feat.append(fact["row"] / 10)
        feat.append(fact["col"] / 5)

        feat.append(n_violations / 5)
        feat.append(total_residual_sum / (max_residual * n_violations + epsilon) if n_violations > 0 else 0)
        if std_val > 0:
            feat.append(abs(abs(fact["value"]) - mean_val) / (std_val + epsilon))
        else:
            feat.append(0.0)

        feat.append(1.0)

        features.append(feat)

    return np.array(features)


class UnifiedRanker:
    def __init__(self):
        self.ranker = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )

    def train(self, train_tasks):
        X_all = []
        y_all = []

        for task in train_tasks:
            if not task.has_violation():
                continue

            candidates, violation_scores = self._get_candidates(task)
            features = build_features(task, candidates, violation_scores)

            for i, fid in enumerate(candidates):
                X_all.append(features[i])
                y_all.append(1 if fid == task.target_fact_id else 0)

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        self.ranker.fit(X_all, y_all)

    def _get_candidates(self, task, top_k=10):
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

    def localize(self, task):
        candidates, violation_scores = self._get_candidates(task)

        if len(candidates) == 0:
            return [f["id"] for f in task.facts]

        features = build_features(task, candidates, violation_scores)

        try:
            probs = self.ranker.predict_proba(features)[:, 1]
            ranked = [candidates[i] for i in np.argsort(-probs)]
        except:
            ranked = candidates

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        return ranked


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


def run_cross_validation():
    """Run 5-fold cross-validation"""
    print("=" * 60)
    print("XBRL Minimal Repair - 5-Fold Cross-Validation")
    print("Iteration 7: Robust estimates with CV")
    print("=" * 60)

    set_seed(42)
    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    unified_all_results = []
    greedy_all_results = []

    fold_unified_top1 = []
    fold_greedy_top1 = []
    fold_unified_total = []
    fold_unified_comp = []

    print("\n[Step 1] Running 5-Fold Cross-Validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(tasks)):
        train = [tasks[i] for i in train_idx]
        test = [tasks[i] for i in test_idx]

        print(f"\n  Fold {fold_idx+1}: Train={len(train)}, Test={len(test)}")

        # Train unified ranker
        ranker = UnifiedRanker()
        ranker.train(train)

        greedy = GreedyBaseline()

        # Evaluate
        unified_results = []
        greedy_results = []

        for task in test:
            unified_pred = ranker.localize(task)
            greedy_pred = greedy.localize(task)

            unified_results.append({
                "eval": task.evaluate(unified_pred),
                "is_total": task.target_is_total,
                "error_type": task.error_info["error_type"]
            })
            greedy_results.append(task.evaluate(greedy_pred))

        unified_top1 = sum(r["eval"]["hit_1"] for r in unified_results) / len(unified_results)
        greedy_top1 = sum(r["hit_1"] for r in greedy_results) / len(greedy_results)

        total_results = [r for r in unified_results if r["is_total"]]
        comp_results = [r for r in unified_results if not r["is_total"]]

        unified_total = sum(r["eval"]["hit_1"] for r in total_results) / len(total_results) if total_results else 0
        unified_comp = sum(r["eval"]["hit_1"] for r in comp_results) / len(comp_results) if comp_results else 0

        print(f"    Unified Top-1: {unified_top1:.2%}")
        print(f"    Greedy Top-1: {greedy_top1:.2%}")
        print(f"    Total-target: {unified_total:.2%}, Component-target: {unified_comp:.2%}")

        fold_unified_top1.append(unified_top1)
        fold_greedy_top1.append(greedy_top1)
        fold_unified_total.append(unified_total)
        fold_unified_comp.append(unified_comp)

        unified_all_results.extend([r["eval"]["hit_1"] for r in unified_results])
        greedy_all_results.extend([r["hit_1"] for r in greedy_results])

    # ============================================================
    # Aggregate results
    # ============================================================
    print("\n[Step 2] Cross-Validation Aggregate Results...")

    mean_unified_top1 = np.mean(fold_unified_top1)
    std_unified_top1 = np.std(fold_unified_top1)
    mean_greedy_top1 = np.mean(fold_greedy_top1)
    std_greedy_top1 = np.std(fold_greedy_top1)

    mean_unified_total = np.mean(fold_unified_total)
    std_unified_total = np.std(fold_unified_total)
    mean_unified_comp = np.mean(fold_unified_comp)
    std_unified_comp = np.std(fold_unified_comp)

    print(f"\nUnified Ranker (5-fold mean ± std):")
    print(f"  Overall Top-1: {mean_unified_top1:.2%} ± {std_unified_top1:.2%}")
    print(f"  Total-target: {mean_unified_total:.2%} ± {std_unified_total:.2%}")
    print(f"  Component-target: {mean_unified_comp:.2%} ± {std_unified_comp:.2%}")

    print(f"\nGreedy Baseline (5-fold mean ± std):")
    print(f"  Overall Top-1: {mean_greedy_top1:.2%} ± {std_greedy_top1:.2%}")

    improvement = mean_unified_top1 - mean_greedy_top1
    print(f"\nImprovement: {improvement:.2%}")

    # ============================================================
    # Paired t-test across folds
    # ============================================================
    print("\n[Step 3] Paired t-test across folds...")

    t_stat, p_value = stats.ttest_rel(fold_unified_top1, fold_greedy_top1)
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {p_value < 0.05}")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "iteration": 7,
        "method": "5_fold_cv",
        "unified": {
            "overall_mean": float(mean_unified_top1),
            "overall_std": float(std_unified_top1),
            "total_mean": float(mean_unified_total),
            "total_std": float(std_unified_total),
            "component_mean": float(mean_unified_comp),
            "component_std": float(std_unified_comp)
        },
        "greedy": {
            "overall_mean": float(mean_greedy_top1),
            "overall_std": float(std_greedy_top1)
        },
        "paired_t_test": {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        },
        "fold_results": {
            "unified_top1": fold_unified_top1,
            "greedy_top1": fold_greedy_top1,
            "unified_total": fold_unified_total,
            "unified_comp": fold_unified_comp
        }
    }

    with open("data/benchmark/cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/cv_results.json")

    return results


if __name__ == "__main__":
    run_cross_validation()
    print("\n" + "=" * 60)
    print("Cross-Validation Complete")
    print("=" * 60)
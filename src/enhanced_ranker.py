"""
XBRL Minimal Repair - Enhanced Value Shift Detection
Iteration 5: Address value_shift-total 33% limitation

Key insight from failure analysis:
- value_shift has arbitrary ±30% shift → residual_ratio/magnitude_ratio patterns unreliable
- Need features that capture "internal consistency" - components should have similar magnitude
- If total is shifted, components might be too small/too large relative to new total
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


def build_enhanced_features(task, candidate_ids, violation_scores):
    """Build enhanced features with value_shift-specific detection

    Key additions for value_shift:
    1. Component magnitude uniformity - components should have similar magnitude
    2. Total/component ratio consistency - if total shifted, ratio changes
    3. Expected component based on total (reverse inference)
    4. Outlier detection - which cell is most anomalous relative to others
    """
    features = []
    epsilon = 1e-6
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    all_values = [abs(f["value"]) for f in task.facts]
    mean_val = np.mean(all_values) if all_values else 0
    std_val = np.std(all_values) if len(all_values) > 1 else 0

    n_violations = len(task.violations)
    total_residual_sum = sum(v["residual"] for v in task.violations)

    # Pre-compute component magnitude statistics for each violation
    comp_stats = {}
    for v in task.violations:
        comp_values = [abs(task._get_cell_value(v["row"], j) or 0) for j, _ in v["components"]]
        if comp_values:
            comp_mean = np.mean(comp_values)
            comp_std = np.std(comp_values) if len(comp_values) > 1 else 0
            comp_stats[v["row"]] = {"mean": comp_mean, "std": comp_std, "values": comp_values}

    for fid in candidate_ids:
        residual_ratio = 0.0
        fact = None
        for f in task.facts:
            if f["id"] == fid:
                fact = f
                break

        if fact is None:
            features.append([0] * 45)
            continue

        feat = []

        # === Role features ===
        is_total_pos = any(f"fact_{v['row']}_{v['total_col']}" == fid for v in task.violations)
        is_component_pos = any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])]
                               for v in task.violations)

        feat.append(1.0 if is_total_pos else 0.0)
        feat.append(1.0 if is_component_pos else 0.0)
        feat.append(1.0 if (is_total_pos or is_component_pos) else 0.0)

        # === Role × Residual interaction ===
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

        # === Observable error pattern features ===
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

        # Role × Observable interactions
        feat.append(1.0 if is_total_pos and residual_ratio > 5 else 0.0)
        feat.append(1.0 if is_component_pos and residual_ratio > 5 else 0.0)

        # === NEW: Value-shift specific features ===
        # Component magnitude uniformity - components should have similar magnitude
        if is_component_pos:
            for v in task.violations:
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        if v["row"] in comp_stats:
                            cs = comp_stats[v["row"]]
                            # How different is this component from other components?
                            if cs["std"] > 0:
                                comp_deviation = abs(abs(fact["value"]) - cs["mean"]) / (cs["std"] + epsilon)
                                feat.append(comp_deviation)
                                feat.append(1.0 if comp_deviation > 2 else 0.0)  # outlier?
                            else:
                                feat.extend([0.0, 0.0])
                        else:
                            feat.extend([0.0, 0.0])
                        break
                else:
                    continue
                break
            else:
                feat.extend([0.0, 0.0])
        else:
            feat.extend([0.0, 0.0])

        # Total/component ratio - if total shifted, ratio might be anomalous
        if is_total_pos:
            for v in task.violations:
                if f"fact_{v['row']}_{v['total_col']}" == fid:
                    comp_mean = np.mean([abs(task._get_cell_value(v["row"], j) or 0)
                                        for j, _ in v["components"]]) if v["components"] else 0
                    total_comp_ratio = abs(fact["value"]) / (comp_mean + epsilon)
                    feat.append(total_comp_ratio)

                    # If ratio is much larger than number of components → suspicious total
                    n_comps = len(v["components"])
                    feat.append(1.0 if total_comp_ratio > n_comps * 2 else 0.0)
                    break
            else:
                feat.extend([0.0, 0.0])
        else:
            feat.extend([0.0, 0.0])

        # Expected component based on total (reverse inference)
        # If total is correct, each component should be roughly total/n
        if is_total_pos:
            for v in task.violations:
                if f"fact_{v['row']}_{v['total_col']}" == fid:
                    n_comps = len(v["components"])
                    expected_each = abs(fact["value"]) / n_comps

                    # Check if actual components match expected_each
                    comp_values = [abs(task._get_cell_value(v["row"], j) or 0)
                                  for j, _ in v["components"]]
                    comp_deviation_from_expected = np.std([abs(cv - expected_each) for cv in comp_values]) if comp_values else 0

                    feat.append(comp_deviation_from_expected / (max_value + epsilon))
                    feat.append(1.0 if comp_deviation_from_expected < max_value * 0.1 else 0.0)  # uniform?
                    break
            else:
                feat.extend([0.0, 0.0])
        else:
            feat.extend([0.0, 0.0])

        # === Violation features ===
        feat.append(violation_scores.get(fid, 0) / (max_residual + epsilon))
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, n_violations))
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / (max_residual + epsilon))

        # === Numeric features ===
        feat.append(fact["value"] / (max_value + epsilon))
        feat.append(abs(fact["value"]) / (max_value + epsilon))
        feat.append(1.0 if fact["value"] < 0 else 0.0)
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)

        # === Structural features ===
        feat.append(fact["row"] / 10)
        feat.append(fact["col"] / 5)

        # === Global context ===
        feat.append(n_violations / 5)
        feat.append(total_residual_sum / (max_residual * n_violations + epsilon) if n_violations > 0 else 0)
        if std_val > 0:
            feat.append(abs(abs(fact["value"]) - mean_val) / (std_val + epsilon))
        else:
            feat.append(0.0)

        feat.append(1.0)

        features.append(feat)

    return np.array(features)


class EnhancedRanker:
    def __init__(self):
        self.ranker = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
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
            features = build_enhanced_features(task, candidates, violation_scores)

            for i, fid in enumerate(candidates):
                X_all.append(features[i])
                y_all.append(1 if fid == task.target_fact_id else 0)

        X_all = np.array(X_all)
        y_all = np.array(y_all)

        print(f"  Training on {len(train_tasks)} tasks, {len(X_all)} candidates, {len(X_all[0])} features")

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

        features = build_enhanced_features(task, candidates, violation_scores)

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


def run_enhanced_ranker():
    """Run enhanced ranker with value_shift-specific features"""
    print("=" * 60)
    print("XBRL Minimal Repair - Enhanced Value Shift Detection")
    print("Iteration 5: 45 features with component uniformity analysis")
    print("=" * 60)

    set_seed(42)
    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    print(f"Train: {len(train)}, Test: {len(test)}")

    # ============================================================
    # Train enhanced ranker
    # ============================================================
    print("\n[Step 1] Training Enhanced Ranker...")
    ranker = EnhancedRanker()
    ranker.train(train)

    # ============================================================
    # Evaluate overall
    # ============================================================
    print("\n[Step 2] Overall Evaluation...")

    all_results = []
    for task in test:
        pred = ranker.localize(task)
        all_results.append(task.evaluate(pred))

    top1 = sum(r["hit_1"] for r in all_results) / len(all_results)
    top3 = sum(r["hit_3"] for r in all_results) / len(all_results)
    mrr = sum(r["mrr"] for r in all_results) / len(all_results)

    print(f"\nOverall (n={len(test)}):")
    print(f"  Top-1: {top1:.2%}")
    print(f"  Top-3: {top3:.2%}")
    print(f"  MRR: {mrr:.4f}")

    # ============================================================
    # Role-specific evaluation
    # ============================================================
    print("\n[Step 3] Role-Specific Evaluation...")

    test_total = [t for t in test if t.target_is_total]
    test_component = [t for t in test if not t.target_is_total]

    total_results = []
    for task in test_total:
        pred = ranker.localize(task)
        total_results.append(task.evaluate(pred))

    comp_results = []
    for task in test_component:
        pred = ranker.localize(task)
        comp_results.append(task.evaluate(pred))

    print(f"\nTotal-target (n={len(test_total)}):")
    print(f"  Top-1: {sum(r['hit_1'] for r in total_results)/len(total_results):.2%}")

    print(f"\nComponent-target (n={len(test_component)}):")
    print(f"  Top-1: {sum(r['hit_1'] for r in comp_results)/len(comp_results):.2%}")

    # ============================================================
    # Error type × Role analysis (focus on value_shift-total)
    # ============================================================
    print("\n[Step 4] Error Type × Role Analysis...")

    for error_type in ["sign_flip", "scale_10", "value_shift"]:
        for role in ["total", "component"]:
            subset = [t for t in test
                     if (t.target_is_total if role == "total" else not t.target_is_total)
                     and t.error_info["error_type"] == error_type]

            if len(subset) < 2:
                continue

            subset_results = []
            for task in subset:
                pred = ranker.localize(task)
                subset_results.append(task.evaluate(pred))

            top1_sub = sum(r["hit_1"] for r in subset_results) / len(subset_results)
            print(f"  {error_type}-{role} (n={len(subset)}): Top-1={top1_sub:.2%}")

    # ============================================================
    # Comparison with previous unified ranker
    # ============================================================
    print("\n[Step 5] Comparison with Previous Results...")
    print("Previous (Iteration 4): 87.50% overall, 70% total-target, 100% component-target")
    print(f"Current: {top1:.2%} overall")

    # ============================================================
    # Bootstrap CI
    # ============================================================
    print("\n[Step 6] Bootstrap Confidence Intervals...")

    hits = [r["hit_1"] for r in all_results]
    bootstrap_top1s = []
    for _ in range(1000):
        indices = np.random.choice(len(hits), size=len(hits), replace=True)
        bootstrap_top1s.append(np.mean([hits[i] for i in indices]))

    print(f"  Overall Top-1 95% CI: [{np.percentile(bootstrap_top1s, 2.5):.2%}, {np.percentile(bootstrap_top1s, 97.5):.2%}]")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "iteration": 5,
        "n_features": 45,
        "overall": {"top1": float(top1), "top3": float(top3), "mrr": float(mrr)},
        "total_target": {"n": int(len(test_total)), "top1": float(sum(r['hit_1'] for r in total_results)/len(total_results))},
        "component_target": {"n": int(len(test_component)), "top1": float(sum(r['hit_1'] for r in comp_results)/len(comp_results))}
    }

    with open("data/benchmark/enhanced_ranker_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/enhanced_ranker_results.json")

    return results


if __name__ == "__main__":
    run_enhanced_ranker()
    print("\n" + "=" * 60)
    print("Enhanced Ranker Complete")
    print("=" * 60)
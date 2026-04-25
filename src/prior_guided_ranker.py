"""
XBRL Minimal Repair - Role Prior + LambdaMART-style Ranking
Iteration 6: Use role probability prior to guide ranking

Key insight:
- Total-target errors have different violation patterns than component-target
- If residual direction suggests total is too high/low relative to components → likely total error
- Use role prior probability to break ties in ranking
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRanker
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


def compute_role_prior(task):
    """Compute probability that root cause is total vs component

    Heuristic based on violation patterns:
    - If total > expected_sum and all components are smaller → likely total too high (scale_10 or sign_flip on total)
    - If total < expected_sum and components sum exceeds total → likely component too high
    - If residual is small relative to total magnitude → likely value_shift
    - If components have uniform magnitude → total shift is more plausible
    """
    if not task.violations:
        return {"total": 0.5, "component": 0.5}

    total_prior = 0.5
    epsilon = 1e-6

    for v in task.violations:
        reported_total = v["reported_total"]
        expected_sum = v["expected_sum"]
        residual = v["residual"]

        # Sign of residual
        residual_sign = reported_total - expected_sum

        # Component magnitudes
        comp_values = [abs(task._get_cell_value(v["row"], j) or 0) for j, _ in v["components"]]
        comp_mean = np.mean(comp_values) if comp_values else 0
        comp_std = np.std(comp_values) if len(comp_values) > 1 else 0

        # Ratio of total to component mean
        total_comp_ratio = abs(reported_total) / (comp_mean + epsilon)

        n_comps = len(v["components"])

        # Heuristic 1: If total >> expected_sum and components are uniform
        # → likely scale_10 on total or total sign_flip
        if residual_sign > 0 and comp_std < comp_mean * 0.5:
            # Total is too high, components are uniform → suspect total
            total_prior += 0.15

        # Heuristic 2: If total < expected_sum
        # → could be component too high OR total too low (ambiguous)
        elif residual_sign < 0:
            # Check if any component is anomalously large
            max_comp = max(comp_values) if comp_values else 0
            if max_comp > comp_mean * 2 and comp_std > comp_mean * 0.5:
                # One component is outlier → suspect that component
                total_prior -= 0.1
            else:
                # Components are uniform, total might be too low
                total_prior += 0.1

        # Heuristic 3: If total_comp_ratio >> n_comps
        # → total is much larger than it should be relative to components
        if total_comp_ratio > n_comps * 1.5:
            total_prior += 0.2

        # Heuristic 4: If residual is small relative to total (value_shift pattern)
        residual_ratio = residual / (abs(reported_total) + epsilon)
        if residual_ratio < 0.4:  # Small shift → could be either
            # Don't adjust much, stay neutral
            pass

    # Clamp to [0.1, 0.9]
    total_prior = max(0.1, min(0.9, total_prior))
    component_prior = 1.0 - total_prior

    return {"total": total_prior, "component": component_prior}


def build_features_with_prior(task, candidate_ids, violation_scores, role_prior):
    """Build features with role prior incorporated"""
    features = []
    epsilon = 1e-6
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    all_values = [abs(f["value"]) for f in task.facts]
    mean_val = np.mean(all_values) if all_values else 0
    std_val = np.std(all_values) if len(all_values) > 1 else 0

    n_violations = len(task.violations)

    for fid in candidate_ids:
        residual_ratio = 0.0
        fact = None
        for f in task.facts:
            if f["id"] == fid:
                fact = f
                break

        if fact is None:
            features.append([0] * 40)
            continue

        feat = []

        # === Role features ===
        is_total_pos = any(f"fact_{v['row']}_{v['total_col']}" == fid for v in task.violations)
        is_component_pos = any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])]
                               for v in task.violations)

        feat.append(1.0 if is_total_pos else 0.0)
        feat.append(1.0 if is_component_pos else 0.0)

        # === ROLE PRIOR FEATURES (key innovation) ===
        # Multiply candidate score by role prior probability
        total_prior = role_prior["total"]
        comp_prior = role_prior["component"]

        if is_total_pos:
            feat.append(total_prior)  # prior boost for total candidates
        else:
            feat.append(0.0)

        if is_component_pos:
            feat.append(comp_prior)  # prior boost for component candidates
        else:
            feat.append(0.0)

        # Role × Prior interaction
        feat.append(1.0 if is_total_pos and total_prior > 0.6 else 0.0)  # high total suspicion
        feat.append(1.0 if is_component_pos and comp_prior > 0.6 else 0.0)  # high comp suspicion

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

        # === Observable patterns ===
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

        # === Violation features ===
        feat.append(violation_scores.get(fid, 0) / (max_residual + epsilon))
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, n_violations))

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
        feat.append(total_prior)  # global role prior
        feat.append(comp_prior)

        if std_val > 0:
            feat.append(abs(abs(fact["value"]) - mean_val) / (std_val + epsilon))
        else:
            feat.append(0.0)

        feat.append(1.0)

        features.append(feat)

    return np.array(features)


class PriorGuidedRanker:
    """Ranker with role prior to guide tie-breaking"""

    def __init__(self):
        self.ranker = XGBClassifier(
            n_estimators=120,
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
            role_prior = compute_role_prior(task)
            features = build_features_with_prior(task, candidates, violation_scores, role_prior)

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

        role_prior = compute_role_prior(task)
        features = build_features_with_prior(task, candidates, violation_scores, role_prior)

        try:
            probs = self.ranker.predict_proba(features)[:, 1]
            ranked = [candidates[i] for i in np.argsort(-probs)]
        except:
            ranked = candidates

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


def run_prior_guided():
    """Run prior-guided ranker"""
    print("=" * 60)
    print("XBRL Minimal Repair - Role Prior Guided Ranking")
    print("Iteration 6: Role prior to break ties")
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
    # Train
    # ============================================================
    print("\n[Step 1] Training Prior-Guided Ranker...")
    ranker = PriorGuidedRanker()
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
    # Role-specific
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
    # Error × Role
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
    # Compare
    # ============================================================
    print("\n[Step 5] Comparison...")
    print("Previous (Iteration 4-5): 87.50% overall, 70% total, 100% component")
    print(f"Current: {top1:.2%} overall")

    # ============================================================
    # Bootstrap
    # ============================================================
    print("\n[Step 6] Bootstrap CI...")

    hits = [r["hit_1"] for r in all_results]
    bootstrap_top1s = []
    for _ in range(1000):
        indices = np.random.choice(len(hits), size=len(hits), replace=True)
        bootstrap_top1s.append(np.mean([hits[i] for i in indices]))

    print(f"  Overall Top-1 95% CI: [{np.percentile(bootstrap_top1s, 2.5):.2%}, {np.percentile(bootstrap_top1s, 97.5):.2%}]")

    # ============================================================
    # Save
    # ============================================================
    results = {
        "iteration": 6,
        "n_features": 40,
        "overall": {"top1": float(top1), "top3": float(top3), "mrr": float(mrr)},
        "total_target": {"n": int(len(test_total)), "top1": float(sum(r['hit_1'] for r in total_results)/len(total_results))},
        "component_target": {"n": int(len(test_component)), "top1": float(sum(r['hit_1'] for r in comp_results)/len(comp_results))}
    }

    with open("data/benchmark/prior_guided_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/prior_guided_results.json")

    return results


if __name__ == "__main__":
    run_prior_guided()
    print("\n" + "=" * 60)
    print("Prior-Guided Ranker Complete")
    print("=" * 60)
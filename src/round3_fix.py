"""
XBRL Minimal Repair - Round 3 Fix
Critical Fix: Role-aware classifier to solve total-target 0% issue
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

        # Determine if target is total or component position
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
        """Check if the target fact is in a total position"""
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


def build_role_specific_features(task, candidate_ids, violation_scores, role="total"):
    """Build role-specific features for total or component position"""
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
            features.append([0] * 20)
            continue

        feat = []

        # Determine if this fact is total or component position
        is_total_pos = any(f"fact_{v['row']}_{v['total_col']}" == fid for v in task.violations)
        is_component_pos = any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])]
                               for v in task.violations)

        # Role indicator
        feat.append(1.0 if is_total_pos else 0.0)
        feat.append(1.0 if is_component_pos else 0.0)

        # Role-specific features

        # Violation features
        feat.append(violation_scores.get(fid, 0) / max_residual)
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, len(task.violations)))
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / max_residual)

        # Numeric features
        feat.append(fact["value"] / max_value)
        feat.append(abs(fact["value"]) / max_value)
        feat.append(1.0 if fact["value"] < 0 else 0.0)

        # Scale features
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)

        # Structural features
        feat.append(fact["row"] / 10)
        feat.append(fact["col"] / 5)

        # === Total-specific features (addressing 0% failure) ===
        if is_total_pos:
            # Feature: expected_total vs reported_total discrepancy
            for v in task.violations:
                if f"fact_{v['row']}_{v['total_col']}" == fid:
                    # Total's discrepancy: |reported_total - expected_sum|
                    total_discrepancy = abs(fact["value"] - v["expected_sum"]) / max_value
                    feat.append(total_discrepancy)

                    # Can changing this total fix all violations?
                    # If total is wrong, fixing it alone should resolve residual
                    fix_potential = v["residual"] / max_residual
                    feat.append(fix_potential)

                    # Signed residual direction
                    signed_residual = (v["reported_total"] - v["expected_sum"]) / max_value
                    feat.append(signed_residual)
                    break
            else:
                feat.extend([0.0, 0.0, 0.0])
        else:
            feat.extend([0.0, 0.0, 0.0])

        # === Component-specific features ===
        if is_component_pos:
            # If this component is wrong, what should it be?
            for v in task.violations:
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        # Expected value for this component
                        other_sum = sum(task._get_cell_value(v["row"], j2) or 0
                                       for j2, _ in v.get("components", [])
                                       if j2 != j)
                        expected_val = v["reported_total"] - other_sum
                        component_discrepancy = abs(fact["value"] - expected_val) / max_value
                        feat.append(component_discrepancy)
                        break
                else:
                    continue
                break
            else:
                feat.append(0.0)
        else:
            feat.append(0.0)

        # Global context
        if std_val > 0:
            feat.append(abs(abs(fact["value"]) - mean_val) / std_val)
        else:
            feat.append(0.0)

        feat.append(1.0)

        features.append(feat)

    return np.array(features)


class RoleAwareClassifier:
    """Two-stage role-aware classifier: predict role, then rank within role"""

    def __init__(self):
        self.role_classifier = LogisticRegression(random_state=42)
        self.total_reranker = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.component_reranker = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)

    def train(self, train_tasks):
        """Train role classifier and role-specific rerankers"""

        # First: train role classifier
        X_role = []
        y_role = []

        for task in train_tasks:
            if not task.has_violation():
                continue

            candidates, violation_scores = self._get_candidates(task)

            for i, fid in enumerate(candidates):
                # Role label: 1 if total position, 0 if component
                is_total = any(f"fact_{v['row']}_{v['total_col']}" == fid for v in task.violations)
                X_role.append([violation_scores.get(fid, 0), is_total])
                y_role.append(1 if fid == task.target_fact_id and task.target_is_total else 0)

        X_role = np.array(X_role)
        y_role = np.array(y_role)

        # For role classification, we use simpler features
        # Actually, we don't need a role classifier if we separate by position

        # Train role-specific rerankers
        total_tasks = [t for t in train_tasks if t.target_is_total]
        component_tasks = [t for t in train_tasks if not t.target_is_total]

        # Train total reranker
        if len(total_tasks) >= 3:
            X_total = []
            y_total = []

            for task in total_tasks:
                candidates, violation_scores = self._get_candidates(task)
                # Filter to total-position candidates only
                total_candidates = [fid for fid in candidates
                                    if any(f"fact_{v['row']}_{v['total_col']}" == fid for v in task.violations)]

                if len(total_candidates) == 0:
                    continue

                features = build_role_specific_features(task, total_candidates, violation_scores, "total")

                for i, fid in enumerate(total_candidates):
                    X_total.append(features[i])
                    y_total.append(1 if fid == task.target_fact_id else 0)

            if len(X_total) > 0 and len(np.unique(y_total)) > 1:
                X_total = np.array(X_total)
                y_total = np.array(y_total)
                self.total_reranker.fit(X_total, y_total)
                print(f"  Trained total reranker on {len(total_tasks)} total-target tasks")

        # Train component reranker
        if len(component_tasks) >= 3:
            X_comp = []
            y_comp = []

            for task in component_tasks:
                candidates, violation_scores = self._get_candidates(task)
                # Filter to component-position candidates only
                comp_candidates = [fid for fid in candidates
                                   if any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])]
                                          for v in task.violations)]

                if len(comp_candidates) == 0:
                    continue

                features = build_role_specific_features(task, comp_candidates, violation_scores, "component")

                for i, fid in enumerate(comp_candidates):
                    X_comp.append(features[i])
                    y_comp.append(1 if fid == task.target_fact_id else 0)

            if len(X_comp) > 0 and len(np.unique(y_comp)) > 1:
                X_comp = np.array(X_comp)
                y_comp = np.array(y_comp)
                self.component_reranker.fit(X_comp, y_comp)
                print(f"  Trained component reranker on {len(component_tasks)} component-target tasks")

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
        """Role-aware localization"""
        candidates, violation_scores = self._get_candidates(task)

        if len(candidates) == 0:
            return [f["id"] for f in task.facts]

        # Separate candidates by role
        total_candidates = [fid for fid in candidates
                           if any(f"fact_{v['row']}_{v['total_col']}" == fid for v in task.violations)]
        comp_candidates = [fid for fid in candidates
                          if any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])]
                                 for v in task.violations)]

        # First decide which role is more likely to contain root cause
        # Heuristic: check residual direction and magnitude
        total_residual_sum = sum(v["residual"] for v in task.violations
                                 if v["reported_total"] != v["expected_sum"])

        # If residual is large and total values are anomalous, suspect total
        # If components have large discrepancies, suspect component

        # For now, use simple heuristic + reranking
        ranked = []
        candidate_probs = {}

        # Rerank total candidates and store probabilities
        if len(total_candidates) > 0:
            total_features = build_role_specific_features(task, total_candidates, violation_scores, "total")

            try:
                probs = self.total_reranker.predict_proba(total_features)[:, 1]
                for i, fid in enumerate(total_candidates):
                    candidate_probs[fid] = probs[i]
            except:
                for fid in total_candidates:
                    candidate_probs[fid] = 0.5

        # Rerank component candidates and store probabilities
        if len(comp_candidates) > 0:
            comp_features = build_role_specific_features(task, comp_candidates, violation_scores, "component")

            try:
                probs = self.component_reranker.predict_proba(comp_features)[:, 1]
                for i, fid in enumerate(comp_candidates):
                    candidate_probs[fid] = probs[i]
            except:
                for fid in comp_candidates:
                    candidate_probs[fid] = 0.5

        # Combine ALL candidates by their probability scores (not role-first)
        ranked = sorted(candidate_probs.keys(), key=lambda x: candidate_probs[x], reverse=True)

        # Add remaining facts
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


def run_round3_fix():
    """Run Round 3 fix: Role-aware classifier"""
    print("=" * 60)
    print("XBRL Minimal Repair - Round 3 Fix")
    print("Addressing: Total-target 0% failure with role-aware classifier")
    print("=" * 60)

    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    # Count total vs component targets
    total_targets = sum(1 for t in tasks if t.target_is_total)
    component_targets = len(tasks) - total_targets
    print(f"Total-target tasks: {total_targets}")
    print(f"Component-target tasks: {component_targets}")

    # ============================================================
    # Split data
    # ============================================================
    set_seed(42)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    print(f"\nTrain: {len(train)}, Test: {len(test)}")

    # ============================================================
    # Train role-aware classifier
    # ============================================================
    print("\n[Step 1] Training Role-Aware Classifier...")

    role_classifier = RoleAwareClassifier()
    role_classifier.train(train)

    # ============================================================
    # Evaluate by role
    # ============================================================
    print("\n[Step 2] Evaluating by Role...")

    # Split test by role
    test_total = [t for t in test if t.target_is_total]
    test_component = [t for t in test if not t.target_is_total]

    print(f"Test total-target: {len(test_total)}")
    print(f"Test component-target: {len(test_component)}")

    # Evaluate on total-target tasks
    if len(test_total) > 0:
        total_results = []
        for task in test_total:
            pred = role_classifier.localize(task)
            total_results.append(task.evaluate(pred))

        total_top1 = sum(r["hit_1"] for r in total_results) / len(total_results)
        total_top3 = sum(r["hit_3"] for r in total_results) / len(total_results)

        print(f"\nTotal-target tasks (n={len(test_total)}):")
        print(f"  Top-1: {total_top1:.2%}")
        print(f"  Top-3: {total_top3:.2%}")

        if total_top1 > 0:
            print(f"  ** IMPROVED from 0% **")
        else:
            print(f"  Still 0% - need further fixes")

    # Evaluate on component-target tasks
    if len(test_component) > 0:
        comp_results = []
        for task in test_component:
            pred = role_classifier.localize(task)
            comp_results.append(task.evaluate(pred))

        comp_top1 = sum(r["hit_1"] for r in comp_results) / len(comp_results)
        comp_top3 = sum(r["hit_3"] for r in comp_results) / len(comp_results)

        print(f"\nComponent-target tasks (n={len(test_component)}):")
        print(f"  Top-1: {comp_top1:.2%}")
        print(f"  Top-3: {comp_top3:.2%}")

    # ============================================================
    # Overall evaluation
    # ============================================================
    print("\n[Step 3] Overall Evaluation...")

    all_results = []
    for task in test:
        pred = role_classifier.localize(task)
        all_results.append(task.evaluate(pred))

    overall_top1 = sum(r["hit_1"] for r in all_results) / len(all_results)
    overall_top3 = sum(r["hit_3"] for r in all_results) / len(all_results)
    overall_mrr = sum(r["mrr"] for r in all_results) / len(all_results)

    print(f"\nOverall (n={len(test)}):")
    print(f"  Top-1: {overall_top1:.2%}")
    print(f"  Top-3: {overall_top3:.2%}")
    print(f"  MRR: {overall_mrr:.4f}")

    # ============================================================
    # Error type × Role analysis
    # ============================================================
    print("\n[Step 4] Error Type × Role Analysis...")

    for error_type in ["sign_flip", "scale_10", "value_shift"]:
        for role in ["total", "component"]:
            filtered = [t for t in test
                        if t.error_info["error_type"] == error_type
                        and (t.target_is_total if role == "total" else not t.target_is_total)]

            if len(filtered) < 2:
                continue

            role_results = []
            for task in filtered:
                pred = role_classifier.localize(task)
                role_results.append(task.evaluate(pred))

            top1 = sum(r["hit_1"] for r in role_results) / len(role_results)
            print(f"  {error_type}-{role} (n={len(filtered)}): Top-1={top1:.2%}")

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "round": 3,
        "fix": "role_aware_classifier",
        "n_tasks": len(tasks),
        "n_total_targets": total_targets,
        "n_component_targets": component_targets,
        "test_total_top1": total_top1 if len(test_total) > 0 else None,
        "test_component_top1": comp_top1 if len(test_component) > 0 else None,
        "overall_top1": overall_top1,
        "overall_top3": overall_top3
    }

    with open("data/benchmark/round3_fix_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/round3_fix_results.json")

    return results


if __name__ == "__main__":
    run_round3_fix()
    print("\n" + "=" * 60)
    print("Round 3 Fix Complete")
    print("=" * 60)
"""
XBRL Minimal Repair - Iteration 11 (Final)
Submission Package: Corrected Stats + Ablation + Case Studies
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


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
        "error_type": sub_type
    }

    return modified, error_info


class ViolationGreedyCandidateGenerator:
    def generate_candidates(self, task, top_k=5):
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


def build_features_with_groups(task, candidate_ids, violation_scores, include_value_shift_specific=True):
    """构建特征，可控制是否包含value_shift专属特征"""
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
            if include_value_shift_specific:
                features.append([0] * 24)
            else:
                features.append([0] * 18)
            continue

        feat = []

        # Violation features (4)
        feat.append(violation_scores.get(fid, 0) / max_residual)
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, len(task.violations)))
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / max_residual)
        if task.violations:
            feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / sum(v["residual"] for v in task.violations))
        else:
            feat.append(0.0)

        # Role features (2)
        feat.append(1.0 if any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations) else 0.0)
        feat.append(1.0 if any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])] for v in task.violations) else 0.0)

        # Numeric features (3)
        feat.append(fact["value"] / max_value)
        feat.append(abs(fact["value"]) / max_value)
        feat.append(1.0 if fact["value"] < 0 else 0.0)

        # Scale features (2)
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)
        is_total = any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations)
        if is_total:
            diffs = [abs(fact["value"] - v["expected_sum"]) / max_value for v in task.violations
                     if abs(fact["value"] - v["reported_total"]) < 0.1]
            feat.append(min(diffs) if diffs else 0.0)
        else:
            feat.append(0.0)

        # Structural features (2)
        feat.append(fact["row"] / 10)
        feat.append(fact["col"] / 5)

        # Discrepancy features (4)
        discrepancy_score = 0.0
        for v in task.violations:
            if fid in v["involved_facts"]:
                for j, comp_val in v.get("components", []):
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

        # value_shift specific features (only if enabled)
        if include_value_shift_specific:
            # absolute_shift
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

            # row_median_dev
            row_values = [abs(f["value"]) for f in task.facts if f["row"] == fact["row"]]
            if row_values:
                row_median = np.median(row_values)
                feat.append(abs(abs(fact["value"]) - row_median) / max_value)
            else:
                feat.append(0.0)

            # col_median_dev
            col_values = [abs(f["value"]) for f in task.facts if f["col"] == fact["col"]]
            if col_values:
                col_median = np.median(col_values)
                feat.append(abs(abs(fact["value"]) - col_median) / max_value)
            else:
                feat.append(0.0)

            # neighbor_dev
            neighbors = []
            for f in task.facts:
                if abs(f["row"] - fact["row"]) <= 1 and abs(f["col"] - fact["col"]) <= 1 and f["id"] != fid:
                    neighbors.append(f["value"])

            if neighbors:
                neighbor_mean = np.mean(neighbors)
                feat.append(abs(fact["value"] - neighbor_mean) / max_value)
            else:
                feat.append(0.0)

            # sign_consistency
            same_row_facts = [f for f in task.facts if f["row"] == fact["row"] and f["id"] != fid]
            if same_row_facts:
                sign_match = sum(1 for f in same_row_facts if (f["value"] < 0) == (fact["value"] < 0))
                feat.append(sign_match / len(same_row_facts))
            else:
                feat.append(0.0)

            # repair_potential
            repair_potential = 0.0
            for v in task.violations:
                if fid in v["involved_facts"]:
                    repair_potential += v["residual"]

            feat.append(repair_potential / (max_residual * len(task.violations)) if task.violations else 0.0)

        features.append(feat)

    return np.array(features)


def model_rerank(clf, task, generator, include_value_shift_specific=True):
    candidates, violation_scores = generator.generate_candidates(task, top_k=5)

    if len(candidates) == 0:
        return [f["id"] for f in task.facts]

    features = build_features_with_groups(task, candidates, violation_scores, include_value_shift_specific)

    probs = clf.predict_proba(features)[:, 1]
    ranked = [candidates[i] for i in np.argsort(-probs)]

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


def correct_repeated_cv(tasks, seeds=[42, 123, 456]):
    """修正统计口径的Repeated CV - 计算per-run Top-1均值和标准差"""
    generator = ViolationGreedyCandidateGenerator()

    run_results = []

    for seed in seeds:
        set_seed(seed)
        indices = list(range(len(tasks)))
        random.shuffle(indices)

        # 5-fold
        n_splits = 5
        fold_size = len(tasks) // n_splits
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_splits)]

        for fold_idx in range(n_splits):
            test_indices = folds[fold_idx]
            train_indices = [i for i in indices if i not in test_indices]

            train_tasks = [tasks[i] for i in train_indices]
            test_tasks = [tasks[i] for i in test_indices]

            # Build training data
            X_train = []
            y_train = []

            for task in train_tasks:
                if not task.has_violation():
                    continue

                candidates, violation_scores = generator.generate_candidates(task, top_k=5)
                features = build_features_with_groups(task, candidates, violation_scores, True)

                for i, fid in enumerate(candidates):
                    X_train.append(features[i])
                    y_train.append(1 if fid == task.target_fact_id else 0)

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            if len(np.unique(y_train)) < 2:
                continue

            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=5, random_state=42
            )
            clf.fit(X_train, y_train)

            # Evaluate on test
            greedy_hits = []
            xgb_hits = []

            for t in test_tasks:
                greedy_pred = generator.generate_candidates(t, top_k=5)[0]
                greedy_hits.append(1 if greedy_pred[0] == t.target_fact_id else 0)

                xgb_pred = model_rerank(clf, t, generator, True)
                xgb_hits.append(1 if xgb_pred[0] == t.target_fact_id else 0)

            # Per-run Top-1
            run_greedy_top1 = np.mean(greedy_hits)
            run_xgb_top1 = np.mean(xgb_hits)

            run_results.append({
                "seed": seed,
                "fold": fold_idx,
                "greedy_top1": run_greedy_top1,
                "xgb_top1": run_xgb_top1
            })

    # Compute mean and std across runs
    greedy_top1s = [r["greedy_top1"] for r in run_results]
    xgb_top1s = [r["xgb_top1"] for r in run_results]

    return {
        "greedy": {
            "mean": np.mean(greedy_top1s),
            "std": np.std(greedy_top1s),
            "runs": greedy_top1s
        },
        "xgb": {
            "mean": np.mean(xgb_top1s),
            "std": np.std(xgb_top1s),
            "runs": xgb_top1s
        }
    }


def value_shift_ablation(tasks, seed=42):
    """value_shift专项ablation"""
    set_seed(seed)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    generator = ViolationGreedyCandidateGenerator()

    # Filter to value_shift only
    value_shift_test = [t for t in test if t.error_info["error_type"] == "value_shift"]

    if len(value_shift_test) == 0:
        return {}

    # Train full model
    X_train_full = []
    y_train = []

    for task in train:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_features_with_groups(task, candidates, violation_scores, True)

        for i, fid in enumerate(candidates):
            X_train_full.append(features[i])
            y_train.append(1 if fid == task.target_fact_id else 0)

    X_train_full = np.array(X_train_full)
    y_train = np.array(y_train)

    # Train without value_shift features
    X_train_no_vs = []
    y_train_no_vs = []

    for task in train:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_features_with_groups(task, candidates, violation_scores, False)

        for i, fid in enumerate(candidates):
            X_train_no_vs.append(features[i])
            y_train_no_vs.append(1 if fid == task.target_fact_id else 0)

    X_train_no_vs = np.array(X_train_no_vs)
    y_train_no_vs = np.array(y_train_no_vs)

    # Train models
    clf_full = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf_full.fit(X_train_full, y_train)

    clf_no_vs = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf_no_vs.fit(X_train_no_vs, y_train_no_vs)

    # Evaluate on value_shift test
    results = {}

    # Greedy baseline
    greedy_hits = []
    for t in value_shift_test:
        pred = generator.generate_candidates(t, top_k=5)[0]
        greedy_hits.append(1 if pred[0] == t.target_fact_id else 0)
    results["greedy"] = np.mean(greedy_hits)

    # Full model
    full_hits = []
    for t in value_shift_test:
        pred = model_rerank(clf_full, t, generator, True)
        full_hits.append(1 if pred[0] == t.target_fact_id else 0)
    results["full_with_vs_features"] = np.mean(full_hits)

    # No value_shift features
    no_vs_hits = []
    for t in value_shift_test:
        pred = model_rerank(clf_no_vs, t, generator, False)
        no_vs_hits.append(1 if pred[0] == t.target_fact_id else 0)
    results["without_vs_features"] = np.mean(no_vs_hits)

    return results


def case_studies(tasks, seed=42):
    """生成case studies"""
    set_seed(seed)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    generator = ViolationGreedyCandidateGenerator()

    # Train model
    X_train = []
    y_train = []

    for task in train:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_features_with_groups(task, candidates, violation_scores, True)

        for i, fid in enumerate(candidates):
            X_train.append(features[i])
            y_train.append(1 if fid == task.target_fact_id else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Find cases where greedy fails but model succeeds
    cases = []

    for t in test:
        greedy_pred = generator.generate_candidates(t, top_k=5)[0]
        greedy_hit = greedy_pred[0] == t.target_fact_id

        model_pred = model_rerank(clf, t, generator, True)
        model_hit = model_pred[0] == t.target_fact_id

        if not greedy_hit and model_hit:
            # Extract feature values for analysis
            candidates, violation_scores = generator.generate_candidates(t, top_k=5)
            features = build_features_with_groups(t, candidates, violation_scores, True)

            # Find greedy's top choice and target
            greedy_top_idx = 0
            target_idx = None
            for i, fid in enumerate(candidates):
                if fid == t.target_fact_id:
                    target_idx = i

            case = {
                "error_type": t.error_info["error_type"],
                "target": t.target_fact_id,
                "greedy_top": greedy_pred[0],
                "model_top": model_pred[0],
                "target_value": t.error_info["error_value"],
                "original_value": t.error_info["original_value"],
                "violations_count": len(t.violations),
                "candidates_count": len(candidates),
                # Feature comparison
                "greedy_top_features": features[greedy_top_idx].tolist()[:5],  # top 5 features
                "target_features": features[target_idx].tolist()[:5] if target_idx else []
            }
            cases.append(case)

    return cases[:5]  # Return top 5


def run_iteration_11():
    """运行第11轮实验 - 最终投稿材料"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 11 (Final)")
    print("Submission Package: Corrected Stats + Ablation + Cases")
    print("=" * 60)

    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    error_dist = defaultdict(int)
    for t in tasks:
        error_dist[t.error_info["error_type"]] += 1
    print(f"Error types: {dict(error_dist)}")

    # ============================================================
    # Experiment 1: Corrected Repeated CV Statistics
    # ============================================================
    print("\n[Experiment 1] Corrected Repeated CV Statistics...")

    cv_stats = correct_repeated_cv(tasks, seeds=[42, 123, 456])

    print(f"\nCorrected Statistics (per-run Top-1 mean ± std):")
    print(f"  Greedy: {cv_stats['greedy']['mean']:.2%} ± {cv_stats['greedy']['std']:.2%}")
    print(f"  XGBoost: {cv_stats['xgb']['mean']:.2%} ± {cv_stats['xgb']['std']:.2%}")

    print(f"\nPer-run Top-1 values:")
    for i, (g, x) in enumerate(zip(cv_stats['greedy']['runs'], cv_stats['xgb']['runs'])):
        print(f"  Run {i+1}: Greedy={g:.2%}, XGB={x:.2%}")

    # ============================================================
    # Experiment 2: value_shift Ablation
    # ============================================================
    print("\n[Experiment 2] value_shift Feature Ablation...")

    vs_ablation = value_shift_ablation(tasks, seed=42)

    if vs_ablation:
        print(f"\n  Greedy baseline: {vs_ablation['greedy']:.2%}")
        print(f"  Full (with VS features): {vs_ablation['full_with_vs_features']:.2%}")
        print(f"  Without VS features: {vs_ablation['without_vs_features']:.2%}")
        print(f"\n  Impact of VS features: {vs_ablation['full_with_vs_features'] - vs_ablation['without_vs_features']:.2%}")

    # ============================================================
    # Experiment 3: Case Studies
    # ============================================================
    print("\n[Experiment 3] Case Studies (Greedy fails, Model succeeds)...")

    cases = case_studies(tasks, seed=42)

    print(f"\nFound {len(cases)} cases")

    for i, case in enumerate(cases):
        print(f"\n  Case {i+1}: {case['error_type']}")
        print(f"    Target: {case['target']} (value={case['target_value']:.2f})")
        print(f"    Greedy top: {case['greedy_top']}")
        print(f"    Model top: {case['model_top']} ✓")
        print(f"    Violations: {case['violations_count']}, Candidates: {case['candidates_count']}")

    # ============================================================
    # Save Final Results
    # ============================================================
    results = {
        "iteration": 11,
        "final": True,
        "n_tasks": len(tasks),
        "corrected_cv_stats": {
            "greedy_mean": cv_stats['greedy']['mean'],
            "greedy_std": cv_stats['greedy']['std'],
            "xgb_mean": cv_stats['xgb']['mean'],
            "xgb_std": cv_stats['xgb']['std']
        },
        "value_shift_ablation": vs_ablation,
        "case_studies": cases,
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_11_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_11_results.json")

    # ============================================================
    # Paper-Ready Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)

    print("\n1. Main Results Table:")
    print("| Method | Top-1 (mean ± std) |")
    print("|--------|-------------------|")
    print(f"| Greedy | {cv_stats['greedy']['mean']:.2%} ± {cv_stats['greedy']['std']:.2%} |")
    print(f"| XGBoost | {cv_stats['xgb']['mean']:.2%} ± {cv_stats['xgb']['std']:.2%} |")

    print("\n2. value_shift Ablation:")
    if vs_ablation:
        print("| Config | Top-1 |")
        print("|--------|-------|")
        print(f"| Greedy | {vs_ablation['greedy']:.2%} |")
        print(f"| Without VS features | {vs_ablation['without_vs_features']:.2%} |")
        print(f"| With VS features | {vs_ablation['full_with_vs_features']:.2%} |")

    print("\n3. Core Conclusion:")
    print("  Constraint violations provide candidate recall;")
    print("  numeric discrepancy features provide root-cause identification.")

    return results


if __name__ == "__main__":
    run_iteration_11()
    print("\n" + "=" * 60)
    print("Iteration 11 Complete - Final Submission Package")
    print("=" * 60)
"""
XBRL Minimal Repair - Iteration 10
Fix value_shift: Add specific features + Repeated CV verification
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


def build_features_with_value_shift_specific(task, candidate_ids, violation_scores):
    """构建包含value_shift专属特征的特征集"""
    features = []
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    # 计算全局统计
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
            features.append([0] * 24)  # 扩展到24维
            continue

        feat = []

        # === 原有特征 (18维) ===

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

        # === 新增 value_shift 专属特征 (6维) ===

        # 19. 绝对偏差特征（value_shift关键）
        # 计算如果这个值是根因，它偏了多少
        absolute_shift = 0.0
        for v in task.violations:
            if fid in v["involved_facts"]:
                # 对于component位置，计算绝对偏差
                for j, _ in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        # 理论上这个值应该是多少
                        ideal = v["expected_sum"] - sum(task._get_cell_value(v["row"], j2) or 0
                                                        for j2, _ in v.get("components", [])
                                                        if j2 != j)
                        absolute_shift = abs(fact["value"] - ideal)
                        break

                # 对于total位置
                if abs(fact["value"] - v["reported_total"]) < 0.1:
                    absolute_shift = abs(v["reported_total"] - v["expected_sum"])

        feat.append(absolute_shift / max_value)

        # 20. 同行相对位置（中位数偏差）
        row_values = [abs(f["value"]) for f in task.facts if f["row"] == fact["row"]]
        if row_values:
            row_median = np.median(row_values)
            feat.append(abs(abs(fact["value"]) - row_median) / max_value)
        else:
            feat.append(0.0)

        # 21. 同列相对位置（中位数偏差）
        col_values = [abs(f["value"]) for f in task.facts if f["col"] == fact["col"]]
        if col_values:
            col_median = np.median(col_values)
            feat.append(abs(abs(fact["value"]) - col_median) / max_value)
        else:
            feat.append(0.0)

        # 22. 周期/上下文一致性特征
        # 计算与周边单元格的signed deviation
        neighbors = []
        for f in task.facts:
            if abs(f["row"] - fact["row"]) <= 1 and abs(f["col"] - fact["col"]) <= 1 and f["id"] != fid:
                neighbors.append(f["value"])

        if neighbors:
            neighbor_mean = np.mean(neighbors)
            feat.append(abs(fact["value"] - neighbor_mean) / max_value)
        else:
            feat.append(0.0)

        # 23. 符号一致性特征
        # 与同行/同列其他值的符号一致率
        same_row_facts = [f for f in task.facts if f["row"] == fact["row"] and f["id"] != fid]
        if same_row_facts:
            sign_match = sum(1 for f in same_row_facts if (f["value"] < 0) == (fact["value"] < 0))
            feat.append(sign_match / len(same_row_facts))
        else:
            feat.append(0.0)

        # 24. 约束修复潜力（value_shift专用）
        # 计算修改这个值能消除多少violation
        repair_potential = 0.0
        for v in task.violations:
            if fid in v["involved_facts"]:
                repair_potential += v["residual"]

        feat.append(repair_potential / (max_residual * len(task.violations)) if task.violations else 0.0)

        features.append(feat)

    return np.array(features)


def train_model(model_type, X_train, y_train):
    if model_type == "xgboost":
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_type == "logistic":
        clf = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    clf.fit(X_train, y_train)
    return clf


def model_rerank(clf, task, generator):
    candidates, violation_scores = generator.generate_candidates(task, top_k=5)

    if len(candidates) == 0:
        return [f["id"] for f in task.facts]

    features = build_features_with_value_shift_specific(task, candidates, violation_scores)

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


def repeated_cv_experiment(tasks, n_splits=5, n_repeats=3, seeds=[42, 123, 456]):
    """多次重复CV验证"""
    generator = ViolationGreedyCandidateGenerator()

    all_results = {
        "greedy": {"top1": [], "top3": [], "mrr": []},
        "xgboost": {"top1": [], "top3": [], "mrr": []},
        "decision_tree": {"top1": [], "top3": [], "mrr": []}
    }

    for repeat_idx, seed in enumerate(seeds):
        set_seed(seed)

        indices = list(range(len(tasks)))
        random.shuffle(indices)

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
                features = build_features_with_value_shift_specific(task, candidates, violation_scores)

                for i, fid in enumerate(candidates):
                    X_train.append(features[i])
                    y_train.append(1 if fid == task.target_fact_id else 0)

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            if len(np.unique(y_train)) < 2:
                continue

            # Train models
            xgb_clf = train_model("xgboost", X_train, y_train)
            dt_clf = train_model("decision_tree", X_train, y_train)

            # Evaluate
            for t in test_tasks:
                # Greedy
                greedy_pred = generator.generate_candidates(t, top_k=5)[0]
                greedy_res = t.evaluate(greedy_pred)

                # XGBoost
                xgb_pred = model_rerank(xgb_clf, t, generator)
                xgb_res = t.evaluate(xgb_pred)

                # Decision Tree
                dt_pred = model_rerank(dt_clf, t, generator)
                dt_res = t.evaluate(dt_pred)

                all_results["greedy"]["top1"].append(greedy_res["hit_1"])
                all_results["greedy"]["top3"].append(greedy_res["hit_3"])
                all_results["greedy"]["mrr"].append(greedy_res["mrr"])

                all_results["xgboost"]["top1"].append(xgb_res["hit_1"])
                all_results["xgboost"]["top3"].append(xgb_res["hit_3"])
                all_results["xgboost"]["mrr"].append(xgb_res["mrr"])

                all_results["decision_tree"]["top1"].append(dt_res["hit_1"])
                all_results["decision_tree"]["top3"].append(dt_res["hit_3"])
                all_results["decision_tree"]["mrr"].append(dt_res["mrr"])

    # Compute statistics
    stats = {}
    for model in all_results:
        stats[model] = {
            "top1_mean": np.mean(all_results[model]["top1"]),
            "top1_std": np.std(all_results[model]["top1"]),
            "top3_mean": np.mean(all_results[model]["top3"]),
            "mrr_mean": np.mean(all_results[model]["mrr"])
        }

    return stats


def run_iteration_10():
    """运行第10轮实验 - 修复value_shift"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 10")
    print("Fix value_shift: Specific Features + Repeated CV")
    print("=" * 60)

    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    error_dist = defaultdict(int)
    for t in tasks:
        error_dist[t.error_info["error_type"]] += 1
    print(f"Error types: {dict(error_dist)}")

    # ============================================================
    # Experiment 1: Repeated CV (3 seeds × 5 folds = 15 experiments)
    # ============================================================
    print("\n[Experiment 1] Repeated CV (3 seeds × 5 folds)...")

    cv_stats = repeated_cv_experiment(tasks, n_splits=5, n_repeats=3, seeds=[42, 123, 456])

    print("\nRepeated CV Results:")
    for model, stats in cv_stats.items():
        print(f"  {model}: Top-1={stats['top1_mean']:.2%}±{stats['top1_std']:.2%}, Top-3={stats['top3_mean']:.2%}, MRR={stats['mrr_mean']:.4f}")

    # ============================================================
    # Experiment 2: Per Error Type Analysis
    # ============================================================
    print("\n[Experiment 2] Per Error Type Analysis with New Features...")

    set_seed(42)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    generator = ViolationGreedyCandidateGenerator()

    # Train with new features
    X_train = []
    y_train = []

    for task in train:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_features_with_value_shift_specific(task, candidates, violation_scores)

        for i, fid in enumerate(candidates):
            X_train.append(features[i])
            y_train.append(1 if fid == task.target_fact_id else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    xgb_clf = train_model("xgboost", X_train, y_train)
    dt_clf = train_model("decision_tree", X_train, y_train)

    # Evaluate per error type
    per_type_results = {}

    for error_type in ["sign_flip", "scale_10", "value_shift"]:
        type_test = [t for t in test if t.error_info["error_type"] == error_type]
        if len(type_test) == 0:
            continue

        type_results = {"greedy": [], "xgb": [], "dt": []}

        for t in type_test:
            greedy_pred = generator.generate_candidates(t, top_k=5)[0]
            type_results["greedy"].append(t.evaluate(greedy_pred))

            xgb_pred = model_rerank(xgb_clf, t, generator)
            type_results["xgb"].append(t.evaluate(xgb_pred))

            dt_pred = model_rerank(dt_clf, t, generator)
            type_results["dt"].append(t.evaluate(dt_pred))

        per_type_results[error_type] = {
            "n_samples": len(type_test),
            "greedy_top1": sum(r["hit_1"] for r in type_results["greedy"]) / len(type_results["greedy"]),
            "xgb_top1": sum(r["hit_1"] for r in type_results["xgb"]) / len(type_results["xgb"]),
            "dt_top1": sum(r["hit_1"] for r in type_results["dt"]) / len(type_results["dt"]),
        }

        print(f"\n  {error_type} ({len(type_test)} samples):")
        print(f"    Greedy: {per_type_results[error_type]['greedy_top1']:.2%}")
        print(f"    XGB: {per_type_results[error_type]['xgb_top1']:.2%}")
        print(f"    DT: {per_type_results[error_type]['dt_top1']:.2%}")

    # ============================================================
    # Experiment 3: Feature Importance Analysis
    # ============================================================
    print("\n[Experiment 3] Feature Importance with value_shift features...")

    feature_names = [
        "violation_score", "violation_count", "residual_contrib", "residual_ratio",
        "is_total", "is_component",
        "value_norm", "abs_value_norm", "is_negative",
        "log_scale", "expected_diff",
        "row_pos", "col_pos",
        "discrepancy", "residual_ratio_val", "row_context", "z_score", "bias",
        # New features for value_shift
        "absolute_shift", "row_median_dev", "col_median_dev",
        "neighbor_dev", "sign_consistency", "repair_potential"
    ]

    importances = xgb_clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

    print("  Top 10 features:")
    for name, imp in top_features:
        print(f"    {name}: {imp:.4f}")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "iteration": 10,
        "n_tasks": len(tasks),
        "repeated_cv": cv_stats,
        "per_error_type": per_type_results,
        "feature_importance": dict(zip(feature_names, importances.tolist())),
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_10_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_10_results.json")

    # ============================================================
    # Paper-Ready Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Paper-Ready Summary")
    print("=" * 60)

    print("\nRepeated CV (15 experiments):")
    print("| Method | Top-1 | Std |")
    print("|--------|-------|-----|")
    for model in ["greedy", "xgboost", "decision_tree"]:
        print(f"| {model} | {cv_stats[model]['top1_mean']:.2%} | {cv_stats[model]['top1_std']:.2%} |")

    return results


if __name__ == "__main__":
    run_iteration_10()
    print("\n" + "=" * 60)
    print("Iteration 10 Complete - value_shift Features Added")
    print("=" * 60)
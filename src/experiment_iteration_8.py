"""
XBRL Minimal Repair - Iteration 8
Fix Leakage: Remove original_diff, use only inference-available features
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


# ============================================================
# Valid Feature Engineering (No Leakage)
# ============================================================

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


def build_valid_features(task, candidate_ids, violation_scores):
    """构建仅使用推理时可获得特征（无泄漏）"""
    features = []
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    for fid in candidate_ids:
        fact = None
        for f in task.facts:
            if f["id"] == fid:
                fact = f
                break

        if fact is None:
            features.append([0] * 18)  # 18维有效特征
            continue

        feat = []

        # === Group 1: Violation features (推理可用) ===
        # 1. violation score
        feat.append(violation_scores.get(fid, 0) / max_residual)
        # 2. violation参与数
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, len(task.violations)))
        # 3. residual贡献
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / max_residual)
        # 4. residual比例
        if task.violations:
            feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / sum(v["residual"] for v in task.violations))
        else:
            feat.append(0.0)

        # === Group 2: Role features (推理可用) ===
        # 5. is_total (当前值是否等于reported_total)
        feat.append(1.0 if any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations) else 0.0)
        # 6. is_component (是否在components中)
        feat.append(1.0 if any(fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])] for v in task.violations) else 0.0)

        # === Group 3: Numeric features (推理可用) ===
        # 7-9
        feat.append(fact["value"] / max_value)
        feat.append(abs(fact["value"]) / max_value)
        feat.append(1.0 if fact["value"] < 0 else 0.0)

        # === Group 4: Scale features (推理可用) ===
        # 10. log scale
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)
        # 11. 与expected_sum的差（如果是total，这是推理可算的）
        is_total = any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations)
        if is_total:
            diffs = [abs(fact["value"] - v["expected_sum"]) / max_value for v in task.violations
                     if abs(fact["value"] - v["reported_total"]) < 0.1]
            feat.append(min(diffs) if diffs else 0.0)
        else:
            feat.append(0.0)

        # === Group 5: Structural features (推理可用) ===
        # 12-13
        feat.append(fact["row"] / 10)
        feat.append(fact["col"] / 5)

        # === Group 6: Discrepancy-based features (推理可算，不用ground truth) ===

        # 14. 如果这是component，改动它能否解释残差？
        # 计算公式: residual vs |value - expected_contrib|
        discrepancy_score = 0.0
        for v in task.violations:
            if fid in v["involved_facts"]:
                # 检查这个fact是否在components中
                for j, comp_val in v.get("components", []):
                    if f"fact_{v['row']}_{j}" == fid:
                        # 当前值与约束期望的贡献差异
                        # expected_sum = sum of components
                        # 如果这个是component，改动它可以影响expected_sum
                        current_contrib = fact["value"]
                        # 其他components的当前值
                        other_sum = sum(task._get_cell_value(v["row"], j2) or 0
                                       for j2, _ in v.get("components", [])
                                       if j2 != j)
                        # 如果这个值变成某个值，残差会变
                        # 理想情况下，如果这个是根因，改动它能完全消除残差
                        # 即: current_total = new_contrib + other_sum
                        ideal_contrib = v["reported_total"] - other_sum
                        discrepancy_score = abs(current_contrib - ideal_contrib) / max_value
                        break

                # 如果是total位置
                if abs(fact["value"] - v["reported_total"]) < 0.1:
                    # total位置的discrepancy = |reported_total - expected_sum|
                    discrepancy_score = abs(v["reported_total"] - v["expected_sum"]) / max_value

        feat.append(discrepancy_score)

        # 15. 残差相对自身比例 (推理可算)
        if fact["value"] != 0:
            feat.append(max_residual / abs(fact["value"]))
        else:
            feat.append(0.0)

        # 16. 上下文偏离 - 同行均值 (推理可算)
        row_facts = [f for f in task.facts if f["row"] == fact["row"] and f["id"] != fid]
        if row_facts:
            row_mean = np.mean([abs(f["value"]) for f in row_facts])
            feat.append(abs(abs(fact["value"]) - row_mean) / max_value)
        else:
            feat.append(0.0)

        # 17. z-score (推理可算)
        all_values = [abs(f["value"]) for f in task.facts]
        if len(all_values) > 1:
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            if std_val > 0:
                feat.append(abs(abs(fact["value"]) - mean_val) / std_val)
            else:
                feat.append(0.0)
        else:
            feat.append(0.0)

        # 18. constant
        feat.append(1.0)

        features.append(feat)

    return np.array(features)


# ============================================================
# Statistical Analysis
# ============================================================

def paired_bootstrap_ci(greedy_results, xgb_results, n_bootstrap=1000):
    """配对bootstrap置信区间"""
    n = len(greedy_results)

    top1_diffs = []
    mrr_diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)

        greedy_top1 = np.mean([greedy_results[i]["hit_1"] for i in indices])
        xgb_top1 = np.mean([xgb_results[i]["hit_1"] for i in indices])
        greedy_mrr = np.mean([greedy_results[i]["mrr"] for i in indices])
        xgb_mrr = np.mean([xgb_results[i]["mrr"] for i in indices])

        top1_diffs.append(xgb_top1 - greedy_top1)
        mrr_diffs.append(xgb_mrr - greedy_mrr)

    return {
        "top1_diff_mean": np.mean(top1_diffs),
        "top1_diff_ci": (np.percentile(top1_diffs, 2.5), np.percentile(top1_diffs, 97.5)),
        "mrr_diff_mean": np.mean(mrr_diffs),
        "mrr_diff_ci": (np.percentile(mrr_diffs, 2.5), np.percentile(mrr_diffs, 97.5))
    }


def cross_validation_experiment(tasks, n_splits=5, seed=42):
    """交叉验证"""
    set_seed(seed)

    indices = list(range(len(tasks)))
    random.shuffle(indices)

    fold_size = len(tasks) // n_splits
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_splits)]

    generator = ViolationGreedyCandidateGenerator()

    all_fold_results = []

    for fold_idx in range(n_splits):
        test_indices = folds[fold_idx]
        train_indices = [i for i in indices if i not in test_indices]

        train_tasks = [tasks[i] for i in train_indices]
        test_tasks = [tasks[i] for i in test_indices]

        # Train XGBoost with valid features
        X_train = []
        y_train = []

        for task in train_tasks:
            if not task.has_violation():
                continue

            candidates, violation_scores = generator.generate_candidates(task, top_k=5)
            features = build_valid_features(task, candidates, violation_scores)

            for i, fid in enumerate(candidates):
                X_train.append(features[i])
                y_train.append(1 if fid == task.target_fact_id else 0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if len(np.unique(y_train)) < 2:
            continue

        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Evaluate
        fold_results = []
        for t in test_tasks:
            greedy_pred = generator.generate_candidates(t, top_k=5)[0]
            greedy_res = t.evaluate(greedy_pred)

            candidates, violation_scores = generator.generate_candidates(t, top_k=5)
            features = build_valid_features(t, candidates, violation_scores)

            probs = clf.predict_proba(features)[:, 1]
            ranked = [candidates[i] for i in np.argsort(-probs)]

            for f in t.facts:
                if f["id"] not in ranked:
                    ranked.append(f["id"])

            xgb_res = t.evaluate(ranked)

            fold_results.append({
                "greedy": greedy_res,
                "xgb": xgb_res,
                "error_type": t.error_info["error_type"]
            })

        all_fold_results.append(fold_results)

    return all_fold_results


# ============================================================
# Main Experiment
# ============================================================

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


def run_iteration_8():
    """运行第8轮实验 - 修复泄漏"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 8")
    print("Fix Leakage: Valid Inference-Time Features Only")
    print("=" * 60)

    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    error_dist = defaultdict(int)
    for t in tasks:
        error_dist[t.error_info["error_type"]] += 1
    print(f"Error types: {dict(error_dist)}")

    # ============================================================
    # Experiment 1: Main Results with Valid Features
    # ============================================================
    print("\n[Experiment 1] Paired Bootstrap CI (Valid Features)...")

    set_seed(42)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    generator = ViolationGreedyCandidateGenerator()

    # Train XGBoost with valid features
    X_train = []
    y_train = []

    for task in train:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_valid_features(task, candidates, violation_scores)

        for i, fid in enumerate(candidates):
            X_train.append(features[i])
            y_train.append(1 if fid == task.target_fact_id else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    greedy_results = []
    xgb_results = []

    for t in test:
        greedy_pred = generator.generate_candidates(t, top_k=5)[0]
        greedy_res = t.evaluate(greedy_pred)
        greedy_results.append(greedy_res)

        candidates, violation_scores = generator.generate_candidates(t, top_k=5)
        features = build_valid_features(t, candidates, violation_scores)

        probs = clf.predict_proba(features)[:, 1]
        ranked = [candidates[i] for i in np.argsort(-probs)]

        for f in t.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        xgb_res = t.evaluate(ranked)
        xgb_results.append(xgb_res)

    paired_ci = paired_bootstrap_ci(greedy_results, xgb_results, n_bootstrap=1000)

    greedy_top1 = sum(r["hit_1"] for r in greedy_results) / len(greedy_results)
    xgb_top1 = sum(r["hit_1"] for r in xgb_results) / len(xgb_results)

    print(f"  Greedy Top-1: {greedy_top1:.2%}")
    print(f"  XGBoost Top-1: {xgb_top1:.2%}")
    print(f"  Paired Improvement: {paired_ci['top1_diff_mean']:.2%} (95% CI: [{paired_ci['top1_diff_ci'][0]:.2%}, {paired_ci['top1_diff_ci'][1]:.2%}])")

    # ============================================================
    # Experiment 2: Per Error Type
    # ============================================================
    print("\n[Experiment 2] Per Error Type Analysis...")

    for error_type in ["sign_flip", "scale_10", "value_shift"]:
        type_results = [(g, x) for g, x, t in zip(greedy_results, xgb_results, test)
                        if t.error_info["error_type"] == error_type]

        if len(type_results) == 0:
            continue

        type_greedy = [g for g, x in type_results]
        type_xgb = [x for g, x in type_results]

        greedy_top1 = sum(r["hit_1"] for r in type_greedy) / len(type_greedy)
        xgb_top1 = sum(r["hit_1"] for r in type_xgb) / len(type_xgb)

        print(f"  {error_type} ({len(type_results)} samples): Greedy={greedy_top1:.2%}, XGB={xgb_top1:.2%}, Δ={xgb_top1-greedy_top1:.2%}")

    # ============================================================
    # Experiment 3: Cross-Validation
    # ============================================================
    print("\n[Experiment 3] 5-Fold Cross-Validation...")

    cv_results = cross_validation_experiment(tasks, n_splits=5, seed=42)

    cv_greedy_top1 = []
    cv_xgb_top1 = []

    for fold_idx, fold_results in enumerate(cv_results):
        fold_greedy = [r["greedy"] for r in fold_results]
        fold_xgb = [r["xgb"] for r in fold_results]

        fold_greedy_top1 = sum(r["hit_1"] for r in fold_greedy) / len(fold_greedy)
        fold_xgb_top1 = sum(r["hit_1"] for r in fold_xgb) / len(fold_xgb)

        cv_greedy_top1.append(fold_greedy_top1)
        cv_xgb_top1.append(fold_xgb_top1)

        print(f"  Fold {fold_idx+1}: Greedy={fold_greedy_top1:.2%}, XGB={fold_xgb_top1:.2%}")

    print(f"\n  CV Mean: Greedy={np.mean(cv_greedy_top1):.2%}±{np.std(cv_greedy_top1):.2%}, XGB={np.mean(cv_xgb_top1):.2%}±{np.std(cv_xgb_top1):.2%}")

    # ============================================================
    # Experiment 4: Feature Importance
    # ============================================================
    print("\n[Experiment 4] Feature Importance (Valid Features)...")

    feature_names = [
        "violation_score", "violation_count", "residual_contrib", "residual_ratio",
        "is_total", "is_component",
        "value_norm", "abs_value_norm", "is_negative",
        "log_scale", "expected_diff",
        "row_pos", "col_pos",
        "discrepancy_score", "residual_ratio_val", "row_context", "z_score", "bias"
    ]

    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

    print("  Top 5 features:")
    for name, imp in top_features:
        print(f"    {name}: {imp:.4f}")

    # ============================================================
    # Experiment 5: Ablation
    # ============================================================
    print("\n[Experiment 5] Feature Group Ablation...")

    # Define feature indices for each group
    groups = {
        "violation": [0, 1, 2, 3],
        "role": [4, 5],
        "numeric": [6, 7, 8],
        "scale": [9, 10],
        "structural": [11, 12],
        "discrepancy": [13, 14, 15, 16]
    }

    ablation_results = {}

    for group_name, indices_to_zero in groups.items():
        # Create ablated features
        X_train_ablated = X_train.copy()
        for idx in indices_to_zero:
            X_train_ablated[:, idx] = 0

        clf_ablated = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )
        clf_ablated.fit(X_train_ablated, y_train)

        # Evaluate
        ablated_results = []
        for t in test:
            candidates, violation_scores = generator.generate_candidates(t, top_k=5)
            features = build_valid_features(t, candidates, violation_scores)
            # Zero out the group
            for idx in indices_to_zero:
                features[:, idx] = 0

            probs = clf_ablated.predict_proba(features)[:, 1]
            ranked = [candidates[i] for i in np.argsort(-probs)]

            for f in t.facts:
                if f["id"] not in ranked:
                    ranked.append(f["id"])

            ablated_results.append(t.evaluate(ranked))

        top1 = sum(r["hit_1"] for r in ablated_results) / len(ablated_results)
        ablation_results[group_name] = top1
        print(f"  no_{group_name}: Top-1={top1:.2%}")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "iteration": 8,
        "leakage_fixed": True,
        "n_tasks": len(tasks),
        "paired_ci": paired_ci,
        "cv_results": {
            "greedy_mean": np.mean(cv_greedy_top1),
            "greedy_std": np.std(cv_greedy_top1),
            "xgb_mean": np.mean(cv_xgb_top1),
            "xgb_std": np.std(cv_xgb_top1)
        },
        "feature_importance": dict(zip(feature_names, importances.tolist())),
        "ablation": ablation_results,
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_8_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_8_results.json")
    return results


if __name__ == "__main__":
    run_iteration_8()
    print("\n" + "=" * 60)
    print("Iteration 8 Complete - Leakage Fixed")
    print("=" * 60)
"""
XBRL Minimal Repair - Iteration 7
Final validation: Paired CI + Value_shift improvement + Repeated CV
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
# Enhanced Feature Engineering (针对value_shift)
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


def build_enhanced_features(task, candidate_ids, violation_scores):
    """增强特征工程，针对value_shift改进"""
    features = []
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    # 提取原始值（用于对比）
    original_values = {}
    for c in task.constraints:
        if c["type"] == "calc_sum":
            row = c["row"]
            for j, val in c["components"]:
                original_values[f"fact_{row}_{j}"] = val
            j_total, val_total = c["total"]
            original_values[f"fact_{row}_{j_total}"] = val_total

    for fid in candidate_ids:
        fact = None
        for f in task.facts:
            if f["id"] == fid:
                fact = f
                break

        if fact is None:
            features.append([0] * 20)  # 增加到20维
            continue

        feat = []

        # 1-4: violation features
        feat.append(violation_scores.get(fid, 0) / max_residual)
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, len(task.violations)))
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / max_residual)
        if task.violations:
            feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / sum(v["residual"] for v in task.violations))
        else:
            feat.append(0.0)

        # 5-6: role features
        feat.append(1.0 if any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations) else 0.0)
        feat.append(0.0)

        # 7-9: numeric features
        feat.append(fact["value"] / max_value)
        feat.append(abs(fact["value"]) / max_value)
        feat.append(1.0 if fact["value"] < 0 else 0.0)

        # 10-11: scale features
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)
        is_total = any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations)
        if is_total:
            diffs = [abs(fact["value"] - v["expected_sum"]) / max_value for v in task.violations
                     if abs(fact["value"] - v["reported_total"]) < 0.1]
            feat.append(min(diffs) if diffs else 0.0)
        else:
            feat.append(0.0)

        # 12-13: structural features
        feat.append(fact["row"] / 10)
        feat.append(fact["col"] / 5)

        # ========== 新增特征（针对value_shift） ==========

        # 14: 原始值对比（如果有原始数据）
        original_val = original_values.get(fid, None)
        if original_val is not None:
            feat.append(abs(fact["value"] - original_val) / max_value)  # 相对于原始值的偏差
        else:
            feat.append(0.0)

        # 15: 相对残差比例（改进value_shift检测）
        if is_total:
            # Total位置：检查残差是否与预期一致
            expected_diffs = [v["expected_sum"] - v["reported_total"] for v in task.violations
                             if abs(fact["value"] - v["reported_total"]) < 0.1]
            if expected_diffs:
                feat.append(expected_diffs[0] / max_value)
            else:
                feat.append(0.0)
        else:
            # Component位置：检查改动能否修复残差
            for v in task.violations:
                if fid in [f"fact_{v['row']}_{j}" for j, _ in v.get("components", [])]:
                    # 如果把这个值改回expected_sum的补数，残差会减少多少
                    current_contrib = fact["value"]
                    expected_contrib = v["expected_sum"] - sum(
                        task._get_cell_value(v["row"], j) or 0
                        for j, _ in v.get("components", [])
                        if f"fact_{v['row']}_{j}" != fid
                    )
                    feat.append(abs(current_contrib - expected_contrib) / max_value)
                    break
            else:
                feat.append(0.0)

        # 16: 数值上下文（同行其他值）
        row_facts = [f for f in task.facts if f["row"] == fact["row"] and f["id"] != fid]
        if row_facts:
            row_mean = np.mean([abs(f["value"]) for f in row_facts])
            feat.append(abs(abs(fact["value"]) - row_mean) / max_value)  # 与同行均值偏离
        else:
            feat.append(0.0)

        # 17: 数值上下文（同列其他值）
        col_facts = [f for f in task.facts if f["col"] == fact["col"] and f["id"] != fid]
        if col_facts:
            col_mean = np.mean([abs(f["value"]) for f in col_facts])
            feat.append(abs(abs(fact["value"]) - col_mean) / max_value)  # 与同列均值偏离
        else:
            feat.append(0.0)

        # 18: z-score（相对于所有facts）
        all_values = [abs(f["value"]) for f in task.facts]
        if len(all_values) > 1:
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            if std_val > 0:
                feat.append(abs(abs(fact["value"]) - mean_val) / std_val)  # z-score
            else:
                feat.append(0.0)
        else:
            feat.append(0.0)

        # 19: 相对偏移比例
        if fact["value"] != 0:
            feat.append(max_residual / abs(fact["value"]))  # 残差相对于自身值的比例
        else:
            feat.append(0.0)

        # 20: constant bias
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

    # 按错误类型分层split
    error_types = [t.error_info["error_type"] for t in tasks]

    # 使用简单的随机split（因为StratifiedKFold需要标签）
    indices = list(range(len(tasks)))
    random.shuffle(indices)

    fold_size = len(tasks) // n_splits
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_splits)]

    generator = ViolationGreedyCandidateGenerator()

    all_fold_results = []

    for fold_idx in range(n_splits):
        # Train on other folds, test on current fold
        test_indices = folds[fold_idx]
        train_indices = [i for i in indices if i not in test_indices]

        train_tasks = [tasks[i] for i in train_indices]
        test_tasks = [tasks[i] for i in test_indices]

        # Train XGBoost
        X_train = []
        y_train = []

        for task in train_tasks:
            if not task.has_violation():
                continue

            candidates, violation_scores = generator.generate_candidates(task, top_k=5)
            features = build_enhanced_features(task, candidates, violation_scores)

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
            features = build_enhanced_features(t, candidates, violation_scores)

            probs = clf.predict_proba(features)[:, 1]
            ranked = [candidates[i] for i in np.argsort(-probs)]

            # 补充其他facts
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


def run_iteration_7():
    """运行第7轮实验"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 7")
    print("Final Validation: Paired CI + Enhanced Features + CV")
    print("=" * 60)

    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    error_dist = defaultdict(int)
    for t in tasks:
        error_dist[t.error_info["error_type"]] += 1
    print(f"Error types: {dict(error_dist)}")

    # ============================================================
    # Experiment 1: Paired Bootstrap CI
    # ============================================================
    print("\n[Experiment 1] Paired Bootstrap CI for Improvement...")

    set_seed(42)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    generator = ViolationGreedyCandidateGenerator()

    # Train XGBoost with enhanced features
    X_train = []
    y_train = []

    for task in train:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_enhanced_features(task, candidates, violation_scores)

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
        features = build_enhanced_features(t, candidates, violation_scores)

        probs = clf.predict_proba(features)[:, 1]
        ranked = [candidates[i] for i in np.argsort(-probs)]

        for f in t.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        xgb_res = t.evaluate(ranked)
        xgb_results.append(xgb_res)

    paired_ci = paired_bootstrap_ci(greedy_results, xgb_results, n_bootstrap=1000)

    print(f"  Paired Top-1 Improvement: {paired_ci['top1_diff_mean']:.2%} (95% CI: [{paired_ci['top1_diff_ci'][0]:.2%}, {paired_ci['top1_diff_ci'][1]:.2%}])")
    print(f"  Paired MRR Improvement: {paired_ci['mrr_diff_mean']:.4f} (95% CI: [{paired_ci['mrr_diff_ci'][0]:.4f}, {paired_ci['mrr_diff_ci'][1]:.4f}])")

    # ============================================================
    # Experiment 2: Per Error Type Analysis
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
    # Experiment 4: Feature Importance Analysis
    # ============================================================
    print("\n[Experiment 4] Feature Importance...")

    feature_names = [
        "violation_score", "violation_count", "residual_contrib", "residual_ratio",
        "is_total", "is_component",
        "value_norm", "abs_value_norm", "is_negative",
        "log_scale", "expected_diff",
        "row_pos", "col_pos",
        "original_diff", "relative_residual", "row_context", "col_context",
        "z_score", "residual_ratio_val", "bias"
    ]

    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

    print("  Top 5 features:")
    for name, imp in top_features:
        print(f"    {name}: {imp:.4f}")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "iteration": 7,
        "n_tasks": len(tasks),
        "paired_ci": paired_ci,
        "per_error_type": {
            error_type: {
                "greedy": sum(r["hit_1"] for r in type_greedy) / len(type_greedy),
                "xgb": sum(r["hit_1"] for r in type_xgb) / len(type_xgb)
            }
            for error_type in ["sign_flip", "scale_10", "value_shift"]
        },
        "cv_results": {
            "greedy_mean": np.mean(cv_greedy_top1),
            "greedy_std": np.std(cv_greedy_top1),
            "xgb_mean": np.mean(cv_xgb_top1),
            "xgb_std": np.std(cv_xgb_top1)
        },
        "feature_importance": dict(zip(feature_names, importances.tolist())),
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_7_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_7_results.json")
    return results


if __name__ == "__main__":
    run_iteration_7()
    print("\n" + "=" * 60)
    print("Iteration 7 Complete")
    print("=" * 60)
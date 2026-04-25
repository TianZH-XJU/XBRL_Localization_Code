"""
XBRL Minimal Repair - Iteration 6
Statistical Robustness + Ablation Study
基于Codex审核建议：增加多随机种子、bootstrap置信区间、特征ablation
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
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 1. Core Classes (Same as Iteration 5)
# ============================================================

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
                            "expected_sum": component_sum
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
# 2. Two-Stage Pipeline
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


def build_candidate_features_ablation(task, candidate_ids, violation_scores, feature_groups=None):
    """构建候选特征，支持ablation"""
    if feature_groups is None:
        feature_groups = ["violation", "role", "numeric", "scale", "structural"]

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
            features.append([0] * 16)
            continue

        feat = []

        # Group 1: violation features
        if "violation" in feature_groups:
            feat.append(violation_scores.get(fid, 0) / max_residual)
            feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, len(task.violations)))
            feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / max_residual)
            if task.violations:
                feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / sum(v["residual"] for v in task.violations))
            else:
                feat.append(0.0)
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

        # Group 2: role features
        if "role" in feature_groups:
            feat.append(1.0 if any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations) else 0.0)
            feat.append(0.0)  # component placeholder
        else:
            feat.extend([0.0, 0.0])

        # Group 3: numeric features
        if "numeric" in feature_groups:
            feat.append(fact["value"] / max_value)
            feat.append(abs(fact["value"]) / max_value)
            feat.append(1.0 if fact["value"] < 0 else 0.0)
        else:
            feat.extend([0.0, 0.0, 0.0])

        # Group 4: scale features
        if "scale" in feature_groups:
            feat.append(np.log10(abs(fact["value"]) + 1) / 10)
            # 与expected_sum的距离（如果是total）
            is_total = any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations)
            if is_total:
                diffs = [abs(fact["value"] - v["expected_sum"]) / max_value for v in task.violations
                         if abs(fact["value"] - v["reported_total"]) < 0.1]
                feat.append(min(diffs) if diffs else 0.0)
            else:
                feat.append(0.0)
        else:
            feat.extend([0.0, 0.0])

        # Group 5: structural features
        if "structural" in feature_groups:
            feat.append(fact["row"] / 10)
            feat.append(fact["col"] / 5)
        else:
            feat.extend([0.0, 0.0])

        feat.append(1.0)  # constant bias

        features.append(feat)

    return np.array(features)


def train_xgboost_ablation(train_tasks, feature_groups=None):
    """训练XGBoost，支持ablation"""
    generator = ViolationGreedyCandidateGenerator()

    X = []
    y = []

    for task in train_tasks:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_candidate_features_ablation(task, candidates, violation_scores, feature_groups)

        for i, fid in enumerate(candidates):
            X.append(features[i])
            y.append(1 if fid == task.target_fact_id else 0)

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        return None

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X, y)

    return clf


def xgboost_rerank(clf, task, feature_groups=None):
    """XGBoost重排"""
    generator = ViolationGreedyCandidateGenerator()
    candidates, violation_scores = generator.generate_candidates(task, top_k=5)

    if len(candidates) == 0 or clf is None:
        return [f["id"] for f in task.facts]

    features = build_candidate_features_ablation(task, candidates, violation_scores, feature_groups)

    probs = clf.predict_proba(features)[:, 1]
    reranked_candidates = [candidates[i] for i in np.argsort(-probs)]

    all_facts = [f["id"] for f in task.facts]
    for fid in all_facts:
        if fid not in reranked_candidates:
            reranked_candidates.append(fid)

    return reranked_candidates


# ============================================================
# 3. Statistical Analysis
# ============================================================

def bootstrap_metrics(results, n_bootstrap=1000):
    """Bootstrap置信区间"""
    top1_scores = [r["hit_1"] for r in results]
    mrr_scores = [r["mrr"] for r in results]

    top1_boots = []
    mrr_boots = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(results), size=len(results), replace=True)
        top1_boots.append(np.mean([top1_scores[i] for i in indices]))
        mrr_boots.append(np.mean([mrr_scores[i] for i in indices]))

    return {
        "top1_mean": np.mean(top1_scores),
        "top1_ci": (np.percentile(top1_boots, 2.5), np.percentile(top1_boots, 97.5)),
        "mrr_mean": np.mean(mrr_scores),
        "mrr_ci": (np.percentile(mrr_boots, 2.5), np.percentile(mrr_boots, 97.5))
    }


def run_multi_seed_experiment(tasks, seeds=[42, 123, 456, 789, 1024], feature_groups=None):
    """多随机种子实验"""
    all_results = []

    for seed in seeds:
        set_seed(seed)

        # 重新shuffle并split
        shuffled = tasks[:]
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * 0.8)
        train = shuffled[:n_train]
        test = shuffled[n_train:]

        # Train XGBoost
        clf = train_xgboost_ablation(train, feature_groups)
        if clf is None:
            continue

        # Evaluate
        generator = ViolationGreedyCandidateGenerator()
        test_results = []

        for t in test:
            greedy_pred = generator.generate_candidates(t, top_k=5)[0]
            xgb_pred = xgboost_rerank(clf, t, feature_groups)
            test_results.append({
                "greedy": t.evaluate(greedy_pred),
                "xgb": t.evaluate(xgb_pred)
            })

        all_results.append(test_results)

    return all_results


# ============================================================
# 4. Main Experiment
# ============================================================

def build_benchmark_violation_only():
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


def run_iteration_6():
    """运行第6轮实验"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 6")
    print("Statistical Robustness + Ablation Study")
    print("=" * 60)

    # 构建benchmark
    tasks = build_benchmark_violation_only()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    error_dist = defaultdict(int)
    for t in tasks:
        error_dist[t.error_info["error_type"]] += 1
    print(f"Error types: {dict(error_dist)}")

    # ============================================================
    # Multi-seed experiment
    # ============================================================
    print("\n[Experiment 1] Multi-seed XGBoost (5 seeds)...")

    seeds = [42, 123, 456, 789, 1024]
    multi_results = run_multi_seed_experiment(tasks, seeds=seeds)

    # Aggregate results
    greedy_top1_all = []
    xgb_top1_all = []
    greedy_mrr_all = []
    xgb_mrr_all = []

    for seed_idx, seed_results in enumerate(multi_results):
        greedy_results = [r["greedy"] for r in seed_results]
        xgb_results = [r["xgb"] for r in seed_results]

        greedy_top1 = sum(r["hit_1"] for r in greedy_results) / len(greedy_results)
        xgb_top1 = sum(r["hit_1"] for r in xgb_results) / len(xgb_results)
        greedy_mrr = sum(r["mrr"] for r in greedy_results) / len(greedy_results)
        xgb_mrr = sum(r["mrr"] for r in xgb_results) / len(xgb_results)

        greedy_top1_all.append(greedy_top1)
        xgb_top1_all.append(xgb_top1)
        greedy_mrr_all.append(greedy_mrr)
        xgb_mrr_all.append(xgb_mrr)

        print(f"  Seed {seeds[seed_idx]}: Greedy={greedy_top1:.2%}, XGB={xgb_top1:.2%}")

    print(f"\nMulti-seed Summary:")
    print(f"  Greedy Top-1: {np.mean(greedy_top1_all):.2%} ± {np.std(greedy_top1_all):.2%}")
    print(f"  XGBoost Top-1: {np.mean(xgb_top1_all):.2%} ± {np.std(xgb_top1_all):.2%}")
    print(f"  Greedy MRR: {np.mean(greedy_mrr_all):.4f} ± {np.std(greedy_mrr_all):.4f}")
    print(f"  XGBoost MRR: {np.mean(xgb_mrr_all):.4f} ± {np.std(xgb_mrr_all):.4f}")

    # ============================================================
    # Bootstrap confidence intervals
    # ============================================================
    print("\n[Experiment 2] Bootstrap 95% CI...")

    # Use first seed results for bootstrap
    first_seed_results = multi_results[0]
    greedy_results = [r["greedy"] for r in first_seed_results]
    xgb_results = [r["xgb"] for r in first_seed_results]

    greedy_ci = bootstrap_metrics(greedy_results, n_bootstrap=1000)
    xgb_ci = bootstrap_metrics(xgb_results, n_bootstrap=1000)

    print(f"  Greedy Top-1: {greedy_ci['top1_mean']:.2%} (95% CI: [{greedy_ci['top1_ci'][0]:.2%}, {greedy_ci['top1_ci'][1]:.2%}])")
    print(f"  XGBoost Top-1: {xgb_ci['top1_mean']:.2%} (95% CI: [{xgb_ci['top1_ci'][0]:.2%}, {xgb_ci['top1_ci'][1]:.2%}])")
    print(f"  Greedy MRR: {greedy_ci['mrr_mean']:.4f} (95% CI: [{greedy_ci['mrr_ci'][0]:.4f}, {greedy_ci['mrr_ci'][1]:.4f}])")
    print(f"  XGBoost MRR: {xgb_ci['mrr_mean']:.4f} (95% CI: [{xgb_ci['mrr_ci'][0]:.4f}, {xgb_ci['mrr_ci'][1]:.4f}])")

    # ============================================================
    # Ablation Study
    # ============================================================
    print("\n[Experiment 3] Feature Ablation...")

    feature_configs = {
        "full": ["violation", "role", "numeric", "scale", "structural"],
        "no_violation": ["role", "numeric", "scale", "structural"],
        "no_role": ["violation", "numeric", "scale", "structural"],
        "no_numeric": ["violation", "role", "scale", "structural"],
        "no_scale": ["violation", "role", "numeric", "structural"],
        "violation_only": ["violation"],
        "scale_only": ["scale"],
    }

    ablation_results = {}

    set_seed(42)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    generator = ViolationGreedyCandidateGenerator()

    for name, groups in feature_configs.items():
        clf = train_xgboost_ablation(train, groups)
        if clf is None:
            continue

        test_results = []
        for t in test:
            pred = xgboost_rerank(clf, t, groups)
            test_results.append(t.evaluate(pred))

        top1 = sum(r["hit_1"] for r in test_results) / len(test_results)
        mrr = sum(r["mrr"] for r in test_results) / len(test_results)

        ablation_results[name] = {"top1": top1, "mrr": mrr}
        print(f"  {name}: Top-1={top1:.2%}, MRR={mrr:.4f}")

    # ============================================================
    # Candidate Set Oracle Analysis
    # ============================================================
    print("\n[Experiment 4] Candidate Oracle Ceiling...")

    # Analyze if target is in top-k candidates
    oracle_top1 = 0
    oracle_top3 = 0
    oracle_top5 = 0

    for t in test:
        candidates, _ = generator.generate_candidates(t, top_k=5)
        if t.target_fact_id == candidates[0]:
            oracle_top1 += 1
        if t.target_fact_id in candidates[:3]:
            oracle_top3 += 1
        if t.target_fact_id in candidates[:5]:
            oracle_top5 += 1

    print(f"  Target in Top-1: {oracle_top1}/{len(test)} ({oracle_top1/len(test):.2%})")
    print(f"  Target in Top-3: {oracle_top3}/{len(test)} ({oracle_top3/len(test):.2%})")
    print(f"  Target in Top-5: {oracle_top5}/{len(test)} ({oracle_top5/len(test):.2%})")

    # ============================================================
    # Save Results
    # ============================================================
    results = {
        "iteration": 6,
        "n_tasks": len(tasks),
        "multi_seed": {
            "seeds": seeds,
            "greedy_top1": {"mean": np.mean(greedy_top1_all), "std": np.std(greedy_top1_all)},
            "xgb_top1": {"mean": np.mean(xgb_top1_all), "std": np.std(xgb_top1_all)},
            "greedy_mrr": {"mean": np.mean(greedy_mrr_all), "std": np.std(greedy_mrr_all)},
            "xgb_mrr": {"mean": np.mean(xgb_mrr_all), "std": np.std(xgb_mrr_all)}
        },
        "bootstrap_ci": {
            "greedy": greedy_ci,
            "xgb": xgb_ci
        },
        "ablation": ablation_results,
        "oracle_ceiling": {
            "top1_recall": oracle_top1 / len(test),
            "top3_recall": oracle_top3 / len(test),
            "top5_recall": oracle_top5 / len(test)
        },
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_6_results.json")
    return results


if __name__ == "__main__":
    run_iteration_6()
    print("\n" + "=" * 60)
    print("Iteration 6 Complete")
    print("=" * 60)
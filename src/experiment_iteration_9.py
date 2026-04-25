"""
XBRL Minimal Repair - Iteration 9
Submission Package: Baselines + Mechanism Analysis + Case Studies
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
            features.append([0] * 18)
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

        feat.append(1.0)

        features.append(feat)

    return np.array(features)


def train_model(model_type, X_train, y_train):
    """训练不同类型的模型"""
    if model_type == "xgboost":
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
    elif model_type == "logistic":
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    clf.fit(X_train, y_train)
    return clf


def model_rerank(clf, task, generator):
    """使用模型重排"""
    candidates, violation_scores = generator.generate_candidates(task, top_k=5)

    if len(candidates) == 0:
        return [f["id"] for f in task.facts]

    features = build_valid_features(task, candidates, violation_scores)

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


def run_iteration_9():
    """运行第9轮实验 - 补充基线和材料"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 9")
    print("Submission Package: Baselines + Analysis + Case Studies")
    print("=" * 60)

    tasks = build_benchmark()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    error_dist = defaultdict(int)
    for t in tasks:
        error_dist[t.error_info["error_type"]] += 1
    print(f"Error types: {dict(error_dist)}")

    # ============================================================
    # Split data
    # ============================================================
    set_seed(42)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    train = shuffled[:n_train]
    test = shuffled[n_train:]

    generator = ViolationGreedyCandidateGenerator()

    # Build training data
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

    print(f"Training samples: {len(X_train)}")

    # ============================================================
    # Experiment 1: Baseline Comparison
    # ============================================================
    print("\n[Experiment 1] Baseline Comparison...")

    models = ["xgboost", "random_forest", "logistic", "decision_tree"]
    results_comparison = {}

    # Greedy baseline
    greedy_results = []
    for t in test:
        greedy_pred = generator.generate_candidates(t, top_k=5)[0]
        greedy_results.append(t.evaluate(greedy_pred))

    greedy_top1 = sum(r["hit_1"] for r in greedy_results) / len(greedy_results)
    results_comparison["greedy"] = {"top1": greedy_top1}
    print(f"  Greedy: Top-1={greedy_top1:.2%}")

    # Train and evaluate each model
    for model_type in models:
        clf = train_model(model_type, X_train, y_train)

        model_results = []
        for t in test:
            pred = model_rerank(clf, t, generator)
            model_results.append(t.evaluate(pred))

        top1 = sum(r["hit_1"] for r in model_results) / len(model_results)
        results_comparison[model_type] = {"top1": top1}
        print(f"  {model_type}: Top-1={top1:.2%}")

    # ============================================================
    # Experiment 2: Feature Group Analysis (for paper table)
    # ============================================================
    print("\n[Experiment 2] Feature Group Analysis...")

    feature_groups = {
        "violation": [0, 1, 2, 3],
        "role": [4, 5],
        "numeric": [6, 7, 8],
        "scale": [9, 10],
        "structural": [11, 12],
        "discrepancy": [13, 14, 15, 16]
    }

    feature_ablation = {}

    # Full model
    clf_full = train_model("xgboost", X_train, y_train)
    full_results = []
    for t in test:
        pred = model_rerank(clf_full, t, generator)
        full_results.append(t.evaluate(pred))
    full_top1 = sum(r["hit_1"] for r in full_results) / len(full_results)
    feature_ablation["full"] = full_top1
    print(f"  full: {full_top1:.2%}")

    # Each group removed
    for group_name, indices in feature_groups.items():
        X_ablated = X_train.copy()
        for idx in indices:
            X_ablated[:, idx] = 0

        clf_ablated = train_model("xgboost", X_ablated, y_train)

        ablated_results = []
        for t in test:
            candidates, violation_scores = generator.generate_candidates(t, top_k=5)
            features = build_valid_features(t, candidates, violation_scores)
            for idx in indices:
                features[:, idx] = 0

            probs = clf_ablated.predict_proba(features)[:, 1]
            ranked = [candidates[i] for i in np.argsort(-probs)]

            for f in t.facts:
                if f["id"] not in ranked:
                    ranked.append(f["id"])

            ablated_results.append(t.evaluate(ranked))

        top1 = sum(r["hit_1"] for r in ablated_results) / len(ablated_results)
        feature_ablation[f"no_{group_name}"] = top1
        print(f"  no_{group_name}: {top1:.2%}")

    # ============================================================
    # Experiment 3: Case Studies
    # ============================================================
    print("\n[Experiment 3] Case Studies (Greedy fails, XGBoost succeeds)...")

    case_studies = []

    for t in test:
        greedy_pred = generator.generate_candidates(t, top_k=5)[0]
        greedy_hit = greedy_pred[0] == t.target_fact_id

        xgb_pred = model_rerank(clf_full, t, generator)
        xgb_hit = xgb_pred[0] == t.target_fact_id

        # Find cases where greedy fails but xgb succeeds
        if not greedy_hit and xgb_hit:
            case = {
                "error_type": t.error_info["error_type"],
                "target_fact": t.target_fact_id,
                "greedy_top1": greedy_pred[0],
                "xgb_top1": xgb_pred[0],
                "violation_count": len(t.violations),
                "facts_count": len(t.facts),
                "original_value": t.error_info["original_value"],
                "error_value": t.error_info["error_value"]
            }
            case_studies.append(case)

    print(f"  Found {len(case_studies)} cases where XGBoost corrects Greedy errors")

    # Show top 3 cases
    for i, case in enumerate(case_studies[:3]):
        print(f"\n  Case {i+1}: {case['error_type']}")
        print(f"    Target: {case['target_fact']}")
        print(f"    Greedy predicted: {case['greedy_top1']}")
        print(f"    XGB predicted: {case['xgb_top1']}")
        print(f"    Original: {case['original_value']}, Error: {case['error_value']}")

    # ============================================================
    # Experiment 4: Mechanism Analysis per Error Type
    # ============================================================
    print("\n[Experiment 4] Mechanism Analysis per Error Type...")

    mechanism_analysis = {}

    for error_type in ["sign_flip", "scale_10", "value_shift"]:
        type_test = [t for t in test if t.error_info["error_type"] == error_type]
        if len(type_test) == 0:
            continue

        # Greedy results
        type_greedy = []
        for t in type_test:
            pred = generator.generate_candidates(t, top_k=5)[0]
            type_greedy.append(t.evaluate(pred))

        # XGBoost results
        type_xgb = []
        for t in type_test:
            pred = model_rerank(clf_full, t, generator)
            type_xgb.append(t.evaluate(pred))

        greedy_top1 = sum(r["hit_1"] for r in type_greedy) / len(type_greedy)
        xgb_top1 = sum(r["hit_1"] for r in type_xgb) / len(type_xgb)

        # Check if target is in candidates
        in_top1 = sum(1 for t in type_test if generator.generate_candidates(t, top_k=5)[0][0] == t.target_fact_id)
        in_top3 = sum(1 for t in type_test if t.target_fact_id in generator.generate_candidates(t, top_k=5)[0][:3])
        in_top5 = sum(1 for t in type_test if t.target_fact_id in generator.generate_candidates(t, top_k=5)[0][:5])

        mechanism_analysis[error_type] = {
            "n_samples": len(type_test),
            "greedy_top1": greedy_top1,
            "xgb_top1": xgb_top1,
            "improvement": xgb_top1 - greedy_top1,
            "oracle_top1_recall": in_top1 / len(type_test),
            "oracle_top3_recall": in_top3 / len(type_test),
            "oracle_top5_recall": in_top5 / len(type_test)
        }

        print(f"\n  {error_type}:")
        print(f"    Greedy Top-1: {greedy_top1:.2%}")
        print(f"    XGB Top-1: {xgb_top1:.2%}")
        print(f"    Improvement: {xgb_top1 - greedy_top1:.2%}")
        print(f"    Oracle Top-1 recall: {in_top1/len(type_test):.2%}")
        print(f"    Oracle Top-3 recall: {in_top3/len(type_test):.2%}")

    # ============================================================
    # Save all results
    # ============================================================
    results = {
        "iteration": 9,
        "n_tasks": len(tasks),
        "baseline_comparison": results_comparison,
        "feature_ablation": feature_ablation,
        "case_studies": case_studies[:5],
        "mechanism_analysis": mechanism_analysis,
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_9_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_9_results.json")

    # ============================================================
    # Generate paper-ready table
    # ============================================================
    print("\n" + "=" * 60)
    print("Paper-Ready Table: Baseline Comparison")
    print("=" * 60)
    print("| Method | Top-1 Accuracy |")
    print("|--------|---------------|")
    for method, res in results_comparison.items():
        print(f"| {method} | {res['top1']:.2%} |")

    print("\n" + "=" * 60)
    print("Paper-Ready Table: Feature Ablation")
    print("=" * 60)
    print("| Configuration | Top-1 Accuracy | Δ from Full |")
    print("|--------------|---------------|-------------|")
    for config, top1 in feature_ablation.items():
        delta = top1 - feature_ablation["full"] if config != "full" else 0
        print(f"| {config} | {top1:.2%} | {delta:.2%} |")

    return results


if __name__ == "__main__":
    run_iteration_9()
    print("\n" + "=" * 60)
    print("Iteration 9 Complete - Submission Package Ready")
    print("=" * 60)
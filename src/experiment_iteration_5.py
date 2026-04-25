"""
XBRL Minimal Repair - Iteration 5
Two-stage: Candidate Generation + Reranking
基于Codex审核建议：GNN不如baseline的核心原因是任务已被violation压缩到小候选集
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 1. Constraint Extraction (Same as Iteration 4)
# ============================================================

class FinancialConstraintExtractor:
    """从原始表格提取约束"""

    def extract_calc_constraints(self, table_data: List[List]) -> List[Dict]:
        """提取加总约束"""
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


# ============================================================
# 2. Task Builder
# ============================================================

class ViolationTask:
    """基于违规的任务"""

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
                    facts.append({
                        "id": f"fact_{i}_{j}",
                        "row": i,
                        "col": j,
                        "value": num
                    })
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
                component_sum = sum(
                    self._get_cell_value(row, j) for j, _ in c["components"]
                    if self._get_cell_value(row, j) is not None
                )
                total_j = c["total"][0]
                current_total = self._get_cell_value(row, total_j)

                if current_total is not None:
                    residual = abs(component_sum - current_total)
                    if residual > max(abs(current_total) * 0.01, 5):
                        involved_ids = []
                        for j, _ in c["components"]:
                            involved_ids.append(f"fact_{row}_{j}")
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
        """评估定位"""
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
    """注入会触发violation的错误"""
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
# 3. Two-Stage Pipeline
# ============================================================

class ViolationGreedyCandidateGenerator:
    """Stage 1: 基于violation生成候选集"""

    def generate_candidates(self, task, top_k=5):
        """生成top-k候选"""
        if not task.violations:
            return [f["id"] for f in task.facts[:top_k]]

        fact_scores = defaultdict(float)
        for v in task.violations:
            for fid in v["involved_facts"]:
                fact_scores[fid] += v["residual"]

        ranked = sorted(fact_scores.keys(), key=lambda x: fact_scores[x], reverse=True)

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        return ranked[:top_k], fact_scores


class CandidateRerankerMLP(nn.Module):
    """Stage 2: MLP候选重排器"""

    def __init__(self, input_dim=16, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.scorer(h).squeeze()


def build_candidate_features(task, candidate_ids, violation_scores):
    """构建候选特征（Codex建议的hand-crafted features）"""
    features = []
    max_residual = max(v["residual"] for v in task.violations) if task.violations else 1
    max_value = max(abs(f["value"]) for f in task.facts) if task.facts else 1

    for fid in candidate_ids:
        # 找到对应的fact
        fact = None
        for f in task.facts:
            if f["id"] == fid:
                fact = f
                break

        if fact is None:
            features.append([0] * 16)
            continue

        # 基础特征
        feat = []
        feat.append(violation_scores.get(fid, 0) / max_residual)  # 1
        feat.append(sum(1 for v in task.violations if fid in v["involved_facts"]) / max(1, len(task.violations)))  # 2
        feat.append(1.0 if any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations) else 0.0)  # 3
        feat.append(0.0)  # 4 (component placeholder)
        feat.append(fact["value"] / max_value)  # 5
        feat.append(abs(fact["value"]) / max_value)  # 6
        feat.append(1.0 if fact["value"] < 0 else 0.0)  # 7
        feat.append(fact["row"] / 10)  # 8
        feat.append(fact["col"] / 5)  # 9
        feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / max_residual)  # 10

        # 11. 与expected_sum的距离（如果是total）
        is_total = any(abs(fact["value"] - v["reported_total"]) < 0.1 for v in task.violations)
        if is_total:
            diffs = [abs(fact["value"] - v["expected_sum"]) / max_value for v in task.violations
                     if abs(fact["value"] - v["reported_total"]) < 0.1]
            feat.append(min(diffs) if diffs else 0.0)
        else:
            feat.append(0.0)

        # 12. 与reported_total的距离（如果是component）
        diffs2 = [abs(fact["value"] - v["reported_total"]) / max_value for v in task.violations
                  if fid in v["involved_facts"] and abs(fact["value"] - v["reported_total"]) > 0.1]
        feat.append(min(diffs2) if diffs2 else 0.0)

        # 13-16
        if task.violations:
            feat.append(sum(v["residual"] for v in task.violations if fid in v["involved_facts"]) / sum(v["residual"] for v in task.violations))
        else:
            feat.append(0.0)
        feat.append(np.log10(abs(fact["value"]) + 1) / 10)  # 14
        feat.append(0.0)  # 15
        feat.append(1.0)  # 16

        features.append(feat)

    return torch.tensor(features, dtype=torch.float)


def train_mlp_reranker(train_tasks, epochs=50, device='cuda', lr=1e-4):
    """训练MLP重排器"""
    generator = ViolationGreedyCandidateGenerator()
    model = CandidateRerankerMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 构建训练数据
    train_data = []
    for task in train_tasks:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_candidate_features(task, candidates, violation_scores)
        if features.size(0) == 0:
            continue

        # 标签：在候选中找出正确答案
        labels = torch.zeros(features.size(0))
        for i, fid in enumerate(candidates):
            if fid == task.target_fact_id:
                labels[i] = 1.0

        train_data.append((features, labels, candidates, task))

    print(f"Training MLP reranker on {len(train_data)} tasks...")

    # 训练
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for features, labels, candidates, task in train_data:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(features)

            # Softmax cross-entropy（候选集单选分类）
            # 更匹配Top-1指标
            probs = F.softmax(scores, dim=0)
            loss = F.cross_entropy(probs.unsqueeze(0), labels.argmax().unsqueeze(0))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def mlp_rerank(model, task, device='cuda'):
    """MLP重排"""
    generator = ViolationGreedyCandidateGenerator()
    candidates, violation_scores = generator.generate_candidates(task, top_k=5)

    if len(candidates) == 0:
        return [f["id"] for f in task.facts]

    features = build_candidate_features(task, candidates, violation_scores)
    features = features.to(device)

    model.eval()
    with torch.no_grad():
        scores = model(features)

    # 重排候选
    scores_np = scores.cpu().numpy()
    reranked_candidates = [candidates[i] for i in np.argsort(-scores_np)]

    # 补充未在候选中的fact
    all_facts = [f["id"] for f in task.facts]
    for fid in all_facts:
        if fid not in reranked_candidates:
            reranked_candidates.append(fid)

    return reranked_candidates


# ============================================================
# 4. XGBoost Reranker (Non-neural baseline)
# ============================================================

def train_xgboost_reranker(train_tasks):
    """训练XGBoost重排器"""
    generator = ViolationGreedyCandidateGenerator()

    X = []
    y = []

    for task in train_tasks:
        if not task.has_violation():
            continue

        candidates, violation_scores = generator.generate_candidates(task, top_k=5)
        features = build_candidate_features(task, candidates, violation_scores)

        for i, fid in enumerate(candidates):
            X.append(features[i].tolist())
            y.append(1 if fid == task.target_fact_id else 0)

    X = np.array(X)
    y = np.array(y)

    print(f"Training XGBoost on {len(X)} candidate samples...")

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    clf.fit(X, y)

    return clf


def xgboost_rerank(clf, task):
    """XGBoost重排"""
    generator = ViolationGreedyCandidateGenerator()
    candidates, violation_scores = generator.generate_candidates(task, top_k=5)

    if len(candidates) == 0:
        return [f["id"] for f in task.facts]

    features = build_candidate_features(task, candidates, violation_scores)

    # 预测概率
    probs = clf.predict_proba(features.numpy())[:, 1]

    # 重排
    reranked_candidates = [candidates[i] for i in np.argsort(-probs)]

    # 补充未在候选中的fact
    all_facts = [f["id"] for f in task.facts]
    for fid in all_facts:
        if fid not in reranked_candidates:
            reranked_candidates.append(fid)

    return reranked_candidates


# ============================================================
# 5. Main Experiment
# ============================================================

def build_benchmark_violation_only():
    """只构建有violation的benchmark"""
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


def run_iteration_5():
    """运行第5轮实验"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 5")
    print("Two-Stage: Candidate Generation + Reranking")
    print("=" * 60)

    # 构建benchmark
    tasks = build_benchmark_violation_only()
    print(f"\nBuilt {len(tasks)} tasks with violations")

    # 统计错误类型
    error_dist = defaultdict(int)
    for t in tasks:
        error_dist[t.error_info["error_type"]] += 1
    print(f"Error types: {dict(error_dist)}")

    # Split
    random.shuffle(tasks)
    n_train = int(len(tasks) * 0.8)
    train = tasks[:n_train]
    test = tasks[n_train:]

    print(f"Train: {len(train)}, Test: {len(test)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============================================================
    # Baseline: Violation Greedy (Stage 1 only)
    # ============================================================
    print("\n[Stage 1] Violation Greedy Baseline...")
    generator = ViolationGreedyCandidateGenerator()

    greedy_res = []
    for t in test:
        candidates, _ = generator.generate_candidates(t, top_k=5)
        greedy_res.append(t.evaluate(candidates))

    def metrics(res):
        return {
            "top1": sum(r["hit_1"] for r in res) / len(res),
            "top3": sum(r["hit_3"] for r in res) / len(res),
            "top5": sum(r["hit_5"] for r in res) / len(res),
            "mrr": sum(r["mrr"] for r in res) / len(res)
        }

    gm = metrics(greedy_res)
    print(f"Violation Greedy (Stage 1 only):")
    print(f"  Top-1: {gm['top1']:.2%}, Top-3: {gm['top3']:.2%}, Top-5: {gm['top5']:.2%}, MRR: {gm['mrr']:.4f}")

    # ============================================================
    # XGBoost Reranker
    # ============================================================
    print("\n[Stage 2a] Training XGBoost Reranker...")
    xgb_model = train_xgboost_reranker(train)

    xgb_res = []
    for t in test:
        xgb_res.append(t.evaluate(xgboost_rerank(xgb_model, t)))

    xm = metrics(xgb_res)
    print(f"XGBoost Reranker:")
    print(f"  Top-1: {xm['top1']:.2%}, Top-3: {xm['top3']:.2%}, Top-5: {xm['top5']:.2%}, MRR: {xm['mrr']:.4f}")

    # ============================================================
    # MLP Reranker
    # ============================================================
    print("\n[Stage 2b] Training MLP Reranker...")
    mlp_model = train_mlp_reranker(train, epochs=50, device=device, lr=1e-4)

    mlp_res = []
    for t in test:
        mlp_res.append(t.evaluate(mlp_rerank(mlp_model, t, device)))

    mm = metrics(mlp_res)
    print(f"MLP Reranker:")
    print(f"  Top-1: {mm['top1']:.2%}, Top-3: {mm['top3']:.2%}, Top-5: {mm['top5']:.2%}, MRR: {mm['mrr']:.4f}")

    # ============================================================
    # 按错误类型分析
    # ============================================================
    print("\n[Analysis] Per Error Type Results:")

    for error_type in ["sign_flip", "scale_10", "value_shift"]:
        type_test = [t for t in test if t.error_info["error_type"] == error_type]
        if len(type_test) == 0:
            continue

        type_greedy = [t.evaluate(generator.generate_candidates(t, top_k=5)[0]) for t in type_test]
        type_xgb = [t.evaluate(xgboost_rerank(xgb_model, t)) for t in type_test]
        type_mlp = [t.evaluate(mlp_rerank(mlp_model, t, device)) for t in type_test]

        print(f"\n{error_type} ({len(type_test)} samples):")
        print(f"  Greedy: Top-1={metrics(type_greedy)['top1']:.2%}")
        print(f"  XGBoost: Top-1={metrics(type_xgb)['top1']:.2%}")
        print(f"  MLP: Top-1={metrics(type_mlp)['top1']:.2%}")

    # ============================================================
    # 保存结果
    # ============================================================
    results = {
        "iteration": 5,
        "n_tasks": len(tasks),
        "approach": "two_stage_reranking",
        "metrics": {
            "greedy_stage1": gm,
            "xgboost_reranker": xm,
            "mlp_reranker": mm
        },
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_5_results.json")
    return results


if __name__ == "__main__":
    run_iteration_5()
    print("\n" + "=" * 60)
    print("Iteration 5 Complete")
    print("=" * 60)
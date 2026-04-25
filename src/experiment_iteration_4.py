"""
XBRL Minimal Repair - Iteration 4
修复训练问题 + 只评估violation任务 + Pairwise Ranking
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
from tqdm import tqdm


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 1. Constraint Extraction (Same as Iteration 3)
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
# 2. Task Builder (Fixed)
# ============================================================

class ViolationTask:
    """基于违规的任务（只保留有violation）"""

    def __init__(self, original_table, modified_table, error_info, constraints):
        self.original_table = original_table
        self.modified_table = modified_table
        self.error_info = error_info
        self.constraints = constraints

        # 提取facts
        self.facts = self._extract_facts(modified_table)

        # 检查违规（来自原始约束）
        self.violations = self._check_violations()

        # Ground truth
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

                # 在modified table检查
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

    # 注入错误
    if sub_type == "sign_flip":
        new_val = -original_value
    elif sub_type == "scale_10":
        new_val = original_value * 10
    else:
        new_val = original_value + random.uniform(-0.3, 0.3) * abs(original_value)

    # 格式化
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
# 3. Baselines (Fixed)
# ============================================================

class ViolationGreedyBaseline:
    """基于violation的Greedy"""

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


class RoleAwareBaseline:
    """角色感知基线（区分total/component）"""

    def localize(self, task):
        if not task.violations:
            return [f["id"] for f in task.facts]

        fact_scores = defaultdict(float)

        for v in task.violations:
            # 找total位置（reported_total对应的fact）
            total_fact_id = None
            for f in task.facts:
                if abs(f["value"] - v["reported_total"]) < 0.1:
                    total_fact_id = f["id"]
                    break

            for fid in v["involved_facts"]:
                if fid == total_fact_id:
                    # Total位置得分较低（更可能是症状）
                    fact_scores[fid] += v["residual"] * 0.5
                else:
                    # Component位置得分较高（更可能是根因）
                    fact_scores[fid] += v["residual"] * 1.5

        ranked = sorted(fact_scores.keys(), key=lambda x: fact_scores[x], reverse=True)

        for f in task.facts:
            if f["id"] not in ranked:
                ranked.append(f["id"])

        return ranked


# ============================================================
# 4. GNN with Fixed Training
# ============================================================

class ConstraintAwareGNN(nn.Module):
    """约束感知GNN"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.fact_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Violation feature encoder (4 -> hidden_dim)
        self.violation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, violation_features=None):
        h = self.fact_encoder(x)

        # 加入violation信息 (编码后融合)
        if violation_features is not None and violation_features.size(0) > 0:
            # 编码violation特征到hidden_dim
            v_encoded = self.violation_encoder(violation_features)
            # 加到fact embedding上
            h = h + v_encoded

        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))

        scores = self.scorer(h).squeeze()
        return scores


def build_features(task):
    """构建特征"""
    facts = task.facts
    violations = task.violations

    if not facts:
        return torch.zeros((0, 8)), torch.zeros((0, 4))

    max_val = max(abs(f["value"]) for f in facts) if facts else 1

    # Fact特征
    fact_features = []
    for f in facts:
        involved = sum(1 for v in violations if f["id"] in v["involved_facts"])
        total_residual = sum(v["residual"] for v in violations if f["id"] in v["involved_facts"])

        is_total = any(abs(f["value"] - v["reported_total"]) < 0.1 for v in violations)

        features = [
            f["value"] / max_val,
            f["row"] / 10,
            f["col"] / 5,
            involved / max(1, len(violations)),
            total_residual / 1000,
            1.0 if is_total else 0.0,
            abs(f["value"]) / max_val,
            1.0
        ]
        fact_features.append(features)

    fact_features = torch.tensor(fact_features, dtype=torch.float)

    # Violation聚合特征（每个fact）
    violation_features = []
    for f in facts:
        vf = [0.0, 0.0, 0.0, 0.0]
        for v in violations:
            if f["id"] in v["involved_facts"]:
                vf[0] += v["residual"] / 1000
                vf[1] += 1.0
                if abs(f["value"] - v["reported_total"]) < 0.1:
                    vf[2] += 1.0
                else:
                    vf[3] += 1.0
        violation_features.append(vf)

    violation_features = torch.tensor(violation_features, dtype=torch.float)

    return fact_features, violation_features


def train_gnn_fixed(tasks, epochs=20, device='cuda'):
    """训练GNN（修复Loss）"""
    model = ConstraintAwareGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Pairwise ranking loss + BCE (正确实现)
    train_data = []

    for task in tasks:
        if not task.has_violation():
            continue

        fact_features, violation_features = build_features(task)
        if fact_features.size(0) < 2:
            continue

        # 标签
        labels = torch.zeros(fact_features.size(0))
        for i, f in enumerate(task.facts):
            if f["id"] == task.target_fact_id:
                labels[i] = 1.0

        train_data.append((fact_features, violation_features, labels, task))

    print(f"Training on {len(train_data)} violation tasks...")

    for epoch in range(epochs):
        total_loss = 0

        for fact_features, violation_features, labels, task in train_data:
            fact_features = fact_features.to(device)
            violation_features = violation_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(fact_features, violation_features)

            # 正确的加权BCE（weights不乘labels）
            weights = torch.ones_like(labels)
            weights[labels == 1] = 3.0  # 正例权重更高

            loss_bce = F.binary_cross_entropy_with_logits(
                scores, labels, reduction='none'
            )
            loss = (loss_bce * weights).mean()

            # Pairwise ranking loss（gold fact分数应比其他高）
            gold_idx = labels.argmax()
            gold_score = scores[gold_idx]
            other_scores = scores[labels == 0]

            if len(other_scores) > 0:
                # Margin ranking loss
                loss_rank = -F.logsigmoid(gold_score - other_scores).mean()
                loss = loss + 0.5 * loss_rank

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def gnn_localize(model, task, device='cuda'):
    """GNN定位"""
    fact_features, violation_features = build_features(task)
    if fact_features.size(0) == 0:
        return [f["id"] for f in task.facts]

    fact_features = fact_features.to(device)
    violation_features = violation_features.to(device)

    model.eval()
    with torch.no_grad():
        scores = model(fact_features, violation_features)

    scores_np = scores.cpu().numpy()
    ranked = np.argsort(-scores_np)
    return [task.facts[i]["id"] for i in ranked]


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

        # 只保留有violation的
        if task.has_violation():
            tasks.append(task)

    return tasks


def run_iteration_4():
    """运行第4轮实验"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 4")
    print("Fixed GNN Training + Violation-Only Evaluation")
    print("=" * 60)

    # 构建benchmark（只保留有violation）
    tasks = build_benchmark_violation_only()
    print(f"\nBuilt {len(tasks)} tasks (ALL with violations)")

    # 错误类型分布
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

    # Baselines
    print("\n[Step 1] Baseline Localization...")
    greedy = ViolationGreedyBaseline()
    role = RoleAwareBaseline()

    greedy_res = []
    role_res = []

    for t in test:
        greedy_res.append(t.evaluate(greedy.localize(t)))
        role_res.append(t.evaluate(role.localize(t)))

    def metrics(res):
        return {
            "top1": sum(r["hit_1"] for r in res) / len(res),
            "top3": sum(r["hit_3"] for r in res) / len(res),
            "top5": sum(r["hit_5"] for r in res) / len(res),
            "mrr": sum(r["mrr"] for r in res) / len(res)
        }

    print("\nViolation Greedy:")
    gm = metrics(greedy_res)
    print(f"  Top-1: {gm['top1']:.2%}, Top-3: {gm['top3']:.2%}, Top-5: {gm['top5']:.2%}, MRR: {gm['mrr']:.4f}")

    print("\nRole-Aware Baseline:")
    rm = metrics(role_res)
    print(f"  Top-1: {rm['top1']:.2%}, Top-3: {rm['top3']:.2%}, Top-5: {rm['top5']:.2%}, MRR: {rm['mrr']:.4f}")

    # Train GNN
    print("\n[Step 2] Training GNN (Fixed)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_gnn_fixed(train, epochs=20, device=device)

    # Eval GNN
    print("\n[Step 3] GNN Evaluation...")
    gnn_res = []
    for t in test:
        gnn_res.append(t.evaluate(gnn_localize(model, t, device)))

    print("\nConstraint-Aware GNN:")
    nnm = metrics(gnn_res)
    print(f"  Top-1: {nnm['top1']:.2%}, Top-3: {nnm['top3']:.2%}, Top-5: {nnm['top5']:.2%}, MRR: {nnm['mrr']:.4f}")

    # Save
    results = {
        "iteration": 4,
        "n_tasks": len(tasks),
        "all_with_violations": True,
        "metrics": {
            "greedy": gm,
            "role_aware": rm,
            "constraint_gnn": nnm
        },
        "error_types": dict(error_dist)
    }

    with open("data/benchmark/iteration_4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to data/benchmark/iteration_4_results.json")
    return results


if __name__ == "__main__":
    run_iteration_4()
    print("\n" + "=" * 60)
    print("Iteration 4 Complete")
    print("=" * 60)
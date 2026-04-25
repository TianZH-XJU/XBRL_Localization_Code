"""
XBRL Minimal Repair - Iteration 2
真正的不一致检测+修复任务（基于Codex审核反馈）
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from ortools.linear_solver import pywraplp
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# 1. Constraint-based Financial Table
# ============================================================

class ConstraintTable:
    """带约束的财务表格"""

    def __init__(self, table_data: List[List], name: str = "table"):
        self.raw_data = table_data
        self.name = name
        self.header = table_data[0] if table_data else []
        self.facts = self._extract_facts()
        self.constraints = self._build_constraints()
        self.violations = []

    def _extract_facts(self) -> List[Dict]:
        """提取数值事实"""
        facts = []
        for i, row in enumerate(self.raw_data):
            for j, cell in enumerate(row):
                num = self._parse_number(cell)
                if num is not None:
                    facts.append({
                        "id": f"fact_{i}_{j}",
                        "row": i,
                        "col": j,
                        "value": num,
                        "label": str(row[0]) if len(row) > 0 and i > 0 else f"cell_{i}_{j}",
                        "raw_cell": str(cell)
                    })
        return facts

    def _parse_number(self, cell) -> Optional[float]:
        """解析数值"""
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

    def _build_constraints(self) -> List[Dict]:
        """构建计算约束（加总关系）"""
        constraints = []

        # 找列间可能的加总关系（最后一列可能是前几列之和）
        ncols = len(self.raw_data[0]) if self.raw_data else 0

        if ncols >= 3:
            # 检查每行是否满足加总关系
            for i, row in enumerate(self.raw_data):
                values = []
                for j in range(ncols):
                    num = self._parse_number(row[j])
                    if num is not None:
                        values.append((j, num))

                # 检查最后一列是否等于前几列之和
                if len(values) >= 3:
                    # 尝试检测加总关系
                    for split_idx in range(1, len(values) - 1):
                        component_sum = sum(v for _, v in values[:split_idx])
                        total = values[-1][1]
                        if abs(component_sum - total) < abs(total) * 0.01:
                            constraints.append({
                                "type": "calc_sum",
                                "row": i,
                                "components": values[:split_idx],
                                "total": values[-1],
                                "equation": f"col[{values[-1][0]}] = sum(col[0..{split_idx-1}])"
                            })

        # 找Total行（行加总）
        for i, row in enumerate(self.raw_data):
            if len(row) > 0:
                label = str(row[0]).lower()
                if 'total' in label or 'sum' in label:
                    # 可能是其他行的加总
                    constraints.append({
                        "type": "row_total",
                        "row": i,
                        "label": row[0],
                        "is_total_row": True
                    })

        return constraints

    def check_violations(self) -> List[Dict]:
        """检查约束违规"""
        self.violations = []

        for c in self.constraints:
            if c["type"] == "calc_sum":
                component_sum = sum(v for _, v in c["components"])
                total_value = c["total"][1]
                residual = abs(component_sum - total_value)

                if residual > abs(total_value) * 0.01:
                    involved_facts = []
                    for j, v in c["components"]:
                        for fact in self.facts:
                            if fact["row"] == c["row"] and fact["col"] == j:
                                involved_facts.append(fact["id"])
                    for fact in self.facts:
                        if fact["row"] == c["row"] and fact["col"] == c["total"][0]:
                            involved_facts.append(fact["id"])

                    self.violations.append({
                        "constraint_id": f"c_{c['row']}_{c['total'][0]}",
                        "type": "calc_sum_violation",
                        "row": c["row"],
                        "residual": residual,
                        "involved_facts": involved_facts,
                        "expected_sum": component_sum,
                        "reported_total": total_value
                    })

        return self.violations

    def get_fact_by_id(self, fact_id: str) -> Optional[Dict]:
        """获取事实"""
        for fact in self.facts:
            if fact["id"] == fact_id:
                return fact
        return None


# ============================================================
# 2. Localization + Repair Task (Unknown Error Location)
# ============================================================

class LocalizationTask:
    """错误定位任务（未知错误位置）"""

    def __init__(self, table: ConstraintTable, error_fact_id: str,
                 original_value: float, error_type: str):
        self.table = table
        self.error_fact_id = error_fact_id  # Ground truth（不给模型）
        self.original_value = original_value
        self.error_type = error_type

        # 检查violation（模型输入）
        self.violations = table.check_violations()

    def evaluate_localization(self, predicted_fact_ids: List[str], k: int = 5) -> Dict:
        """评估定位"""
        hit_at_k = self.error_fact_id in predicted_fact_ids[:k]

        # MRR
        mrr = 0
        for i, fact_id in enumerate(predicted_fact_ids):
            if fact_id == self.error_fact_id:
                mrr = 1.0 / (i + 1)
                break

        return {
            "hit_at_1": self.error_fact_id == predicted_fact_ids[0] if predicted_fact_ids else False,
            "hit_at_3": self.error_fact_id in predicted_fact_ids[:3] if len(predicted_fact_ids) >= 3 else False,
            "hit_at_5": hit_at_k,
            "mrr": mrr
        }


# ============================================================
# 3. Baseline: Unknown Error Location
# ============================================================

class GreedyLocalizationBaseline:
    """Greedy定位基线"""

    def localize(self, task: LocalizationTask) -> List[str]:
        """定位错误单元格（从violation推断）"""
        if not task.violations:
            # 无violation，随机返回
            return [f["id"] for f in task.table.facts[:5]]

        # 统计每个fact参与violation的次数
        fact_counts = defaultdict(int)
        for v in task.violations:
            for fact_id in v["involved_facts"]:
                fact_counts[fact_id] += 1

        # 按参与度排序
        ranked = sorted(fact_counts.keys(), key=lambda x: fact_counts[x], reverse=True)

        # 补充未参与的fact
        all_facts = [f["id"] for f in task.table.facts]
        for fact_id in all_facts:
            if fact_id not in ranked:
                ranked.append(fact_id)

        return ranked[:5]


class RandomLocalizationBaseline:
    """随机定位基线"""

    def localize(self, task: LocalizationTask) -> List[str]:
        """随机返回"""
        all_facts = [f["id"] for f in task.table.facts]
        random.shuffle(all_facts)
        return all_facts[:5]


class ILPRepairBaseline:
    """ILP修复（修复候选集）"""

    def repair(self, task: LocalizationTask, candidate_fact_ids: List[str]) -> Dict:
        """找到最小修复"""
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return {"success": False, "error": "No solver"}

        facts = task.table.facts
        violations = task.violations

        if not facts or not violations:
            return {"success": False, "error": "No data"}

        n = len(facts)
        m = len(violations)

        # 变量
        z = [solver.IntVar(0, 1, f'z_{i}') for i in range(n)]  # 是否修改fact i
        v_new = [solver.NumVar(-1e10, 1e10, f'v_{i}') for i in range(n)]  # 新值

        # 约束：未修改则保持原值
        for i in range(n):
            solver.Add(v_new[i] >= facts[i]['value'] - 1e-6 * z[i])
            solver.Add(v_new[i] <= facts[i]['value'] + 1e-6 * z[i])

        # 约束：修复后满足所有calc约束（简化）
        # 只考虑简单的加总约束
        for v in violations:
            # 找涉及的fact indices
            indices = []
            for fact_id in v["involved_facts"]:
                for i, f in enumerate(facts):
                    if f["id"] == fact_id:
                        indices.append(i)

            if len(indices) >= 2:
                # 简化约束：至少有一个被修改
                solver.Add(sum(z[i] for i in indices) >= 1)

        # 目标：最小化修改数量
        solver.Minimize(sum(z))

        status = solver.Solve()

        repairs = []
        if status == pywraplp.Solver.OPTIMAL:
            for i in range(n):
                if z[i].solution_value() > 0.5:
                    repairs.append({
                        "fact_id": facts[i]["id"],
                        "original": facts[i]["value"],
                        "new_value": v_new[i].solution_value()
                    })

        return {
            "success": len(repairs) > 0,
            "repairs": repairs,
            "n_repairs": len(repairs)
        }


# ============================================================
# 4. GNN for Localization (Fixed Training)
# ============================================================

class TableGNN(nn.Module):
    """表格图神经网络（修正版）"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # Cell特征编码器
        self.cell_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 简化的图结构（不使用PyG）
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 错误检测头
        self.error_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 错误分数
        )

        # 错误类型头
        self.error_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # sign_flip, scale, value, none
        )

    def forward(self, x, adj=None):
        """前向传播"""
        h = self.cell_encoder(x)

        # 简化图传播（平均邻居）
        if adj is not None and adj.size(1) > 0:
            # 邻居聚合
            neighbor_sum = torch.zeros_like(h)
            for i in range(h.size(0)):
                neighbors = adj[1][adj[0] == i]
                if len(neighbors) > 0:
                    neighbor_sum[i] = h[neighbors].mean(dim=0)
            h = h + neighbor_sum

        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))

        error_scores = self.error_scorer(h).squeeze()
        error_types = self.error_type_head(h)

        return error_scores, error_types


def build_graph_features(table: ConstraintTable) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建图特征"""
    facts = table.facts
    n = len(facts)

    if n == 0:
        return torch.zeros((0, 6)), torch.zeros((2, 0), dtype=torch.long)

    # 节点特征
    x = []
    max_value = max(abs(f["value"]) for f in facts) if facts else 1

    for f in facts:
        features = [
            f["value"] / max_value,  # 归一化值
            f["row"] / 10,
            f["col"] / 5,
            1.0 if f["row"] == 0 else 0.0,
            1.0 if f["value"] < 0 else 0.0,
            1.0  # 常量
        ]
        x.append(features)

    x = torch.tensor(x, dtype=torch.float)

    # 边：同行、同列
    edges = []
    for i, f_i in enumerate(facts):
        for j, f_j in enumerate(facts):
            if i != j:
                if f_i["row"] == f_j["row"] or f_i["col"] == f_j["col"]:
                    edges.append([i, j])

    if edges:
        adj = torch.tensor(edges, dtype=torch.long).t()
    else:
        adj = torch.zeros((2, 0), dtype=torch.long)

    return x, adj


def train_gnn_localization(tasks: List[LocalizationTask], epochs: int = 20,
                           device: str = 'cuda') -> TableGNN:
    """训练GNN定位模型（修正版）"""
    model = TableGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 使用加权CE（一正多负）
    weights = torch.tensor([1.0, 1.0, 1.0, 0.1]).to(device)  # none类权重低
    criterion_type = nn.CrossEntropyLoss(weight=weights)
    criterion_score = nn.BCEWithLogitsLoss()

    train_data = []
    for task in tasks:
        x, adj = build_graph_features(task.table)
        if x.size(0) == 0:
            continue

        # 标签
        error_labels = torch.zeros(x.size(0), dtype=torch.float)
        type_labels = torch.full((x.size(0),), 3, dtype=torch.long)  # 默认none

        error_type_map = {"sign_flip": 0, "scale_error": 1, "value_error": 2}

        for i, f in enumerate(task.table.facts):
            if f["id"] == task.error_fact_id:
                error_labels[i] = 1.0
                type_labels[i] = error_type_map.get(task.error_type, 3)

        train_data.append((x, adj, error_labels, type_labels))

    print(f"Training on {len(train_data)} samples...")

    for epoch in range(epochs):
        total_loss = 0
        for x, adj, error_labels, type_labels in train_data:
            x = x.to(device)
            if adj.size(1) > 0:
                adj = adj.to(device)
            error_labels = error_labels.to(device)
            type_labels = type_labels.to(device)

            optimizer.zero_grad()
            error_scores, error_types = model(x, adj)

            # 只在错误节点计算repair相关loss
            # score loss（全节点）
            loss_score = criterion_score(error_scores, error_labels)

            # type loss（加权）
            loss_type = criterion_type(error_types, type_labels)

            loss = loss_score + 0.5 * loss_type
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def gnn_localize(model: TableGNN, task: LocalizationTask, device: str = 'cuda') -> List[str]:
    """GNN定位"""
    x, adj = build_graph_features(task.table)
    if x.size(0) == 0:
        return []

    x = x.to(device)
    if adj.size(1) > 0:
        adj = adj.to(device)

    model.eval()
    with torch.no_grad():
        error_scores, error_types = model(x, adj)

    # 按错误分数排序
    scores = error_scores.cpu().numpy()
    facts = task.table.facts

    ranked_indices = np.argsort(-scores)  # 降序
    ranked_fact_ids = [facts[i]["id"] for i in ranked_indices]

    return ranked_fact_ids[:5]


# ============================================================
# 5. Benchmark Builder (With Constraints)
# ============================================================

def build_benchmark_with_constraints(base_benchmark: Dict) -> List[LocalizationTask]:
    """构建带约束的benchmark"""
    tasks = []

    for instance in base_benchmark['instances']:
        table_data = instance['modified_table']
        table = ConstraintTable(table_data, instance['id'])

        # 构建任务（不泄露答案）
        error_row = instance['error_info']['target_row']
        error_col = instance['error_info']['target_col']
        error_fact_id = f"fact_{error_row}_{error_col}"

        task = LocalizationTask(
            table=table,
            error_fact_id=error_fact_id,
            original_value=instance['error_info']['original_value'],
            error_type=instance['error_info']['error_type']
        )

        tasks.append(task)

    return tasks


# ============================================================
# 6. Experiment Runner
# ============================================================

def run_iteration_2():
    """运行第2轮实验"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 2")
    print("Unknown Error Location + Constraint-based")
    print("=" * 60)

    # 加载原始benchmark
    with open("data/benchmark/inconsistency_repair_benchmark.json") as f:
        base_benchmark = json.load(f)

    # 构建任务
    tasks = build_benchmark_with_constraints(base_benchmark)
    print(f"\nBuilt {len(tasks)} localization tasks")

    # 统计violation分布
    n_with_violations = sum(1 for t in tasks if t.violations)
    print(f"Tasks with violations: {n_with_violations}/{len(tasks)}")

    # Baseline定位
    print("\n[Step 1] Running Baseline Localization...")

    greedy_baseline = GreedyLocalizationBaseline()
    random_baseline = RandomLocalizationBaseline()

    greedy_results = []
    random_results = []

    for task in tqdm(tasks, desc="Baseline localization"):
        greedy_pred = greedy_baseline.localize(task)
        random_pred = random_baseline.localize(task)

        greedy_results.append(task.evaluate_localization(greedy_pred))
        random_results.append(task.evaluate_localization(random_pred))

    # 计算metrics
    def compute_loc_metrics(results: List[Dict]) -> Dict:
        return {
            "top1": sum(1 for r in results if r["hit_at_1"]) / len(results),
            "top3": sum(1 for r in results if r["hit_at_3"]) / len(results),
            "top5": sum(1 for r in results if r["hit_at_5"]) / len(results),
            "mrr": sum(r["mrr"] for r in results) / len(results)
        }

    print("\nGreedy Localization:")
    greedy_metrics = compute_loc_metrics(greedy_results)
    print(f"  Top-1: {greedy_metrics['top1']:.2%}")
    print(f"  Top-3: {greedy_metrics['top3']:.2%}")
    print(f"  Top-5: {greedy_metrics['top5']:.2%}")
    print(f"  MRR: {greedy_metrics['mrr']:.4f}")

    print("\nRandom Localization:")
    random_metrics = compute_loc_metrics(random_results)
    print(f"  Top-1: {random_metrics['top1']:.2%}")
    print(f"  Top-3: {random_metrics['top3']:.2%}")
    print(f"  Top-5: {random_metrics['top5']:.2%}")
    print(f"  MRR: {random_metrics['mrr']:.4f}")

    # 训练GNN
    print("\n[Step 2] Training GNN for Localization...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Split
    split_idx = int(len(tasks) * 0.8)
    train_tasks = tasks[:split_idx]
    test_tasks = tasks[split_idx:]

    print(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    model = train_gnn_localization(train_tasks, epochs=30, device=device)

    # 评估GNN
    print("\n[Step 3] Evaluating GNN Localization...")
    gnn_results = []

    for task in tqdm(test_tasks, desc="GNN localization"):
        pred = gnn_localize(model, task, device=device)
        gnn_results.append(task.evaluate_localization(pred))

    print("\nGNN Localization (Test Set):")
    gnn_metrics = compute_loc_metrics(gnn_results)
    print(f"  Top-1: {gnn_metrics['top1']:.2%}")
    print(f"  Top-3: {gnn_metrics['top3']:.2%}")
    print(f"  Top-5: {gnn_metrics['top5']:.2%}")
    print(f"  MRR: {gnn_metrics['mrr']:.4f}")

    # 保存结果
    all_results = {
        "iteration": 2,
        "metrics": {
            "greedy": greedy_metrics,
            "random": random_metrics,
            "gnn_test": gnn_metrics
        },
        "n_tasks": len(tasks),
        "n_with_violations": n_with_violations
    }

    with open("data/benchmark/iteration_2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to data/benchmark/iteration_2_results.json")

    return all_results


if __name__ == "__main__":
    results = run_iteration_2()
    print("\n" + "=" * 60)
    print("Iteration 2 Complete")
    print("=" * 60)
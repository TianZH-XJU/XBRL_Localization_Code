"""
XBRL Minimal Repair - Iteration 3
真正的Constraint-based Localization + Repair
从original_table生成约束，确保violation存在
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
# 1. Constraint Extraction from ORIGINAL Table
# ============================================================

class FinancialConstraintExtractor:
    """从原始表格提取约束（不依赖modified_table）"""

    def extract_calc_constraints(self, table_data: List[List]) -> List[Dict]:
        """提取加总约束（从原始表格）"""
        constraints = []

        if not table_data or len(table_data) < 3:
            return constraints

        ncols = len(table_data[0]) if table_data else 0

        # 检查每行的加总关系
        for i, row in enumerate(table_data):
            values = []
            for j in range(ncols):
                num = self._parse_number(row[j])
                if num is not None:
                    values.append((j, num, str(row[0]) if j == 0 else ""))

            if len(values) >= 3:
                # 检查是否有加总关系：最后值 = 前几个之和
                for split_idx in range(1, len(values) - 1):
                    component_sum = sum(v for _, v, _ in values[:split_idx])
                    total = values[-1][1]
                    residual = abs(component_sum - total)

                    # 允许小误差（rounding）
                    if residual < max(abs(total) * 0.02, 10):
                        constraint = {
                            "type": "calc_sum",
                            "row": i,
                            "components": [(j, v) for j, v, _ in values[:split_idx]],
                            "total": (values[-1][0], values[-1][1]),
                            "equation": f"col_{values[-1][0]} = sum(cols_0-{split_idx-1})",
                            "original_residual": residual
                        }
                        constraints.append(constraint)
                        break  # 只取一个加总关系

        return constraints

    def extract_row_total_constraints(self, table_data: List[List]) -> List[Dict]:
        """提取行total约束（基于标签）"""
        constraints = []

        for i, row in enumerate(table_data):
            if len(row) > 0:
                label = str(row[0]).lower()
                if any(kw in label for kw in ['total', 'sum', 'subtotal', 'net']):
                    # 找对应的组成部分（通常是前面的行）
                    component_rows = []
                    for j in range(i):
                        prev_label = str(table_data[j][0]).lower()
                        if not any(kw in prev_label for kw in ['total', 'sum', 'subtotal']):
                            # 可能是组成部分
                            component_rows.append(j)

                    if component_rows:
                        constraints.append({
                            "type": "row_total",
                            "row": i,
                            "label": row[0],
                            "component_rows": component_rows,
                            "expected_components": len(component_rows)
                        })

        return constraints

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


class ConstraintBasedTable:
    """带约束的表格（约束来自original_table）"""

    def __init__(self, table_data: List[List], original_data: List[List] = None,
                 name: str = "table"):
        self.raw_data = table_data
        self.original_data = original_data if original_data else table_data
        self.name = name

        # 从原始表格提取约束！
        self.constraints = self._build_constraints_from_original()
        self.facts = self._extract_facts()

    def _build_constraints_from_original(self) -> List[Dict]:
        """从原始表格构建约束"""
        extractor = FinancialConstraintExtractor()

        calc_constraints = extractor.extract_calc_constraints(self.original_data)
        row_total_constraints = extractor.extract_row_total_constraints(self.original_data)

        all_constraints = calc_constraints + row_total_constraints
        return all_constraints

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
                        "label": str(row[0]) if len(row) > 0 and j == 0 else f"cell_{i}_{j}",
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

    def check_violations(self) -> List[Dict]:
        """检查约束违规（基于原始约束）"""
        violations = []

        for c in self.constraints:
            if c["type"] == "calc_sum":
                # 在当前表格上检查
                row = c["row"]

                # 获取当前值
                component_values = []
                for j, _ in c["components"]:
                    val = self._get_cell_value(row, j)
                    if val is not None:
                        component_values.append(val)

                total_j, original_total = c["total"]
                current_total = self._get_cell_value(row, total_j)

                if current_total is not None and component_values:
                    component_sum = sum(component_values)
                    residual = abs(component_sum - current_total)

                    # 现在允许更大的residual（因为扰动可能打破约束）
                    if residual > max(abs(current_total) * 0.01, 5):
                        involved_facts = []
                        for j, _ in c["components"]:
                            for fact in self.facts:
                                if fact["row"] == row and fact["col"] == j:
                                    involved_facts.append(fact["id"])
                        for fact in self.facts:
                            if fact["row"] == row and fact["col"] == total_j:
                                involved_facts.append(fact["id"])

                        violations.append({
                            "constraint_id": f"c_{row}_{total_j}",
                            "type": "calc_sum_violation",
                            "row": row,
                            "residual": residual,
                            "involved_facts": involved_facts,
                            "expected_sum": component_sum,
                            "reported_total": current_total,
                            "original_total": original_total
                        })

        return violations

    def _get_cell_value(self, row: int, col: int) -> Optional[float]:
        """获取单元格值"""
        if row < len(self.raw_data) and col < len(self.raw_data[row]):
            return self._parse_number(self.raw_data[row][col])
        return None


# ============================================================
# 2. Error Injection with Constraint Violation
# ============================================================

class ConstraintBasedErrorInjector:
    """注入会触发约束违规的错误"""

    def inject_calc_error(self, table_data: List[List], constraint: Dict,
                          error_type: str = "component") -> Tuple[List[List], Dict]:
        """注入计算约束错误"""
        modified_data = [row[:] for row in table_data]

        row = constraint["row"]

        if error_type == "component":
            # 修改一个子项
            if constraint["components"]:
                target_j, original_value = constraint["components"][0]

                # 随机选择错误类型
                sub_error_type = random.choice(["sign_flip", "scale_10", "value_random"])

                if sub_error_type == "sign_flip":
                    new_value = -original_value
                elif sub_error_type == "scale_10":
                    new_value = original_value * 10
                else:
                    delta = random.uniform(-0.3, 0.3) * abs(original_value)
                    new_value = original_value + delta

                # 修改单元格
                modified_data[row][target_j] = self._format_value(new_value, table_data[row][target_j])

                error_info = {
                    "target_row": row,
                    "target_col": target_j,
                    "original_value": original_value,
                    "error_value": new_value,
                    "error_type": sub_error_type,
                    "constraint_id": constraint.get("equation", f"calc_{row}")
                }

        elif error_type == "total":
            # 修改总和
            total_j, original_total = constraint["total"]

            sub_error_type = random.choice(["sign_flip", "scale_10", "value_random"])

            if sub_error_type == "sign_flip":
                new_value = -original_total
            elif sub_error_type == "scale_10":
                new_value = original_total * 10
            else:
                delta = random.uniform(-0.3, 0.3) * abs(original_total)
                new_value = original_total + delta

            modified_data[row][total_j] = self._format_value(new_value, table_data[row][total_j])

            error_info = {
                "target_row": row,
                "target_col": total_j,
                "original_value": original_total,
                "error_value": new_value,
                "error_type": sub_error_type,
                "constraint_id": constraint.get("equation", f"calc_{row}")
            }

        else:
            # 同时修改多个（更难）
            target_j, original_value = constraint["components"][0]
            modified_data[row][target_j] = self._format_value(-original_value, table_data[row][target_j])

            error_info = {
                "target_row": row,
                "target_col": target_j,
                "original_value": original_value,
                "error_value": -original_value,
                "error_type": "sign_flip",
                "constraint_id": constraint.get("equation", f"calc_{row}")
            }

        return modified_data, error_info

    def _format_value(self, new_value: float, original_cell) -> str:
        """格式化新值"""
        original_str = str(original_cell)

        if new_value < 0:
            # 用括号表示负数
            return f"({abs(int(new_value))})"
        else:
            return str(int(new_value))


# ============================================================
# 3. Localization + Repair Task
# ============================================================

class ConstraintBasedLocalizationTask:
    """基于约束的定位任务"""

    def __init__(self, original_table: List[List], modified_table: List[List],
                 error_info: Dict, constraints: List[Dict]):
        self.original_table = original_table
        self.table = ConstraintBasedTable(modified_table, original_table)
        self.error_info = error_info
        self.constraints = constraints

        # 检查违规
        self.violations = self.table.check_violations()

    def evaluate_localization(self, predicted_fact_ids: List[str]) -> Dict:
        """评估定位"""
        target_id = f"fact_{self.error_info['target_row']}_{self.error_info['target_col']}"

        hit_at_1 = predicted_fact_ids[0] == target_id if predicted_fact_ids else False
        hit_at_3 = target_id in predicted_fact_ids[:3] if len(predicted_fact_ids) >= 3 else False
        hit_at_5 = target_id in predicted_fact_ids[:5] if len(predicted_fact_ids) >= 5 else False

        # MRR
        mrr = 0
        for i, fact_id in enumerate(predicted_fact_ids):
            if fact_id == target_id:
                mrr = 1.0 / (i + 1)
                break

        return {
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "mrr": mrr,
            "has_violation": len(self.violations) > 0
        }


# ============================================================
# 4. Baselines (Constraint-based)
# ============================================================

class ViolationBasedGreedy:
    """基于违规的Greedy定位"""

    def localize(self, task: ConstraintBasedLocalizationTask) -> List[str]:
        """定位错误（从violation推断）"""
        if not task.violations:
            # 无违规时返回所有facts
            return [f["id"] for f in task.table.facts]

        # 统计每个fact参与violation的次数
        fact_counts = defaultdict(int)
        fact_residuals = defaultdict(float)

        for v in task.violations:
            for fact_id in v["involved_facts"]:
                fact_counts[fact_id] += 1
                fact_residuals[fact_id] += v["residual"]

        # 按参与度和residual排序
        ranked = sorted(
            fact_counts.keys(),
            key=lambda x: (fact_counts[x], fact_residuals[x]),
            reverse=True
        )

        # 补充未参与的
        all_facts = [f["id"] for f in task.table.facts]
        for fact_id in all_facts:
            if fact_id not in ranked:
                ranked.append(fact_id)

        return ranked


class PropagationGraphBaseline:
    """传播图基线（区分症状和根因）"""

    def localize(self, task: ConstraintBasedLocalizationTask) -> List[str]:
        """使用传播图推断根因"""
        if not task.violations:
            return [f["id"] for f in task.table.facts]

        # 构建传播图
        facts = task.table.facts
        violations = task.violations

        # 计算每个fact的"根因分数"
        root_scores = defaultdict(float)

        for v in violations:
            # 参与violation的facts
            for fact_id in v["involved_facts"]:
                # 找到对应的fact
                fact = None
                for f in facts:
                    if f["id"] == fact_id:
                        fact = f
                        break

                if fact:
                    # 分数考虑：
                    # 1. 是否是"total"位置（可能是症状）
                    # 2. 是否是"component"位置（可能是根因）
                    # 3. 数值大小
                    # 4. 参与violation数

                    is_total = any(v["reported_total"] == fact["value"] for v in violations)

                    if is_total:
                        # Total位置更可能是症状
                        root_scores[fact_id] += 0.5
                    else:
                        # Component位置更可能是根因
                        root_scores[fact_id] += 1.5

                    # 加上数值大小因子
                    root_scores[fact_id] += abs(fact["value"]) / 1e6

        # 按分数排序
        ranked = sorted(
            [f["id"] for f in facts],
            key=lambda x: root_scores.get(x, 0),
            reverse=True
        )

        return ranked


# ============================================================
# 5. GNN with Constraint Nodes
# ============================================================

class ConstraintGNN(nn.Module):
    """带约束节点的图神经网络"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # Fact编码器
        self.fact_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Constraint编码器
        self.constraint_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 图传播层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 错误检测头
        self.error_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, fact_features, constraint_features, fact_constraint_edges):
        """前向传播"""
        # 编码
        fact_h = self.fact_encoder(fact_features)
        constraint_h = self.constraint_encoder(constraint_features)

        # 传播：fact接收constraint信息
        if fact_constraint_edges.size(1) > 0:
            for i in range(fact_h.size(0)):
                # 找连接的constraints
                connected_constraints = fact_constraint_edges[1][fact_constraint_edges[0] == i]
                if len(connected_constraints) > 0:
                    constraint_info = constraint_h[connected_constraints].mean(dim=0)
                    fact_h[i] = self.fc1(torch.cat([fact_h[i], constraint_info]))

        fact_h = torch.relu(self.fc2(fact_h))

        # 输出错误分数
        error_scores = self.error_scorer(fact_h).squeeze()

        return error_scores


def build_constraint_graph_features(task: ConstraintBasedLocalizationTask):
    """构建带约束的图特征"""
    facts = task.table.facts
    violations = task.violations

    n_facts = len(facts)
    n_constraints = len(violations)

    if n_facts == 0:
        return (torch.zeros((0, 10)), torch.zeros((0, 5)),
                torch.zeros((2, 0), dtype=torch.long))

    # Fact特征
    max_value = max(abs(f["value"]) for f in facts) if facts else 1

    fact_features = []
    for f in facts:
        # 检查是否参与violation
        involved_count = sum(1 for v in violations if f["id"] in v["involved_facts"])
        total_residual = sum(v["residual"] for v in violations if f["id"] in v["involved_facts"])

        # 是否可能是total位置
        is_total = any(abs(v["reported_total"] - f["value"]) < 0.01 for v in violations)

        features = [
            f["value"] / max_value,
            f["row"] / 10,
            f["col"] / 5,
            1.0 if involved_count > 0 else 0.0,
            total_residual / 1000,
            1.0 if is_total else 0.0,
            involved_count / max(1, len(violations)),
            abs(f["value"]) / max_value,
            1.0 if f["value"] < 0 else 0.0,
            1.0  # bias
        ]
        fact_features.append(features)

    fact_features = torch.tensor(fact_features, dtype=torch.float)

    # Constraint特征
    constraint_features = []
    for v in violations:
        features = [
            v["residual"] / 1000,
            len(v["involved_facts"]) / 10,
            v["reported_total"] / max_value if max_value > 0 else 0,
            v["expected_sum"] / max_value if max_value > 0 else 0,
            1.0  # bias
        ]
        constraint_features.append(features)

    if constraint_features:
        constraint_features = torch.tensor(constraint_features, dtype=torch.float)
    else:
        constraint_features = torch.zeros((0, 5), dtype=torch.float)

    # 边：fact - constraint
    edges = []
    for c_idx, v in enumerate(violations):
        for fact_id in v["involved_facts"]:
            for f_idx, f in enumerate(facts):
                if f["id"] == fact_id:
                    edges.append([f_idx, c_idx])

    if edges:
        fact_constraint_edges = torch.tensor(edges, dtype=torch.long).t()
    else:
        fact_constraint_edges = torch.zeros((2, 0), dtype=torch.long)

    return fact_features, constraint_features, fact_constraint_edges


def train_constraint_gnn(tasks: List[ConstraintBasedLocalizationTask],
                        epochs: int = 30, device: str = 'cuda'):
    """训练Constraint GNN"""
    model = ConstraintGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # 构建训练数据
    train_data = []
    for task in tasks:
        fact_features, constraint_features, edges = build_constraint_graph_features(task)

        if fact_features.size(0) == 0:
            continue

        # 标签
        target_id = f"fact_{task.error_info['target_row']}_{task.error_info['target_col']}"
        labels = torch.zeros(fact_features.size(0))

        for i, f in enumerate(task.table.facts):
            if f["id"] == target_id:
                labels[i] = 1.0

        train_data.append((fact_features, constraint_features, edges, labels, task))

    print(f"Training on {len(train_data)} samples...")

    for epoch in range(epochs):
        total_loss = 0
        n_valid = 0

        for fact_features, constraint_features, edges, labels, task in train_data:
            if fact_features.size(0) == 0:
                continue

            fact_features = fact_features.to(device)
            constraint_features = constraint_features.to(device)
            edges = edges.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(fact_features, constraint_features, edges)

            # 只在参与violation的节点上计算loss（加权）
            involved_mask = torch.zeros(fact_features.size(0), device=device)
            for v in task.violations:
                for i, f in enumerate(task.table.facts):
                    if f["id"] in v["involved_facts"]:
                        involved_mask[i] = 1.0

            if involved_mask.sum() > 0:
                # 加权loss
                weights = involved_mask * 2 + 1  # involved节点权重更高
                weighted_labels = labels * weights
                loss = criterion(scores, weighted_labels)
            else:
                loss = criterion(scores, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_valid += 1

        if n_valid > 0:
            avg_loss = total_loss / n_valid
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def gnn_constraint_localize(model: ConstraintGNN, task: ConstraintBasedLocalizationTask,
                            device: str = 'cuda') -> List[str]:
    """GNN定位"""
    fact_features, constraint_features, edges = build_constraint_graph_features(task)

    if fact_features.size(0) == 0:
        return [f["id"] for f in task.table.facts]

    fact_features = fact_features.to(device)
    constraint_features = constraint_features.to(device)
    edges = edges.to(device)

    model.eval()
    with torch.no_grad():
        scores = model(fact_features, constraint_features, edges)

    # 排序
    scores_np = scores.cpu().numpy()
    facts = task.table.facts

    ranked_indices = np.argsort(-scores_np)
    ranked_ids = [facts[i]["id"] for i in ranked_indices]

    return ranked_ids


# ============================================================
# 6. Benchmark Builder
# ============================================================

def build_constraint_based_benchmark(base_benchmark: Dict) -> List[ConstraintBasedLocalizationTask]:
    """构建基于约束的benchmark（确保有violation）"""
    tasks = []
    injector = ConstraintBasedErrorInjector()

    for instance in tqdm(base_benchmark['instances'], desc="Building constraint tasks"):
        original_table = instance['modified_table']  # 原始数据（在benchmark里是modified_table作为原始）
        modified_table = [row[:] for row in original_table]  # 复制

        # 先提取约束
        extractor = FinancialConstraintExtractor()
        constraints = extractor.extract_calc_constraints(original_table)

        if not constraints:
            # 没有约束，使用原始benchmark的错误
            modified_table = instance['modified_table']  # 已经有扰动
            error_info = instance['error_info']
        else:
            # 有约束，注入会触发violation的错误
            constraint = random.choice(constraints)
            error_type = random.choice(["component", "total"])

            modified_table, error_info = injector.inject_calc_error(
                original_table, constraint, error_type
            )

        # 构建任务
        task = ConstraintBasedLocalizationTask(
            original_table=original_table,
            modified_table=modified_table,
            error_info=error_info,
            constraints=constraints
        )

        # 只保留有violation的任务
        if task.violations or not constraints:
            tasks.append(task)

    print(f"\nBuilt {len(tasks)} tasks")
    print(f"Tasks with violations: {sum(1 for t in tasks if t.violations)}/{len(tasks)}")

    return tasks


# ============================================================
# 7. Experiment Runner
# ============================================================

def run_iteration_3():
    """运行第3轮实验"""
    print("=" * 60)
    print("XBRL Minimal Repair - Iteration 3")
    print("Constraint-based Localization + Propagation Graph")
    print("=" * 60)

    # 加载原始benchmark
    with open("data/benchmark/inconsistency_repair_benchmark.json") as f:
        base_benchmark = json.load(f)

    # 构建constraint-based任务
    tasks = build_constraint_based_benchmark(base_benchmark)

    # 统计错误类型分布
    error_types = defaultdict(int)
    for t in tasks:
        error_types[t.error_info["error_type"]] += 1
    print(f"\nError type distribution: {dict(error_types)}")

    # Split (shuffle +按table)
    random.seed(42)
    random.shuffle(tasks)
    split_idx = int(len(tasks) * 0.8)
    train_tasks = tasks[:split_idx]
    test_tasks = tasks[split_idx:]

    print(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Baselines
    print("\n[Step 1] Running Baseline Localization...")

    greedy = ViolationBasedGreedy()
    pg_baseline = PropagationGraphBaseline()

    greedy_results = []
    pg_results = []

    for task in tqdm(test_tasks, desc="Baseline localization"):
        greedy_pred = greedy.localize(task)
        pg_pred = pg_baseline.localize(task)

        greedy_results.append(task.evaluate_localization(greedy_pred))
        pg_results.append(task.evaluate_localization(pg_pred))

    # 计算metrics
    def compute_metrics(results: List[Dict]) -> Dict:
        return {
            "top1": sum(1 for r in results if r["hit_at_1"]) / len(results),
            "top3": sum(1 for r in results if r["hit_at_3"]) / len(results),
            "top5": sum(1 for r in results if r["hit_at_5"]) / len(results),
            "mrr": sum(r["mrr"] for r in results) / len(results),
            "n_with_violation": sum(1 for r in results if r["has_violation"])
        }

    print("\nViolation-based Greedy:")
    greedy_metrics = compute_metrics(greedy_results)
    print(f"  Top-1: {greedy_metrics['top1']:.2%}")
    print(f"  Top-3: {greedy_metrics['top3']:.2%}")
    print(f"  Top-5: {greedy_metrics['top5']:.2%}")
    print(f"  MRR: {greedy_metrics['mrr']:.4f}")

    print("\nPropagation Graph Baseline:")
    pg_metrics = compute_metrics(pg_results)
    print(f"  Top-1: {pg_metrics['top1']:.2%}")
    print(f"  Top-3: {pg_metrics['top3']:.2%}")
    print(f"  Top-5: {pg_metrics['top5']:.2%}")
    print(f"  MRR: {pg_metrics['mrr']:.4f}")

    # 训练GNN
    print("\n[Step 2] Training Constraint GNN...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = train_constraint_gnn(train_tasks, epochs=30, device=device)

    # 评估GNN
    print("\n[Step 3] Evaluating Constraint GNN...")
    gnn_results = []

    for task in tqdm(test_tasks, desc="GNN localization"):
        pred = gnn_constraint_localize(model, task, device=device)
        gnn_results.append(task.evaluate_localization(pred))

    print("\nConstraint GNN (Test Set):")
    gnn_metrics = compute_metrics(gnn_results)
    print(f"  Top-1: {gnn_metrics['top1']:.2%}")
    print(f"  Top-3: {gnn_metrics['top3']:.2%}")
    print(f"  Top-5: {gnn_metrics['top5']:.2%}")
    print(f"  MRR: {gnn_metrics['mrr']:.4f}")

    # 保存
    all_results = {
        "iteration": 3,
        "metrics": {
            "greedy": greedy_metrics,
            "propagation_graph": pg_metrics,
            "constraint_gnn": gnn_metrics
        },
        "n_tasks": len(tasks),
        "n_train": len(train_tasks),
        "n_test": len(test_tasks),
        "n_with_violation_total": sum(1 for t in tasks if t.violations),
        "n_with_violation_test": sum(1 for t in test_tasks if t.violations),
        "error_types": dict(error_types)
    }

    with open("data/benchmark/iteration_3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to data/benchmark/iteration_3_results.json")

    return all_results


if __name__ == "__main__":
    results = run_iteration_3()

    print("\n" + "=" * 60)
    print("Iteration 3 Complete")
    print("=" * 60)
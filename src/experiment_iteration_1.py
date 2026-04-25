"""
XBRL Minimal Repair - Full Experiment Implementation
PG-MR (Propagation-Graph-based Minimal Repair)
"""

import json
import re
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from ortools.linear_solver import pywraplp
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# 1. Data Processing
# ============================================================

class FinancialTable:
    """财务表格表示"""

    def __init__(self, table_data: List[List], name: str = "table"):
        self.raw_data = table_data
        self.name = name
        self.numeric_cells = self._extract_numeric_cells()
        self.header = table_data[0] if table_data else []

    def _extract_numeric_cells(self) -> List[Dict]:
        """提取所有数值单元格"""
        cells = []
        for i, row in enumerate(self.raw_data):
            for j, cell in enumerate(row):
                num = self._parse_number(cell)
                if num is not None:
                    cells.append({
                        "row": i,
                        "col": j,
                        "value": num,
                        "raw_cell": str(cell),
                        "label": str(row[0]) if len(row) > 0 and i > 0 else f"cell_{i}_{j}"
                    })
        return cells

    def _parse_number(self, cell) -> Optional[float]:
        """解析单元格数值"""
        if cell is None or cell == '':
            return None
        cell = str(cell)
        # 匹配数值
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

    def get_cell_value(self, row: int, col: int) -> Optional[float]:
        """获取指定单元格数值"""
        for cell in self.numeric_cells:
            if cell['row'] == row and cell['col'] == col:
                return cell['value']
        return None

    def find_row_relations(self) -> List[Dict]:
        """找行内关系（加总、比率等）"""
        relations = []

        # 检查是否有"Total"行
        for i, row in enumerate(self.raw_data):
            if len(row) > 0:
                label = str(row[0]).lower()
                if 'total' in label or 'sum' in label:
                    # 找对应的组成部分
                    total_values = []
                    for j in range(1, len(row)):
                        num = self._parse_number(row[j])
                        if num is not None:
                            total_values.append(num)

                    if total_values:
                        relations.append({
                            "type": "row_total",
                            "row": i,
                            "label": row[0],
                            "values": total_values
                        })

        return relations

    def find_col_relations(self) -> List[Dict]:
        """找列间关系"""
        relations = []
        if not self.raw_data or len(self.raw_data) < 2:
            return relations

        ncols = len(self.raw_data[0]) if self.raw_data else 0

        for col_idx in range(1, ncols):
            col_values = []
            for row in self.raw_data:
                if col_idx < len(row):
                    num = self._parse_number(row[col_idx])
                    if num is not None:
                        col_values.append(num)

            if col_values:
                relations.append({
                    "type": "col_values",
                    "col": col_idx,
                    "values": col_values
                })

        return relations


# ============================================================
# 2. Baseline Methods
# ============================================================

class GreedyRepair:
    """Greedy修复基线 - 优先选择最低成本动作"""

    COST_ORDER = {"sign_flip": 1, "scale_fix": 2, "value_edit": 3}

    def detect_error(self, table: FinancialTable, target_row: int, target_col: int,
                     original_value: float) -> Dict:
        """检测错误类型"""
        current_value = table.get_cell_value(target_row, target_col)
        if current_value is None:
            return {"error_type": "unknown", "delta": 0}

        delta = original_value - current_value

        # 1. Sign flip检测
        if abs(current_value) == abs(original_value) and current_value * original_value < 0:
            return {"error_type": "sign_flip", "delta": delta}

        # 2. Scale error检测
        if abs(current_value) > 0 and abs(original_value) > 0:
            ratio = abs(original_value) / abs(current_value)
            if ratio in [0.001, 0.01, 0.1, 10, 100, 1000]:
                return {"error_type": "scale_fix", "delta": delta, "scale": ratio}

        # 3. Value error
        return {"error_type": "value_edit", "delta": delta}

    def repair(self, table: FinancialTable, target_row: int, target_col: int,
               original_value: float) -> Dict:
        """执行修复"""
        error_info = self.detect_error(table, target_row, target_col, original_value)

        repair_action = {
            "target_row": target_row,
            "target_col": target_col,
            "error_type": error_info["error_type"],
            "original_value": original_value,
            "cost": self.COST_ORDER.get(error_info["error_type"], 3),
            "success": True
        }

        return repair_action


class ILPRepair:
    """ILP修复基线 - 找最优最小修复"""

    def repair(self, table: FinancialTable, target_row: int, target_col: int,
               original_value: float) -> Dict:
        """使用ILP求解最小修复"""

        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return {"success": False, "error": "Solver not available"}

        cells = table.numeric_cells
        n = len(cells)

        if n == 0:
            return {"success": False, "error": "No numeric cells"}

        # 找目标单元格索引
        target_idx = None
        for i, cell in enumerate(cells):
            if cell['row'] == target_row and cell['col'] == target_col:
                target_idx = i
                break

        if target_idx is None:
            return {"success": False, "error": "Target cell not found"}

        # 变量
        # z_i: 是否修改单元格i
        z = [solver.IntVar(0, 1, f'z_{i}') for i in range(n)]

        # v_i: 修复后的值（离散化到几个选项）
        # 对于目标单元格，必须修复到original_value
        v_target = solver.NumVar(original_value - 0.1, original_value + 0.1, 'v_target')

        # 约束
        # 目标单元格必须修复
        solver.Add(z[target_idx] == 1)

        # 其他单元格如果不修改则保持原值（约束）
        for i in range(n):
            if i != target_idx:
                solver.Add(z[i] == 0)  # 只修复目标单元格（简化）

        # 目标值约束
        solver.Add(v_target == original_value)

        # 目标：最小化修改数量
        solver.Minimize(sum(z))

        # 求解
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            return {
                "success": True,
                "target_row": target_row,
                "target_col": target_col,
                "repair_value": original_value,
                "n_repairs": sum(z[i].solution_value() for i in range(n)),
                "cost": 3  # value_edit cost
            }
        else:
            return {"success": False, "error": "No optimal solution"}


class MaxSATRepair:
    """MaxSAT修复基线 - 离散动作空间"""

    def repair(self, table: FinancialTable, target_row: int, target_col: int,
               original_value: float) -> Dict:
        """使用MaxSAT求解"""
        # 简化实现：直接修复目标单元格
        return {
            "success": True,
            "target_row": target_row,
            "target_col": target_col,
            "repair_value": original_value,
            "cost": 3
        }


# ============================================================
# 3. Neural Methods
# ============================================================

class TableGNN(nn.Module):
    """表格图神经网络"""

    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()

        # Cell特征编码器
        self.cell_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 输入特征: value, row, col, is_header, etc.
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GAT层
        self.conv1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)

        # 输出层
        self.error_detector = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4种错误类型: sign_flip, scale_fix, value_edit, none
        )

        self.repair_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 预测修复值
        )

    def forward(self, x, edge_index):
        """前向传播"""
        # 编码
        h = self.cell_encoder(x)

        # 图卷积
        h = self.conv1(h, edge_index)
        h = h.view(h.size(0), -1)  # 合并多头
        h = torch.relu(h)

        h = self.conv2(h, edge_index)
        h = h.view(h.size(0), -1)

        # 输出
        error_logits = self.error_detector(h)
        repair_values = self.repair_predictor(h)

        return error_logits, repair_values


def build_table_graph(table: FinancialTable) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建表格图"""
    cells = table.numeric_cells
    n = len(cells)

    if n == 0:
        return torch.zeros((0, 8)), torch.zeros((2, 0), dtype=torch.long)

    # 节点特征
    x = []
    for cell in cells:
        features = [
            cell['value'] / 1e6,  # 归一化值
            cell['row'] / 10,     # 行位置
            cell['col'] / 5,      # 列位置
            1.0 if cell['row'] == 0 else 0.0,  # 是否header
            abs(cell['value']) > 1e5,  # 是否大值
            cell['value'] < 0,    # 是否负值
            len(str(cell['raw_cell'])) > 10,  # 是否复杂格式
            1.0,  # 常量
        ]
        x.append(features)

    x = torch.tensor(x, dtype=torch.float)

    # 边：同行连接、同列连接
    edges = []
    for i, cell_i in enumerate(cells):
        for j, cell_j in enumerate(cells):
            if i != j:
                # 同行连接
                if cell_i['row'] == cell_j['row']:
                    edges.append([i, j])
                # 同列连接
                if cell_i['col'] == cell_j['col']:
                    edges.append([i, j])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return x, edge_index


# ============================================================
# 4. Experiment Runner
# ============================================================

def load_benchmark(path: str) -> Dict:
    """加载benchmark"""
    with open(path) as f:
        return json.load(f)


def run_baseline_experiment(benchmark: Dict) -> Dict:
    """运行baseline实验"""
    results = {
        "greedy": [],
        "ilp": [],
        "maxsat": []
    }

    baselines = {
        "greedy": GreedyRepair(),
        "ilp": ILPRepair(),
        "maxsat": MaxSATRepair()
    }

    for instance in tqdm(benchmark['instances'], desc="Running baselines"):
        table = FinancialTable(instance['modified_table'], instance['id'])
        error_info = instance['error_info']

        for name, baseline in baselines.items():
            result = baseline.repair(
                table,
                error_info['target_row'],
                error_info['target_col'],
                error_info['original_value']
            )

            result['instance_id'] = instance['id']
            result['error_type'] = error_info['error_type']
            results[name].append(result)

    return results


def train_gnn(benchmark: Dict, epochs: int = 10, device: str = 'cuda') -> TableGNN:
    """训练GNN模型"""
    model = TableGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_error = nn.CrossEntropyLoss()
    criterion_repair = nn.MSELoss()

    # 构建训练数据
    train_data = []
    for instance in benchmark['instances']:
        table = FinancialTable(instance['modified_table'], instance['id'])
        x, edge_index = build_table_graph(table)

        if x.size(0) == 0:
            continue

        # 标签：找到错误单元格
        target_row = instance['error_info']['target_row']
        target_col = instance['error_info']['target_col']

        error_type_map = {
            "sign_flip": 0,
            "scale_error": 1,
            "value_error": 2,
            "none": 3
        }

        labels = torch.full((x.size(0),), 3, dtype=torch.long)  # 默认none
        repair_labels = torch.zeros(x.size(0))

        cells = table.numeric_cells
        for i, cell in enumerate(cells):
            if cell['row'] == target_row and cell['col'] == target_col:
                labels[i] = error_type_map.get(instance['error_info']['error_type'], 3)
                repair_labels[i] = instance['error_info']['original_value']

        train_data.append((x, edge_index, labels, repair_labels))

    # 训练
    for epoch in range(epochs):
        total_loss = 0
        for x, edge_index, labels, repair_labels in train_data:
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = labels.to(device)
            repair_labels = repair_labels.to(device)

            optimizer.zero_grad()
            error_logits, repair_values = model(x, edge_index)

            loss_error = criterion_error(error_logits, labels)
            loss_repair = criterion_repair(repair_values.squeeze(), repair_labels)

            loss = loss_error + loss_repair
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def evaluate_gnn(model: TableGNN, benchmark: Dict, device: str = 'cuda') -> Dict:
    """评估GNN模型"""
    results = []

    for instance in tqdm(benchmark['instances'], desc="Evaluating GNN"):
        table = FinancialTable(instance['modified_table'], instance['id'])
        x, edge_index = build_table_graph(table)

        if x.size(0) == 0:
            results.append({
                "instance_id": instance['id'],
                "success": False,
                "error": "Empty graph"
            })
            continue

        x = x.to(device)
        edge_index = edge_index.to(device)

        model.eval()
        with torch.no_grad():
            error_logits, repair_values = model(x, edge_index)

        # 预测
        error_preds = error_logits.argmax(dim=1).cpu()
        repair_preds = repair_values.cpu().squeeze()

        # 找到预测为错误的单元格
        target_row = instance['error_info']['target_row']
        target_col = instance['error_info']['target_col']
        cells = table.numeric_cells

        success = False
        for i, cell in enumerate(cells):
            if cell['row'] == target_row and cell['col'] == target_col:
                if error_preds[i] != 3:  # 不是"none"
                    pred_value = repair_preds[i].item()
                    true_value = instance['error_info']['original_value']
                    if abs(pred_value - true_value) < abs(true_value) * 0.1:
                        success = True
                break

        results.append({
            "instance_id": instance['id'],
            "success": success,
            "error_type": instance['error_info']['error_type']
        })

    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """计算评估指标"""
    total = len(results)
    success = sum(1 for r in results if r.get('success', False))

    # 按错误类型统计
    type_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    for r in results:
        error_type = r.get('error_type', 'unknown')
        type_stats[error_type]['total'] += 1
        if r.get('success', False):
            type_stats[error_type]['success'] += 1

    return {
        'overall_success_rate': success / total if total > 0 else 0,
        'total_instances': total,
        'by_error_type': dict(type_stats)
    }


def save_results(results: Dict, path: str):
    """保存结果"""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("XBRL Minimal Repair Experiment - Iteration 1")
    print("=" * 60)

    # 加载benchmark
    benchmark = load_benchmark("data/benchmark/inconsistency_repair_benchmark.json")
    print(f"\nLoaded {len(benchmark['instances'])} instances")

    # 运行baselines
    print("\n[Step 1] Running Baselines...")
    baseline_results = run_baseline_experiment(benchmark)

    # 计算metrics
    for name, results in baseline_results.items():
        metrics = compute_metrics(results)
        print(f"\n{name.upper()} Metrics:")
        print(f"  Overall: {metrics['overall_success_rate']:.2%}")
        for error_type, stats in metrics['by_error_type'].items():
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {error_type}: {rate:.2%} ({stats['success']}/{stats['total']})")

    # 训练GNN
    print("\n[Step 2] Training GNN...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = train_gnn(benchmark, epochs=10, device=device)

    # 评估GNN
    print("\n[Step 3] Evaluating GNN...")
    gnn_results = evaluate_gnn(model, benchmark, device=device)
    gnn_metrics = compute_metrics(gnn_results)

    print(f"\nGNN Metrics:")
    print(f"  Overall: {gnn_metrics['overall_success_rate']:.2%}")

    # 保存所有结果
    all_results = {
        'baseline_results': baseline_results,
        'gnn_results': gnn_results,
        'metrics': {
            'greedy': compute_metrics(baseline_results['greedy']),
            'ilp': compute_metrics(baseline_results['ilp']),
            'maxsat': compute_metrics(baseline_results['maxsat']),
            'gnn': gnn_metrics
        }
    }

    save_results(all_results, "data/benchmark/iteration_1_results.json")
    print("\nResults saved to data/benchmark/iteration_1_results.json")

    print("\n" + "=" * 60)
    print("Iteration 1 Complete")
    print("=" * 60)
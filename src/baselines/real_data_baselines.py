"""
XBRL Minimal Repair Experiment - Using Real TAT-QA Financial Tables
"""

import json
import re
import random
from typing import List, Dict, Tuple, Optional
from ortools.linear_solver import pywraplp


class FinancialTable:
    """财务表格表示"""
    def __init__(self, table_data: List[List], name: str = "table"):
        self.raw_data = table_data
        self.name = name
        self.numeric_cells = self._extract_numeric_cells()

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
                        "raw_cell": str(cell)
                    })
        return cells

    def _parse_number(self, cell) -> Optional[float]:
        """解析单元格数值"""
        if cell is None or cell == '':
            return None
        cell = str(cell)
        # 匹配数值（括号表示负数）
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

    def set_cell_value(self, row: int, col: int, new_value: float) -> str:
        """设置单元格值"""
        # 构造新单元格字符串
        if new_value < 0:
            return f"({abs(int(new_value))})"
        else:
            return str(int(new_value))

    def find_calculation_relations(self) -> List[Dict]:
        """找可能的加总关系"""
        relations = []

        # 检查列是否可能是其他列的和
        ncols = len(self.raw_data[0]) if self.raw_data else 0

        for col_idx in range(ncols):
            col_values = []
            for row_idx, row in enumerate(self.raw_data):
                if col_idx < len(row):
                    num = self._parse_number(row[col_idx])
                    if num is not None:
                        col_values.append((row_idx, num))

            # 如果列有多个值，检查是否可能构成加总关系
            if len(col_values) >= 3:
                # 尝试找加总关系（如前N个之和等于最后一个）
                for i in range(len(col_values) - 1):
                    sum_values = sum(v for _, v in col_values[:i+1])
                    last_value = col_values[-1][1]
                    if abs(sum_values - last_value) < 1:
                        relations.append({
                            "type": "col_sum",
                            "col": col_idx,
                            "components": col_values[:i+1],
                            "total": col_values[-1],
                            "residual": abs(sum_values - last_value)
                        })

        return relations


class RepairAction:
    """修复动作"""
    def __init__(self, row: int, col: int, action_type: str,
                 original_value: float, new_value: float, cost: float):
        self.row = row
        self.col = col
        self.action_type = action_type
        self.original_value = original_value
        self.new_value = new_value
        self.cost = cost


class GreedyRepairBaseline:
    """Greedy修复基线"""

    COST_ORDER = {"sign_flip": 1, "scale_fix": 2, "value_edit": 3}

    def repair(self, table: FinancialTable, target_value: float,
               target_row: int, target_col: int) -> List[RepairAction]:
        """修复表格使目标单元格恢复正确值"""
        repairs = []

        current_value = table.get_cell_value(target_row, target_col)
        if current_value is None:
            return repairs

        # 计算需要的修复
        delta = target_value - current_value

        if delta == 0:
            return repairs  # 无需修复

        # 尝试不同修复策略
        # 1. Sign flip
        if current_value * target_value < 0:  # 符号相反
            repairs.append(RepairAction(
                target_row, target_col, "sign_flip",
                current_value, target_value, 1
            ))
            return repairs

        # 2. Scale fix
        if abs(current_value) > 0 and abs(target_value) > 0:
            ratio = abs(target_value) / abs(current_value)
            if ratio in [0.1, 10, 100, 1000]:
                repairs.append(RepairAction(
                    target_row, target_col, "scale_fix",
                    current_value, target_value, 2
                ))
                return repairs

        # 3. Value edit (直接修改)
        repairs.append(RepairAction(
            target_row, target_col, "value_edit",
            current_value, target_value, 3
        ))

        return repairs


class ILPRepairBaseline:
    """ILP修复基线"""

    def repair(self, table: FinancialTable, target_value: float,
               target_row: int, target_col: int) -> List[RepairAction]:
        """使用ILP找到最小代价修复"""

        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return []

        cells = table.numeric_cells
        n = len(cells)

        if n == 0:
            return []

        # 变量：是否修改每个单元格
        z = [solver.IntVar(0, 1, f'z_{i}') for i in range(n)]

        # 变量：修复后的值
        v_new = [solver.NumVar(-1e10, 1e10, f'v_{i}') for i in range(n)]

        # 约束：如果未修改则保持原值
        for i in range(n):
            solver.Add(v_new[i] >= cells[i]['value'] - 1e-6 * z[i])
            solver.Add(v_new[i] <= cells[i]['value'] + 1e-6 * z[i])

        # 目标：最小化修改数量
        solver.Minimize(sum(z))

        status = solver.Solve()

        repairs = []
        if status == pywraplp.Solver.OPTIMAL:
            for i in range(n):
                if z[i].solution_value() > 0.5:
                    repairs.append(RepairAction(
                        cells[i]['row'], cells[i]['col'], "value_edit",
                        cells[i]['value'], v_new[i].solution_value(), 3
                    ))

        return repairs


def run_experiment(benchmark_path: str):
    """运行实验"""
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    print(f"Running experiment on {len(benchmark['instances'])} instances")

    results = {
        "greedy": [],
        "ilp": []
    }

    baselines = {
        "greedy": GreedyRepairBaseline(),
        "ilp": ILPRepairBaseline()
    }

    for instance in benchmark['instances']:
        table = FinancialTable(instance['modified_table'], instance['id'])
        error_info = instance['error_info']

        # 目标是恢复原始值
        target_value = error_info['original_value']
        target_row = error_info['target_row']
        target_col = error_info['target_col']

        for name, baseline in baselines.items():
            repairs = baseline.repair(table, target_value, target_row, target_col)

            # 评估修复
            success = False
            for r in repairs:
                if r.row == target_row and r.col == target_col:
                    if abs(r.new_value - target_value) < 0.1:
                        success = True
                        break

            results[name].append({
                "instance_id": instance['id'],
                "error_type": error_info['error_type'],
                "success": success,
                "n_repairs": len(repairs),
                "repair_types": [r.action_type for r in repairs],
                "total_cost": sum(r.cost for r in repairs) if repairs else 0
            })

    return results


def print_results(results: Dict):
    """打印结果统计"""
    for name, res_list in results.items():
        if not res_list:
            continue

        success_rate = sum(1 for r in res_list if r['success']) / len(res_list)
        avg_repairs = sum(r['n_repairs'] for r in res_list) / len(res_list)
        avg_cost = sum(r['total_cost'] for r in res_list) / len(res_list)

        # 按错误类型统计
        type_stats = {}
        for r in res_list:
            et = r['error_type']
            if et not in type_stats:
                type_stats[et] = {'success': 0, 'total': 0}
            type_stats[et]['total'] += 1
            if r['success']:
                type_stats[et]['success'] += 1

        print(f"\n{name.upper()} Results:")
        print(f"  Overall success rate: {success_rate:.2%}")
        print(f"  Avg repairs: {avg_repairs:.2f}")
        print(f"  Avg cost: {avg_cost:.2f}")

        print(f"  By error type:")
        for et, stats in type_stats.items():
            rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"    {et}: {rate:.2%} ({stats['success']}/{stats['total']})")


if __name__ == "__main__":
    print("="*60)
    print("XBRL Minimal Repair Experiment - Real TAT-QA Data")
    print("="*60)

    results = run_experiment("data/benchmark/inconsistency_repair_benchmark.json")
    print_results(results)

    # 保存结果
    with open("data/benchmark/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to data/benchmark/experiment_results.json")
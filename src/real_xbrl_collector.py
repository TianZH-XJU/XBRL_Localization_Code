"""
Real XBRL Data Collection Script
从SEC EDGAR获取真实XBRL计算不一致案例
"""

import requests
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time


class SECXBRLCollector:
    """从SEC EDGAR收集真实XBRL数据"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Research Project your.email@university.edu"
        }
        self.base_url = "https://www.sec.gov"

    def get_company_filings(self, cik: str, filing_type: str = "10-K") -> List[Dict]:
        """获取某公司的所有filing列表"""
        url = f"{self.base_url}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={filing_type}&output=json"

        try:
            response = requests.get(url, headers=self.headers)
            data = response.json()
            return data.get("filings", [])
        except Exception as e:
            print(f"Error fetching filings: {e}")
            return []

    def get_xbrl_documents(self, accession_number: str) -> List[Dict]:
        """获取特定filing的XBRL文档列表"""
        # 移除accession number中的横线
        acc_num = accession_number.replace("-", "")

        # 获取filing详情
        url = f"{self.base_url}/Archives/edgar/data/{acc_num}/index.json"

        try:
            response = requests.get(url, headers=self.headers)
            data = response.json()

            # 筛选XBRL相关文档
            xbrl_docs = []
            for item in data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith(".xml") and ("_cal" in name or "_def" in name):
                    xbrl_docs.append({
                        "name": name,
                        "type": "calculation" if "_cal" in name else "definition",
                        "url": f"{self.base_url}/Archives/edgar/data/{acc_num}/{name}"
                    })

            return xbrl_docs
        except Exception as e:
            print(f"Error fetching XBRL docs: {e}")
            return []

    def parse_calculation_linkbase(self, url: str) -> Dict:
        """解析calculation linkbase，提取计算关系"""
        try:
            response = requests.get(url, headers=self.headers)
            root = ET.fromstring(response.content)

            calculations = {}

            # 解析calculationArc元素
            # XBRL calculation linkbase结构复杂，这里简化处理
            for arc in root.iter():
                if 'calculationArc' in arc.tag or 'arc' in arc.tag.lower():
                    # 提取parent-child关系
                    parent = arc.get('parent', '')
                    child = arc.get('child', '')
                    weight = arc.get('weight', '1')

                    if parent and child:
                        if parent not in calculations:
                            calculations[parent] = []
                        calculations[parent].append({
                            'child': child,
                            'weight': float(weight)
                        })

            return calculations
        except Exception as e:
            print(f"Error parsing calculation linkbase: {e}")
            return {}

    def extract_facts_from_instance(self, url: str) -> Dict:
        """从XBRL实例文档提取数值"""
        try:
            response = requests.get(url, headers=self.headers)
            root = ET.fromstring(response.content)

            facts = {}

            # XBRL facts通常在 <numeric> 或 <nonNumeric> 元素中
            # 命名空间复杂，需要处理多个namespace
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                if tag in ['numeric', 'nonNumeric', 'fact']:
                    # 提取fact id和value
                    fact_id = elem.get('id', '')
                    context_ref = elem.get('contextRef', '')
                    unit_ref = elem.get('unitRef', '')

                    value_text = elem.text
                    if value_text:
                        try:
                            value = float(value_text.strip())
                            facts[fact_id] = {
                                'value': value,
                                'context': context_ref,
                                'unit': unit_ref
                            }
                        except:
                            pass

            return facts
        except Exception as e:
            print(f"Error parsing instance: {e}")
            return {}

    def find_calculation_inconsistencies(self, facts: Dict, calculations: Dict) -> List[Dict]:
        """发现计算不一致"""
        inconsistencies = []

        for parent, children in calculations.items():
            # 检查: sum(children * weight) ≈ parent
            if parent not in facts:
                continue

            parent_value = facts[parent]['value']

            # 计算子项加权和
            child_sum = 0.0
            missing_children = []

            for child_info in children:
                child_id = child_info['child']
                weight = child_info['weight']

                if child_id in facts:
                    child_sum += facts[child_id]['value'] * weight
                else:
                    missing_children.append(child_id)

            # 计算残差
            residual = abs(parent_value - child_sum)

            # 如果残差超过阈值，记录不一致
            threshold = max(abs(parent_value) * 0.01, 1)  # 1%或至少1

            if residual > threshold:
                inconsistencies.append({
                    'parent': parent,
                    'parent_value': parent_value,
                    'expected_sum': child_sum,
                    'residual': residual,
                    'children': children,
                    'missing': missing_children
                })

        return inconsistencies

    def collect_sample_cases(self, n_cases: int = 50) -> List[Dict]:
        """收集真实XBRL计算不一致案例"""

        # 一些常见的大公司CIK
        sample_ciks = [
            "0000320193",  # Apple
            "0001067983",  # Berkshire Hathaway
            "0000051143",  # IBM
            "0000789019",  # Microsoft
            "0001018724",  # Amazon
            "0001326801",  # Meta
            "0001652044",  # Google
            "0000320193",  # Tesla
        ]

        collected_cases = []

        for cik in sample_ciks:
            print(f"\nProcessing CIK: {cik}")

            filings = self.get_company_filings(cik, "10-K")

            for filing in filings[:3]:  # 只看最近3个filing
                accession = filing.get("accessionNumber", "")
                if not accession:
                    continue

                print(f"  Filing: {accession}")
                time.sleep(0.5)  # 遵守rate limit

                xbrl_docs = self.get_xbrl_documents(accession)

                if not xbrl_docs:
                    continue

                # 解析calculation linkbase
                calc_url = None
                instance_url = None

                for doc in xbrl_docs:
                    if doc['type'] == 'calculation':
                        calc_url = doc['url']
                    elif doc['name'].endswith('.xml') and '_cal' not in doc['name']:
                        instance_url = doc['url']

                if calc_url and instance_url:
                    calculations = self.parse_calculation_linkbase(calc_url)
                    facts = self.extract_facts_from_instance(instance_url)

                    if calculations and facts:
                        inconsistencies = self.find_calculation_inconsistencies(facts, calculations)

                        for inc in inconsistencies:
                            collected_cases.append({
                                'cik': cik,
                                'accession': accession,
                                'inconsistency': inc,
                                'calculations': calculations,
                                'facts': facts
                            })

                            if len(collected_cases) >= n_cases:
                                return collected_cases

            time.sleep(1)  # 公司间rate limit

        return collected_cases


def main():
    """收集真实XBRL案例"""
    print("=" * 60)
    print("Real XBRL Data Collection from SEC EDGAR")
    print("=" * 60)

    collector = SECXBRLCollector()

    # 收集10个真实案例用于快速测试
    cases = collector.collect_sample_cases(n_cases=10)

    print(f"\nCollected {len(cases)} real XBRL calculation inconsistency cases")

    # 保存结果
    with open("data/benchmark/real_xbrl_cases.json", "w") as f:
        json.dump(cases, f, indent=2, default=str)

    print("Saved to data/benchmark/real_xbrl_cases.json")

    # 分析统计
    if cases:
        residuals = [c['inconsistency']['residual'] for c in cases]
        print(f"\nStatistics:")
        print(f"  Mean residual: {sum(residuals)/len(residuals):.2f}")
        print(f"  Max residual: {max(residuals):.2f}")
        print(f"  Min residual: {min(residuals):.2f}")


if __name__ == "__main__":
    # 注意：运行此脚本需要遵守SEC rate limit
    # 建议：每秒不超过10次请求
    print("Note: This script respects SEC rate limits (max 10 requests/sec)")
    print("Running quick test with 10 cases...")

    main()  # 实际运行收集


"""
替代方案：使用现成的XBRL validation数据

XBRL-US Data Quality Committee发布已标注的错误案例：
https://xbrl.us/dqc/rules/

常见错误类型（可直接使用）：
1. DQC_0001: Calculation inconsistency
2. DQC_0002: Negative value where positive expected
3. DQC_0003: Missing dimension values
4. DQC_0004: Unit inconsistency

这些规则定义了真实XBRL中常见的错误模式。
"""
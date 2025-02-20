import redshift_connector
import os
import psycopg2
from dotenv import load_dotenv
import time
import re

load_dotenv(override=True)


def connect_to_redshift(
    disable_result_cache: bool = True,
) -> redshift_connector.Connection:
    conn = redshift_connector.connect(
        host=os.getenv("REDSHIFT_ENDPOINT"),
        port=os.getenv("REDSHIFT_PORT"),
        database=os.getenv("REDSHIFT_DATABASE"),
        user=os.getenv("REDSHIFT_USER"),
        password=os.getenv("REDSHIFT_PASSWORD"),
    )
    conn.rollback()
    conn.autocommit = True
    if disable_result_cache:
        cursor = conn.cursor()
        cursor.execute("SET enable_result_cache_for_session TO OFF;")
    return conn


def connect_to_redshift_serverless(disable_result_cache: bool = True) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=os.getenv("REDSHIFT_ENDPOINT"),
        port=os.getenv("REDSHIFT_PORT"),
        database="dev",
        user=os.getenv("REDSHIFT_USER"),
        password=os.getenv("REDSHIFT_PASSWORD"),
    )
    conn.rollback()
    conn.autocommit = True
    if disable_result_cache:
        cursor = conn.cursor()
        cursor.execute("SET enable_result_cache_for_session TO OFF;")
    return conn


def get_query_runtimes(query: str) -> list[float]:
    # TODO: Change to connect_to_redshift
    conn = connect_to_redshift_serverless()
    cursor = conn.cursor()
    runtimes = []
    for _ in range(10):
        start = time.perf_counter()
        cursor.execute(query)
        runtimes.append(time.perf_counter() - start)
    return runtimes


def get_explain_query_output(query: str) -> str:
    conn = connect_to_redshift_serverless()
    cursor = conn.cursor()
    cursor.execute(f"EXPLAIN {query}")
    return cursor.fetchall()


class PlanNode:
    def __init__(self, indent, text):
        self.indent = indent  # number of leading spaces
        self.text = text.strip()
        self.operator = None  # e.g. "XN Merge", "XN Seq Scan", etc.
        self.cost_range = None  # a string like "100..120"
        self.rows = None  # cardinality (number of rows)
        self.width = None  # width in bytes
        self.details = {}  # any extra details (like "Merge Key", "Filter", etc.)
        self.children = []  # child PlanNodes
        self.operatorId = None
        self.table = None


def parse_operator_line(line):
    seq_scan_pattern = re.compile(
        r"(?:->\s*)?(XN\s+\w+\s+\w+)\s+on\s+(\w+)(?:\s+\w+)?\s+\(cost=([\d\.]+)\.\.([\d\.]+)\s+rows=(\d+)\s+width=(\d+)\)"
    )
    match = seq_scan_pattern.search(line)
    if match:
        operator = match.group(1)  # e.g., "Seq Scan"
        table = match.group(2)  # e.g., "part"
        cost_low = float(match.group(3))  # e.g., 0.00
        cost_high = float(match.group(4))  # e.g., 1999.99
        cost_range = f"{cost_low}..{cost_high}"
        rows = int(match.group(5))  # e.g., 199999
        width = int(match.group(6))  # e.g., 47
        return operator, table, cost_range, rows, width

    pattern = re.compile(
        r"(?:->\s*)?(XN\s+\w+(?:\s+\w+)?(?:\s+on\s+\w+)?)\s*\(cost=([\d\.]+)\.\.([\d\.]+)\s+rows=(\d+)\s+width=(\d+)\)"
    )
    match = pattern.search(line)
    if match:
        operator = match.group(1).strip()
        # TODO: Temporarily restrict valid operators
        if operator in ("XN Hash Join", "XN Merge Join"):
            raise Exception(f"Unsupported operator: {operator}")
        if operator not in ("XN Sort", "XN HashAggregate"):
            return None
        cost_low = float(match.group(2))
        cost_high = float(match.group(3))
        cost_range = f"{cost_low}..{cost_high}"
        rows = int(match.group(4))
        width = int(match.group(5))
        return operator, None, cost_range, rows, width
    return None


def build_plan_tree(lines):
    """
    Build a tree of PlanNode objects based on indentation.
    Operator lines (those with cost, rows, etc.) become nodes.
    Other lines (e.g. "Merge Key: ..." or "Filter: ...") are attached as details
    to the most recent node at the current (or higher) indentation.
    """
    root = None
    stack = []  # will hold nodes by increasing indentation
    operator_counter = 0

    for tup in lines:
        line = tup[0]
        indent = len(line) - len(line.lstrip(" "))
        op_info = parse_operator_line(line)

        if op_info:
            operator, table, cost_range, rows, width = op_info
            node = PlanNode(indent, line)
            node.operator = operator
            node.table = table
            node.cost_range = cost_range
            node.rows = rows
            node.width = width
            node.operatorId = operator_counter
            operator_counter += 1
        else:
            # TODO: Move details to other key-value pairs inside dictionary
            detail_text = line.strip()
            if ":" in detail_text:
                parts = detail_text.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                # Attach the detail to the most recent node in the stack
                if stack:
                    stack[-1].details[key] = value
            else:
                # For lines such as "Send to leader", attach as a note.
                if stack:
                    stack[-1].details.setdefault("notes", []).append(detail_text)
            continue

        # Insert the new node into the tree based on indentation.
        if not stack:
            root = node
            stack.append(node)
        else:
            # If this node is more indented than the last node, it is a child.
            if indent > stack[-1].indent:
                stack[-1].children.append(node)
                stack.append(node)
            else:
                # Otherwise, pop until we find the proper parent.
                while stack and indent <= stack[-1].indent:
                    stack.pop()
                if stack:
                    stack[-1].children.append(node)
                else:
                    # In case nothing remains, treat as a new root.
                    root = node
                stack.append(node)
    return root, operator_counter


operator_mapping = {
    "XN Sort": "sort",
    "XN HashAggregate": "groupby",
    "XN Seq Scan": "tablescan",
}


def convert_plan_node(node):
    """
    Recursively converts a PlanNode into a dictionary resembling the target JSON plan.
    Some fields (like producedIUs, values, aggregates, etc.) are omitted or set to defaults.
    """
    result = {}
    logical_op = operator_mapping.get(node.operator, node.operator.lower() if node.operator else None)
    result["tablename"] = node.table
    result["restrictions"] = []
    result["residuals"] = []
    result["operator"] = logical_op
    # result["physicalOperator"] = logical_op
    result["cardinality"] = node.rows if node.rows is not None else 0
    result["producedIUs"] = []  # placeholder; would be filled in a complete implementation
    result["operatorId"] = node.operatorId
    result["analyzePlanId"] = node.operatorId
    # result["analyzePlanCardinality"] = node.rows if node.rows is not None else 0

    if node.details:
        result["details"] = node.details

    if node.children:
        if len(node.children) == 1:
            result["input"] = convert_plan_node(node.children[0])
        else:
            result["inputs"] = [convert_plan_node(child) for child in node.children]

    return result


def extract_pipelines(root, num_operators):
    # TODO: Non-hardcoded pipeline extraction
    if num_operators == 1:
        operators_lists = [[0]]
    elif num_operators == 2:
        operators_lists = [[0], [0, 1]]
    elif num_operators == 3:
        operators_lists = [[0], [0, 1], [1, 2]]
    else:
        raise Exception(f"Unsupported number of operators: {num_operators}")

    pipelines = []

    for operators_list in operators_lists:
        pipelines.append(
            {"start": 0, "stop": 0, "duration": 0, "parallelism": "multi-threaded", "operators": operators_list}
        )

    return pipelines


def get_query_plan(query: str) -> dict:
    explain_query_output = get_explain_query_output(query)
    plan_root, num_operators = build_plan_tree(explain_query_output)
    return {
        "plan": convert_plan_node(plan_root),
        "ius": [],
        "output": [],
        "type": "select",
        "query": True,
        "analyzePlanPipelines": extract_pipelines(plan_root, num_operators),
    }

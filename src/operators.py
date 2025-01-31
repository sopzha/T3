from dataclasses import dataclass
from string import digits
from typing import Optional

from src.util import AutoNumber


class OperatorType(AutoNumber):
    TableScan = ()
    InlineTable = ()
    PipelineBreakerScan = ()
    Temp = ()
    EarlyExecution = ()
    Select = ()
    Map = ()
    MultiWayJoin = ()
    HashJoin = ()
    IndexNLJoin = ()
    GroupJoin = ()
    GroupBy = ()
    Sort = ()
    SetOperation = ()
    Window = ()
    FileOutput = ()
    CsvWriter = ()
    AssertSingle = ()
    EarlyProbe = ()
    AnalyzePlan = ()

    def is_join_type(self):
        return self in {OperatorType.HashJoin, OperatorType.IndexNLJoin, OperatorType.GroupJoin}


@dataclass
class Expressions:
    join_filter_count: int = 0
    false_count: int = 0
    like_count: int = 0
    like_selectivity: float = 0.0
    compare_count: int = 0
    compare_selectivity: float = 0.0
    in_expression_count: int = 0
    in_expression_selectivity: float = 0.0
    between_count: int = 0
    between_selectivity: float = 0.0
    or_expression_count: int = 0
    or_selectivity: float = 0.0
    starts_with_count: int = 0
    starts_with_selectivity: float = 0.0


@dataclass
class Operator:
    type: OperatorType
    operator_name: str
    op_id: int

    # features
    output_cardinality: float
    input_cardinality: float
    right_input_cardinality: Optional[float]
    output_tuple_size: float
    expressions: Expressions

    parents: list["Operator"]
    input_op: Optional["Operator"]
    right_input_op: Optional["Operator"]
    json: dict

    def precedes(self, other_op: "Operator") -> int:
        """
        checks whether the operator is a transitive predecessor of other_op
        returns -1 if self precedes other, 0 if they are equal, and 1 if self does not precede
        """
        if self == other_op:
            return 0
        current_ancestors = other_op.parents.copy()
        while len(current_ancestors) > 0:
            current_ancestor = current_ancestors.pop()
            if current_ancestor == self:
                return -1
            current_ancestors.extend(current_ancestor.parents)
        return 1


def parse_operator_type(op: dict) -> OperatorType:
    name = op["operator"]
    name = name.rstrip(digits)
    if name == "join":
        if op["physicalOperator"] in ("hashjoin", "singletonjoin", "bnljoin"):
            return OperatorType.HashJoin
        elif op["physicalOperator"] == "indexnljoin":
            return OperatorType.IndexNLJoin

    name_map = {
        "fileoutput": OperatorType.FileOutput,
        "csvwriter": OperatorType.CsvWriter,
        "sort": OperatorType.Sort,
        "window": OperatorType.Window,
        "select": OperatorType.Select,
        "groupby": OperatorType.GroupBy,
        "groupjoin": OperatorType.GroupJoin,
        "multiwayjoin": OperatorType.MultiWayJoin,
        "tablescan": OperatorType.TableScan,
        "inlinetable": OperatorType.InlineTable,
        "map": OperatorType.Map,
        "earlyexecution": OperatorType.EarlyExecution,
        "pipelinebreakerscan": OperatorType.PipelineBreakerScan,
        "temp": OperatorType.Temp,
        "setoperation": OperatorType.SetOperation,
        "assertsingle": OperatorType.AssertSingle,
        "earlyprobe": OperatorType.EarlyProbe,
        "analyzeplan": OperatorType.AnalyzePlan,
    }
    assert name in name_map, f"{name} missing in operator name map {name_map}"
    return name_map[name]

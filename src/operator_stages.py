from dataclasses import dataclass
from typing import Optional

from src.operators import Operator
from src.operators import OperatorType
from src.util import AutoNumber


class OperatorStage(AutoNumber):
    Scan = ()
    Build = ()
    Probe = ()
    PassThrough = ()


@dataclass
class ExecutionPhase:
    operator: Operator
    stage: OperatorStage
    pipeline: "Pipeline"
    # After a union-all, the same operator will be executed within multiple pipelines
    # We will have to adjust all cardinality related numbers to the fraction of tuples in this operator that originate
    # From this pipeline
    fraction: float = 1.0

    def __str__(self):
        return f"{self.operator.type.name}:{self.stage.name}"

    def copy(self) -> "ExecutionPhase":
        return ExecutionPhase(self.operator, self.stage, self.pipeline, self.fraction)

    def _get_pipeline_start_op(self) -> "ExecutionPhase":
        return self.pipeline.operators[0]

    def _get_pipeline_scan_cardinality(self) -> float:
        return self.pipeline.get_pipeline_scan_cardinality()

    def get_input_percentage(self) -> float:
        if self._get_pipeline_scan_cardinality() == 0:
            return 0
        return self.operator.input_cardinality * self.fraction / self._get_pipeline_scan_cardinality()

    def get_output_percentage(self) -> float:
        if self._get_pipeline_scan_cardinality() == 0:
            return 0
        return self.operator.output_cardinality * self.fraction / self._get_pipeline_scan_cardinality()

    def get_right_percentage(self) -> Optional[float]:
        if self.operator.right_input_cardinality is None:
            return None
        if self._get_pipeline_scan_cardinality() == 0:
            return 0
        return self.operator.right_input_cardinality * self.fraction / self._get_pipeline_scan_cardinality()

    def get_input_cardinality(self) -> float:
        input_cardinality = self.operator.input_cardinality
        if self.stage == OperatorStage.Probe:
            return input_cardinality
        else:
            return input_cardinality * self.fraction

    def get_output_cardinality(self) -> float:
        output_cardinality = self.operator.output_cardinality
        if self.pipeline.operators[-1] == self:
            return output_cardinality
        else:
            return output_cardinality * self.fraction

    def get_right_input_cardinality(self) -> float:
        right_input_cardinality = (
            self.operator.right_input_cardinality if self.operator.right_input_cardinality is not None else 0
        )
        if self.stage == OperatorStage.Probe:
            return right_input_cardinality * self.fraction
        else:
            return right_input_cardinality


@dataclass
class Pipeline:
    operators: list[ExecutionPhase]
    operator_mapping: dict[Operator, ExecutionPhase]
    start: float
    stop: float

    def __init__(self, execution_phases: list[ExecutionPhase], start: float, stop: float):
        self.operators = execution_phases
        self.operator_mapping = {e.operator: e for e in self.operators}
        self.start = start
        self.stop = stop

    def get_execution_phase(self, op_id: int) -> ExecutionPhase:
        for op in self.operators:
            if op.operator.op_id == op_id:
                return op

    def get_pipeline_scan_cardinality(self) -> float:
        if len(self.operators) == 0:
            return 0
        if self.operators[0].operator.type in (OperatorType.GroupBy, OperatorType.Sort, OperatorType.Temp):
            return self.operators[0].operator.output_cardinality
        return self.operators[0].operator.input_cardinality

    def get_pipeline_sink_cardinality(self) -> float:
        if self.operators[-1].operator.type == OperatorType.GroupBy:
            return self.operators[-1].operator.input_cardinality
        else:
            return self.operators[-1].operator.output_cardinality


def get_operator_stage(op_index: int, op: Operator, pipeline_ops: list[Operator]) -> OperatorStage:
    if op.type in (
        OperatorType.TableScan,
        OperatorType.EarlyExecution,
        OperatorType.PipelineBreakerScan,
        OperatorType.InlineTable,
    ):
        return OperatorStage.Scan
    elif op.type in (OperatorType.Map, OperatorType.Select, OperatorType.AssertSingle, OperatorType.EarlyProbe):
        return OperatorStage.PassThrough
    elif op.type in (OperatorType.CsvWriter, OperatorType.FileOutput, OperatorType.Temp):
        return OperatorStage.Build
    elif op.type in (OperatorType.GroupBy, OperatorType.Sort, OperatorType.Window):
        if op_index == 0 and len(pipeline_ops) == 1:
            return OperatorStage.Scan
        elif op_index == len(pipeline_ops) - 1:
            return OperatorStage.Build
        elif op_index == 0:
            return OperatorStage.Scan
        assert False, f"{op.type.name} should be at begin or end of pipeline"
    elif op.type == OperatorType.HashJoin:
        assert op_index > 0, "join should not be at start of pipeline"
        input_op = pipeline_ops[op_index - 1]
        assert input_op.json == op.json["right"] or input_op.json == op.json["left"]
        if op_index != len(pipeline_ops) - 1 or input_op.json == op.json["right"]:
            return OperatorStage.Probe
        else:
            return OperatorStage.Build
    elif op.type == OperatorType.IndexNLJoin:
        assert op_index > 0
        input_op = pipeline_ops[op_index - 1]
        assert input_op.json == op.json["right"] or input_op.json == op.json["left"]
        if op_index != len(pipeline_ops) - 1:
            assert input_op.json == op.json["left"]
            return OperatorStage.Probe
        elif len(op.parents) == 0:
            # if this operator is the root of the tree, we might go left or right
            if input_op.json == op.json["left"]:
                return OperatorStage.Probe
            elif input_op.json == op.json["right"]:
                assert False, "build of an indexnl join should never be a pipeline"
            else:
                assert False, f"Error parsing indexNlJoin: unexpected input operator\n{op.json}"
        else:
            assert input_op.json == op.json["right"], (
                f"Error parsing index nl join\n" f"{input_op.json}\n{op.json['right']}"
            )
            return OperatorStage.Build
    elif op.type == OperatorType.SetOperation:
        if op_index == 0:
            # print(f"setop is scan ({op.json['operation']})")
            return OperatorStage.Scan
        elif op_index == len(pipeline_ops) - 1:
            # print("setop is build")
            return OperatorStage.Build
        else:
            # print("setop is pass-through")
            assert False, "our db does not show setoperations as pass-through"
    elif op.type == OperatorType.MultiWayJoin:
        if op_index == 0:
            return OperatorStage.Scan
        return OperatorStage.Build
    elif op.type == OperatorType.GroupJoin:
        if op_index == 0:
            return OperatorStage.Scan
        input_op = pipeline_ops[op_index - 1]
        assert input_op.json == op.json["right"] or input_op.json == op.json["left"]
        if input_op.json == op.json["right"]:
            return OperatorStage.Probe
        else:
            assert input_op.json == op.json["left"]
            return OperatorStage.Build
    assert False, f"unhandled operator: {op.type.name}"


def build_execution_phase(
    op_index: int, op: Operator, pipeline_ops: list[Operator], pipeline: Pipeline
) -> ExecutionPhase:
    stage = get_operator_stage(op_index, op, pipeline_ops)
    return ExecutionPhase(op, stage, pipeline)


def build_pipeline(pipeline_ops: list[Operator], start: float, stop: float) -> Pipeline:
    pipeline = Pipeline([], start, stop)
    execution_phases = [
        build_execution_phase(op_index, op, pipeline_ops, pipeline) for op_index, op in enumerate(pipeline_ops)
    ]
    pipeline.operators = execution_phases
    return pipeline

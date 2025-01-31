import random

import numpy as np

from src.database_manager import DatabaseManager
from src.query_generation.query_structures import IntermediateResult, get_random_columns, get_random_ir_columns
from src.query_generation.selections import SelectionFactory, sample_uniform_selection
from src.schemata import Schema
from src.util import AutoNumber


class AggregationFunction(AutoNumber):
    Sum = ()
    Min = ()
    Max = ()
    Avg = ()
    Rank = ()
    DenseRank = ()
    PercentRank = ()
    RowNumber = ()

    def to_string(self) -> str:
        if self == AggregationFunction.Sum:
            return "sum"
        elif self == AggregationFunction.Min:
            return "min"
        elif self == AggregationFunction.Max:
            return "max"
        elif self == AggregationFunction.Avg:
            return "avg"
        elif self == AggregationFunction.Rank:
            return "rank"
        elif self == AggregationFunction.DenseRank:
            return "dense_rank"
        elif self == AggregationFunction.PercentRank:
            return "percent_rank"
        elif self == AggregationFunction.RowNumber:
            return "row_number"
        else:
            assert False, f"unknown aggregation: {self._name_}"

    @staticmethod
    def sample() -> "AggregationFunction":
        return random.choice(list(AggregationFunction))


ZERO_ARGUMENT_AGG_FUNCTIONS = [
    AggregationFunction.Rank,
    AggregationFunction.DenseRank,
    AggregationFunction.PercentRank,
    AggregationFunction.RowNumber,
]


def get_preceding() -> str:
    # unbounded, N
    if np.random.random() < 0.5:
        return "UNBOUNDED PRECEDING"
    else:
        return f"{random.randint(0, 10)} PRECEDING"


def get_following() -> str:
    if np.random.random() < 0.5:
        return "CURRENT ROW"
    else:
        return f"{random.randint(0, 10)} FOLLOWING"


class WindowFunctionFactory:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.selection_factory = SelectionFactory(schema)

    @staticmethod
    def sample_window_function(intermediate_result: IntermediateResult) -> str:
        columns = [c for c in intermediate_result.columns if c.column.can_have_statistics()]
        agg_function = AggregationFunction.sample()
        agg_column = random.choice(columns)
        agg_column_str = agg_column.to_string()
        if agg_function in ZERO_ARGUMENT_AGG_FUNCTIONS:
            agg_column_str = ""
        partition_by_str = ""
        if np.random.random() < 0.7:
            partition_column = random.choice(columns)
            partition_by_str = f"PARTITION BY {partition_column.to_string()} "

        order_by_str = ""
        if np.random.random() < 0.7:
            order_by_column = random.choice(columns)
            order_by_str = f"ORDER BY {order_by_column.to_string()} "

        frame_str = ""
        if partition_by_str == "" or np.random.random() < 0.2:
            frame_str = f"ROWS BETWEEN {get_preceding()} AND {get_following()}"
        return (
            f"{agg_function.to_string()}({agg_column_str}) "
            f"OVER ({partition_by_str}{order_by_str}{frame_str}) window_column"
        )

    def get_subquery(self, statement_separator: str = " ") -> tuple[str, IntermediateResult]:
        selection = self.selection_factory.sample_selection()
        selection.sort = None

        binding_table = selection.column.table
        intermediate_result = binding_table.to_intermediate_result()
        window_function = WindowFunctionFactory.sample_window_function(intermediate_result)

        selected_column_objects = get_random_columns([binding_table])
        while not any([c.column.can_have_statistics() for c in selected_column_objects]):
            selected_column_objects = get_random_columns([binding_table])
        selected_columns = [f"{c.table.binding_name}.{c.column.name}" for c in selected_column_objects]
        selected_columns.append(window_function)
        select_string = ", ".join(selected_columns)
        table_string = f"{binding_table.table.table_name} {binding_table.binding_name}"
        return (
            f"SELECT {select_string}{statement_separator}"
            f"FROM {table_string}{statement_separator}"
            f"WHERE {selection.get_where_string()}{statement_separator}"
        ), IntermediateResult([binding_table], selected_column_objects)

    def get_query(self, statement_separator: str = " ") -> str:
        sub_query, ir = self.get_subquery(statement_separator)
        selection = sample_uniform_selection(ir)
        selected_columns = get_random_ir_columns(ir)
        selected_column_strs = [c.to_string() for c in selected_columns]
        selected_column_strs.append("window_column")
        select_str = ", ".join(selected_column_strs)
        order_by_str = ""
        if np.random.random() < 0.5:
            order_by_column = random.choice(ir.columns)
            order_by_str = f"ORDER BY {order_by_column.to_string()}"

        return (
            f"SELECT {select_str}{statement_separator}"
            f"FROM ({sub_query}) t0{statement_separator}"
            f"WHERE {selection.get_where_string()}{statement_separator}"
            f"{order_by_str}"
        )


def main():
    db = DatabaseManager.get_database("tpcdsSf1")
    factory = WindowFunctionFactory(db.schema)
    for _ in range(10):
        print(factory.get_query("\n"))
        print()


if __name__ == "__main__":
    main()

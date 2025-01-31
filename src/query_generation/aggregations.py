from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.database_manager import DatabaseManager
from src.query_generation.query_structures import BindingColumn, IntermediateResult, get_binding, BindingTable
from src.query_generation.selections import (
    sample_uniform_selection,
    can_have_selection,
    sample_sort,
    sample_complex_selection,
)
from src.schemata import Table, Column, Schema, Type
from src.util import AutoNumber, filter_unique_unhashable

MINIMAL_TABLE_SIZE = 1000


class AggregationFunction(AutoNumber):
    Count = ()
    Sum = ()
    Min = ()
    Max = ()
    Avg = ()

    def to_string(self) -> str:
        if self == AggregationFunction.Count:
            return "count"
        elif self == AggregationFunction.Sum:
            return "sum"
        elif self == AggregationFunction.Min:
            return "min"
        elif self == AggregationFunction.Max:
            return "max"
        elif self == AggregationFunction.Avg:
            return "avg"


def get_available_aggregation_functions(column: Column) -> list[AggregationFunction]:
    if column.type in (Type.Bigint, Type.Integer, Type.Decimal, Type.Double):
        return [AggregationFunction.Sum, AggregationFunction.Min, AggregationFunction.Max, AggregationFunction.Avg]
    elif column.type in (Type.Date, Type.Time):
        return [AggregationFunction.Min, AggregationFunction.Max]
    else:
        return []


@dataclass
class Aggregation:
    column: BindingColumn
    function: AggregationFunction

    def to_string(self) -> str:
        return f"{self.function.to_string()}({self.column.to_string()})"


@dataclass
class GroupBy:
    input: IntermediateResult
    aggregations: list[Aggregation]
    output: IntermediateResult


def get_eligible_tables(schema: Schema) -> list:
    eligible_tables = [t for t in schema.tables.values() if t.size is not None and t.size >= MINIMAL_TABLE_SIZE]
    eligible_tables = [
        t
        for t in eligible_tables
        if len(
            get_eligible_columns(
                IntermediateResult(
                    [BindingTable(t, "")], [BindingColumn(BindingTable(t, ""), c) for c in t.columns.values()]
                )
            )
        )
        > 0
    ]
    return eligible_tables


def sample_single_table(schema: Schema) -> IntermediateResult:
    eligible_tables = get_eligible_tables(schema)
    table = np.random.choice(eligible_tables)
    binding_table = BindingTable(table, get_binding(0))
    columns = [BindingColumn(binding_table, c) for c in table.columns.values()]
    return IntermediateResult([binding_table], columns)


def get_eligible_columns(input: IntermediateResult) -> list[BindingColumn]:
    eligible_columns = [c for c in input.columns if len(get_available_aggregation_functions(c.column)) > 0]
    return eligible_columns


def sample_aggregations(input: IntermediateResult, verbose: bool = False) -> Optional[list[Aggregation]]:
    if len(input.columns) <= 1:
        return None
    n_aggs = np.random.randint(1, min(7, len(input.columns)))
    result = []
    use_count_star = np.random.choice([False, True])
    eligible_columns = get_eligible_columns(input)
    if len(eligible_columns) == 0:
        if verbose:
            print(f"could not find column for {', '.join(str(x) for x in input.tables)}")
        return None
    for _ in range(n_aggs):
        column = np.random.choice(eligible_columns)
        agg_fun = np.random.choice(get_available_aggregation_functions(column.column))
        result.append(Aggregation(column, agg_fun))
    if use_count_star:
        result[0].function = AggregationFunction.Count
        n_aggs -= 1
    return result


def sample_group_by_columns(input: IntermediateResult) -> list[BindingColumn]:
    n_cols = np.random.randint(1, min(5, len(input.columns)))
    return np.random.choice(input.columns, size=n_cols)


def sample_group_by_query(
    schema: Schema,
    select_input: bool,
    statement_separator: str = "\n",
    pseudo_group_by: bool = False,
    complex_select: bool = False,
) -> Optional[str]:
    if len(get_eligible_tables(schema)) == 0:
        print(f"cannot do group-by for {schema.name}")
        return f"SELECT 1"
    in_table = sample_single_table(schema)
    aggregations = sample_aggregations(in_table)
    if aggregations is None:
        return sample_group_by_query(schema, select_input, statement_separator, pseudo_group_by, complex_select)
    aggregation_columns: list[BindingColumn] = filter_unique_unhashable([agg.column for agg in aggregations])
    select_string = ", ".join(agg.to_string() for agg in aggregations)
    from_string = ", ".join(f"{t.table.table_name} {t.binding_name}" for t in in_table.tables)
    group_string: str
    sort_clause = ""
    if pseudo_group_by:
        group_string = ", ".join(c.to_string() for c in aggregation_columns)
    else:
        group_columns = sample_group_by_columns(in_table)
        group_string = ", ".join(c.to_string() for c in group_columns)
        irs = IntermediateResult(in_table.tables, group_columns)
        sort = sample_sort(irs)
        if sort is not None:
            sort_clause = f"{statement_separator}{sort.get_sql()}"

    where_clause = ""
    if select_input:
        if not can_have_selection(in_table):
            return sample_group_by_query(schema, select_input, statement_separator, pseudo_group_by, complex_select)
        if complex_select:
            where_clause = f"WHERE {sample_complex_selection(in_table).get_where_string()}"
        else:
            selection = sample_uniform_selection(in_table)
            where_clause = f"WHERE {selection.get_where_string()}{statement_separator}"
    return (
        f"SELECT {select_string}{statement_separator}"
        f"FROM {from_string}{statement_separator}"
        f"{where_clause}"
        f"GROUP BY {group_string}{sort_clause}"
    )


def main():
    db = DatabaseManager.get_database("tpcdsSf1")
    for _ in range(100):
        print(sample_group_by_query(db.schema, True, complex_select=True))
        print()


if __name__ == "__main__":
    main()

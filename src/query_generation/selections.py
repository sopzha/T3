from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.database_manager import DatabaseManager
from src.query_generation.expressions import sample_expression
from src.query_generation.query_structures import (
    get_random_columns,
    BindingColumn,
    BindingTable,
    get_binding,
    IntermediateResult,
)
from src.schemata import Table, Column, Schema

SORT_PROBABILITY = 0.2
LIMIT_PROBABILITY = 0.5


@dataclass
class Sort:
    column: BindingColumn
    limit: Optional[int]
    asc: bool

    def get_sql(self):
        limit = ""
        if self.limit is not None:
            limit = f" LIMIT {self.limit}"
        return f"ORDER BY {self.column.to_string()} {'ASC' if self.asc else 'DESC'}{limit}"


def sample_sort(input: IntermediateResult) -> Optional[Sort]:
    sort = None
    if np.random.random() < SORT_PROBABILITY:
        col = np.random.choice(input.columns)
        limit = None
        if np.random.random() < LIMIT_PROBABILITY:
            limit = np.random.choice([10, 20, 100])
        asc = np.random.random() < 0.5
        sort = Sort(col, limit, asc)
    return sort


def get_threshold_string(column: BindingColumn, threshold: float):
    where_string = f"{column.table.binding_name}.{column.column.name} <= {threshold}"
    return where_string


@dataclass
class Selection:
    column: Optional[BindingColumn]
    where_string: str
    sort: Optional[Sort]

    def get_where_string(self):
        return self.where_string


class SelectionQuery:
    selection: Selection
    columns: list[BindingColumn]

    def __init__(self, selection: Selection):
        self.selection = selection
        self.table = self.selection.column.table
        self.columns = get_random_columns([self.table])

    def to_sql(self, statement_separator: str = "\n"):
        select_string = ", ".join(f"{c.table.binding_name}.{c.column.name}" for c in self.columns)
        table_string = f"{self.table.table.table_name} {self.table.binding_name}"
        where_string = self.selection.get_where_string()
        sort_clause = ""
        if self.selection.sort is not None:
            sort_clause = f"{statement_separator}{self.selection.sort.get_sql()}"
        return (
            f"SELECT {select_string}{statement_separator}"
            f"FROM {table_string}{statement_separator}"
            f"WHERE {where_string}{sort_clause};"
        )

    def to_intermediate_result(self) -> IntermediateResult:
        return IntermediateResult([self.table], self.columns)


class SelectionFactory:
    def __init__(self, schema: Schema):
        self.schema: Schema = schema
        self.table_columns: dict[str, tuple[list[Column], list[float]]] = {}
        self.tables = self.suitable_tables(self.schema)
        for t in self.tables:
            self.table_columns[t.table_name] = self.weighted_suitable_columns(t)
        self.table_weights: np.ndarray = np.array([t.size for t in self.tables], dtype=np.float32)
        self.table_weights /= np.sum(self.table_weights)
        # print(self.tables)
        # print(self.table_weights)

    @staticmethod
    def weighted_suitable_columns(table: Table) -> tuple[list[Column], list[float]]:
        columns = np.array([c for c in table.columns.values() if c.can_have_statistics()])
        column_sizes = np.array([c.distinct_count for c in columns])
        sum_distinct = np.sum(column_sizes)
        column_weights = column_sizes / sum_distinct
        sorted_indices = np.argsort(column_weights)[::-1]
        return columns[sorted_indices], column_weights[sorted_indices]

    @staticmethod
    def suitable_tables(schema: Schema) -> list[Table]:
        size_threshold = 1000
        tables = [t for t in schema.tables.values() if t.size >= size_threshold]
        tables = [t for t in tables if len(SelectionFactory.weighted_suitable_columns(t)[0]) > 0]
        return tables

    def sample_selection(self) -> Selection:
        table = np.random.choice(np.array(self.tables), p=self.table_weights)
        binding_table = BindingTable(table, get_binding(0))
        columns, weights = self.table_columns[table.table_name]
        column: Column = np.random.choice(columns, p=weights)
        binding_column = BindingColumn(binding_table, column)
        if np.random.random() < 0.5:
            selectivity = np.random.random()
        else:
            selectivity = 0.7 + np.random.random() * 0.3
        threshold = column.min_val + (column.max_val - column.min_val) * selectivity
        sort = sample_sort(binding_table.to_intermediate_result())
        where_str = get_threshold_string(binding_column, threshold)
        return Selection(binding_column, where_str, sort)

    def sample_selection_query(self) -> str:
        selection = self.sample_selection()
        return SelectionQuery(selection).to_sql()


def can_have_selection(input: IntermediateResult) -> bool:
    return len([c for c in input.columns if c.column.has_statistics()]) > 0


def sample_uniform_selection(input: IntermediateResult) -> Selection:
    columns = [c for c in input.columns if c.column.has_statistics()]
    column: BindingColumn = np.random.choice(columns)
    selectivity = np.random.random()
    threshold = column.column.min_val + (column.column.max_val - column.column.min_val) * selectivity
    sort = sample_sort(input)
    where_str = get_threshold_string(column, threshold)
    return Selection(column, where_str, sort)


def sample_complex_selection(input: IntermediateResult) -> Selection:
    n_expressions = np.random.geometric(p=0.4)
    where_str = " AND ".join(sample_expression(input) for _ in range(n_expressions))
    return Selection(None, where_str, None)


def sample_complex_selection_query(schema: Schema, statement_separator: str = " ") -> str:
    tables = [t for t in schema.tables.values() if t.size > 50]
    table = np.random.choice(np.array([t for t in tables]))
    binding_table = BindingTable(table, get_binding(0))
    select_string = ", ".join(f"{c.table.binding_name}.{c.column.name}" for c in get_random_columns([binding_table]))
    table_string = f"{binding_table.table.table_name} {binding_table.binding_name}"
    irs = binding_table.to_intermediate_result()
    where_str = sample_complex_selection(irs).get_where_string()
    return (
        f"SELECT {select_string}{statement_separator}" f"FROM {table_string}{statement_separator}" f"WHERE {where_str};"
    )


def main():
    db = DatabaseManager.get_database("tpcdsSf1")
    selection_factory = SelectionFactory(db.schema)
    for _ in range(100):
        print(selection_factory.sample_selection_query())
        print()
        print(sample_complex_selection_query(db.schema))


if __name__ == "__main__":
    main()

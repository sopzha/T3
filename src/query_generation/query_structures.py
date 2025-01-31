from dataclasses import dataclass
import random

from numpy.random import geometric

from src.schemata import Table, Column


@dataclass
class BindingTable:
    table: Table
    binding_name: str

    def __str__(self):
        return f"{self.binding_name}.{self.table.table_name}"

    def to_intermediate_result(self) -> "IntermediateResult":
        columns = [BindingColumn(self, c) for c in self.table.columns.values()]
        return IntermediateResult([self], columns)


def get_binding(n: int) -> str:
    return f"t{n}"


@dataclass
class BindingColumn:
    table: BindingTable
    column: Column

    def to_string(self) -> str:
        return f"{self.table.binding_name}.{self.column.name}"


@dataclass
class IntermediateResult:
    tables: list[BindingTable]
    columns: list[BindingColumn]


def get_random_columns(tables: list[BindingTable]) -> list[BindingColumn]:
    flattened_columns = [BindingColumn(t, c) for t in tables for c in t.table.columns.values()]
    n_columns = min(len(flattened_columns), geometric(p=0.2), 20)
    columns = random.sample(flattened_columns, n_columns)
    columns.sort(key=lambda x: x.table.binding_name)
    return columns


def get_random_ir_columns(ir: IntermediateResult) -> list[BindingColumn]:
    n_columns = min(len(ir.columns), geometric(p=0.2), 20)
    columns = random.sample(ir.columns, n_columns)
    columns.sort(key=lambda x: x.table.binding_name)
    return columns

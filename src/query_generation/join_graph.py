import operator
from dataclasses import dataclass
import random
from functools import reduce
from typing import Optional

from src.database_manager import DatabaseManager
from src.query_generation.query_structures import (
    BindingTable,
    get_random_columns,
    get_binding,
    IntermediateResult,
    BindingColumn,
)
from src.query_generation.selections import (
    Selection,
    sample_uniform_selection,
    can_have_selection,
    Sort,
    sample_sort,
    sample_complex_selection,
)
from src.schemata import Schema, Table

from numpy.random import geometric


@dataclass
class Join:
    table1: BindingTable
    table2: BindingTable
    column1: str
    column2: str


@dataclass
class JoinGraph:
    tables: list[BindingTable]
    joins: list[Join]
    selections: list[Selection]
    sort: Optional[Sort]

    def sample_sort(self):
        assert self.sort is None
        self.sort = sample_sort(self.get_intermediate_result())

    def get_intermediate_result(self) -> IntermediateResult:
        columns = []
        for table in self.tables:
            for column in table.table.columns.values():
                columns.append(BindingColumn(table, column))
        return IntermediateResult(self.tables, columns)

    def get_where_conditions(self):
        return [
            f"{join.table1.binding_name}.{join.column1} = {join.table2.binding_name}.{join.column2}"
            for join in self.joins
        ] + [selection.get_where_string() for selection in self.selections]


def get_possible_joins(tables: list[BindingTable], schema: Schema, new_binding: str) -> list[Join]:
    """
    The second table will always be the new table
    """
    result = []
    for t in tables:
        for c, other_list in schema.join_columns[t.table.table_name].items():
            if t.table.columns[c].distinct_count == 0:
                continue
            for other_t_name, other_c in other_list:
                other_t = schema.tables[other_t_name]
                if other_t.columns[other_c].distinct_count == 0:
                    continue
                result.append(Join(t, BindingTable(other_t, new_binding), c, other_c))
    return result


def get_output_size(join: Join) -> float:
    column1 = join.table1.table.columns[join.column1]
    if column1.distinct_count is None or column1.distinct_count == 0:
        print(column1.name)
    t1_size = join.table1.table.size
    c1_avg_duplicates = t1_size / column1.distinct_count
    assert c1_avg_duplicates >= 1.0
    column1_key = c1_avg_duplicates < 1.01

    column2 = join.table2.table.columns[join.column2]
    if column2.distinct_count is None or column2.distinct_count == 0:
        print(column2.name)
    t2_size = join.table2.table.size
    c2_avg_duplicates = t2_size / column2.distinct_count
    assert c2_avg_duplicates >= 1.0
    column2_key = c2_avg_duplicates < 1.01

    if column1_key and column2_key:
        return min(t1_size, t2_size)
    elif column1_key:
        return t2_size
    elif column2_key:
        return t1_size
    else:
        return c1_avg_duplicates * c2_avg_duplicates * min(t1_size, t2_size)


def get_selectivity(join: Join) -> float:
    return get_output_size(join) / (join.table1.table.size * join.table2.table.size)


def get_cardinality(tables: list[BindingTable], joins: list[Join]) -> float:
    cross_product_size = reduce(operator.mul, (t.table.size for t in tables), 1.0)
    selectivity_product = reduce(operator.mul, (get_selectivity(j) for j in joins), 1.0)
    result = cross_product_size * selectivity_product
    return result


def cardinality_factor(join: Join) -> float:
    """
    the factor of which table 2 of the join will increase the size on the previous result
    """
    return get_selectivity(join) * join.table2.table.size


def sample_join_graph(
    schema: Schema, use_selections: bool, use_complex_selections: bool = False, cardinality_limit=1e7, tries=20
) -> JoinGraph:
    if tries < 1:
        raise RuntimeError(f"could not sample join graph for schema {schema.name} after several tries")
    tables = list(schema.tables.values())
    start_table: Table = random.choice(tables)

    n_joins = geometric(p=0.4)
    n_joins = min(n_joins, 5)

    joins: list[Join] = []
    included_tables: list[BindingTable] = [BindingTable(start_table, get_binding(0))]
    selections = []
    for i in range(n_joins):
        cardinality = get_cardinality(included_tables, joins)
        possible_joins = get_possible_joins(included_tables, schema, get_binding(i + 1))
        possible_joins = [j for j in possible_joins if cardinality * cardinality_factor(j) <= cardinality_limit]
        if len(possible_joins) == 0:
            return sample_join_graph(schema, use_selections, tries=tries - 1)
        new_join = random.choice(possible_joins)
        included_tables.append(new_join.table2)
        joins.append(new_join)
        assert get_cardinality(included_tables, joins) <= cardinality_limit * 1.01, (
            f"exceeded cardinality limit ({get_cardinality(included_tables, joins)})\n"
            f"{joins_to_sql(JoinGraph(included_tables, joins, selections, None), schema)}"
        )
    if use_selections:
        for t in included_tables:
            irs = t.to_intermediate_result()
            if use_complex_selections:
                selections.append(sample_complex_selection(irs))
            else:
                if can_have_selection(irs):
                    selections.append(sample_uniform_selection(irs))
    sort = None
    result = JoinGraph(included_tables, joins, selections, sort)
    result.sample_sort()
    return result


def joins_to_sql(join_graph: JoinGraph, schema: Schema, statement_separator: str = "\n"):
    table_string = ", ".join(f"{schema.name}.{t.table.table_name} {t.binding_name}" for t in join_graph.tables)

    projections = get_random_columns(join_graph.tables)
    select_string = ", ".join(f"{c.table.binding_name}.{c.column.name}" for c in projections)

    where_conditions = join_graph.get_where_conditions()
    where_string = " AND ".join(where_conditions)

    sort_clause = ""
    if join_graph.sort is not None:
        sort_clause = f"{statement_separator}{join_graph.sort.get_sql()}"
    return (
        f"SELECT {select_string}{statement_separator}"
        f"FROM {table_string}{statement_separator}"
        f"WHERE {where_string}{sort_clause};"
    )


def generate_join_query(schema: Schema, use_selections: bool, use_complex_selections: bool) -> str:
    join_graph = sample_join_graph(schema, use_selections, use_complex_selections)
    return joins_to_sql(join_graph, schema, " ")


def generate_join_queries(schema: Schema, use_selections: bool, n: int, use_complex_selections: bool) -> list[str]:
    return [generate_join_query(schema, use_selections, use_complex_selections) for _ in range(n)]


def main():
    db = DatabaseManager.get_database("tpchSf1")
    for q in generate_join_queries(db.schema, True, 30, True):
        print(q)


if __name__ == "__main__":
    main()

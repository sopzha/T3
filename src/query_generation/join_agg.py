from src.database_manager import DatabaseManager
from src.query_generation.aggregations import sample_aggregations, sample_group_by_columns
from src.query_generation.join_graph import sample_join_graph, JoinGraph
from src.query_generation.query_structures import get_random_columns, BindingColumn, IntermediateResult
from src.query_generation.selections import sample_sort
from src.schemata import Schema


def join_aggregations_to_sql(join_graph: JoinGraph, schema: Schema, simple: bool, statement_separator: str = "\n"):
    irs = join_graph.get_intermediate_result()
    aggregations = sample_aggregations(irs)
    if aggregations is None:
        return None
    # aggregation_columns: list[BindingColumn] = filter_unique_unhashable([agg.column for agg in aggregations])

    table_string = ", ".join(f"{schema.name}.{t.table.table_name} {t.binding_name}" for t in join_graph.tables)

    select_string = ", ".join(agg.to_string() for agg in aggregations)

    where_string = " AND ".join(join_graph.get_where_conditions())

    # group to a single element
    if simple:
        return (
            f"SELECT {select_string}{statement_separator}"
            f"FROM {table_string}{statement_separator}"
            f"WHERE {where_string}{statement_separator}"
        )

    group_by_columns = sample_group_by_columns(irs)
    group_string = ", ".join(c.to_string() for c in group_by_columns)
    sort_clause = ""
    irs = IntermediateResult(irs.tables, group_by_columns)
    sort = sample_sort(irs)
    if sort is not None:
        sort_clause = f"{statement_separator}{sort.get_sql()}"

    return (
        f"SELECT {select_string}{statement_separator}"
        f"FROM {table_string}{statement_separator}"
        f"WHERE {where_string}{statement_separator}"
        f"GROUP BY {group_string}{sort_clause};"
    )


def generate_join_agg_query(schema: Schema, use_selections: bool, use_complex_selections: bool) -> str:
    while True:
        join_graph = sample_join_graph(schema, use_selections, use_complex_selections)
        result = join_aggregations_to_sql(join_graph, schema, False, " ")
        if result is not None:
            return result


def generate_join_simple_agg_query(schema: Schema, use_selections: bool, use_complex_selection: bool) -> str:
    while True:
        join_graph = sample_join_graph(schema, use_selections, use_complex_selection)
        result = join_aggregations_to_sql(join_graph, schema, True, " ")
        if result is not None:
            return result


def main():
    db = DatabaseManager.get_database("tpcdsSf1")
    schema = db.schema
    for _ in range(100):
        query = generate_join_simple_agg_query(schema, True, True)
        print(query)


if __name__ == "__main__":
    main()

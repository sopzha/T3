from src.train import optimize_all
from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache, statistics_with_error_function, get_errors
from src.figures.infra import write_latex_file, set_figure_path
from src.metrics import q_error
from src.optimizer import QueryCategory


def latex_accuracy_table(reports: dict) -> str:
    columns = ["p50", "p90", "Avg"]
    n_columns = len(columns)
    result = f"""\\begin{{table}}
  \\centering
  \\begin{{tabular}}{{r|{" ".join("r" for _ in range(n_columns))}}}
    Queries & {" & ".join(columns)} \\\\
    \\hline
"""
    for n, r in reports.items():
        current_line = f"    {n} & " + " & ".join(f"{r[c]:.2f}" for c in columns) + " \\\\\n"
        result += current_line
    result += """  \\end{tabular}
  \\caption{Accuracy of T3 measured in q-error.
    T3 is trained on 20 training database instances and has never seen any information about TPC-DS data or queries.
    }
  \\label{tbl_tpcds_acc}
\\end{table}\n"""
    return result


def write_accuracy_table(estimation_cache: QueryEstimationCache):
    predicted_cardinalities = False
    data = [
        (
            "Train Queries",
            DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), predicted_cardinalities),
        ),
        (
            "All TPC-DS Test Queries",
            DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), predicted_cardinalities),
        ),
        (
            "TPC-DS Benchmark Queries",
            DataCollector.collect_benchmarks(
                DatabaseManager.get_test_databases(), predicted_cardinalities, query_category=[QueryCategory.fixed]
            ),
        ),
        (
            "TPC-DS sf 100 Test Queries",
            DataCollector.collect_benchmarks([DatabaseManager.get_database("tpcdsSf100")], predicted_cardinalities),
        ),
        (
            "TPC-DS sf 100 Benchmark Queries",
            DataCollector.collect_benchmarks(
                [DatabaseManager.get_database("tpcdsSf100")],
                predicted_cardinalities,
                query_category=[QueryCategory.fixed],
            ),
        ),
    ]
    report = {}
    for n, queries in data:
        estimates = [estimation_cache.queries[q.name].estimated_time for q in queries]
        q_errors = get_errors(queries, estimates, q_error, None, None)
        report[n] = statistics_with_error_function(q_errors, n, "Q-Error", False)

    write_latex_file(
        latex_accuracy_table(report),
        "tbl_tpcds_acc",
    )


def main():
    set_figure_path("./figure_output")
    model = optimize_all()
    estimation_cache = QueryEstimationCache(model, False)
    write_accuracy_table(estimation_cache)


if __name__ == "__main__":
    main()

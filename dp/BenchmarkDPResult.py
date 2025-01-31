from pathlib import Path

import numpy as np

from src.util import get_lines
from src.benchmark import Benchmarker
from src.database_manager import DatabaseManager
from src.figures.infra import write_latex_file


def benchmark_dp_queries(verbose: bool = False):
    model_queries, cout_queries, query_names = (
        get_lines(f) for f in (Path("dp/model_plans.sql"), Path("dp/cout_plans.sql"), Path("dp/query_names.txt"))
    )
    q33_inidces = [i for i, n in enumerate(query_names) if n.startswith("33")]
    model_queries, cout_queries, query_names = (
        [x for i, x in enumerate(l) if i not in q33_inidces] for l in (model_queries, cout_queries, query_names)
    )
    address = "http://localhost:8000"
    benchmarker = Benchmarker(address)
    db = DatabaseManager.get_database("job")
    names = []
    cout_times = []
    model_times = []
    native_db_times = []
    n_runs = 10
    for n, q_model, q_cout in zip(query_names, model_queries, cout_queries):
        cout_time = np.min(benchmarker.n_raw_runs(db, q_cout, n_runs))
        model_time = np.min(benchmarker.n_raw_runs(db, q_model, n_runs))
        cout_times.append(cout_time)
        model_times.append(model_time)
        names.append(n[:-1])
        if verbose and (model_time < cout_time * 0.95 or model_time > cout_time * 1.05):
            print(f"name: {n[:-1]} cout_time = {cout_time} model_time = {model_time} speedup {cout_time / model_time}")
            print(f"cout_query:\n{q_cout.strip()}\nmodel_query:\n{q_model.strip()}\n")

    n_model_better = 0
    n_cout_better = 0
    for n, t_cout, t_model in zip(names, cout_times, model_times):
        if t_model < t_cout * 0.95 or t_model > t_cout * 1.05:
            if t_model < t_cout:
                n_model_better += 1
            else:
                n_cout_better += 1
    # print(f"Our model is better for {n_model_better} queries, cout is better for {n_cout_better} queries")

    for q in benchmarker.get_fixed_queries(db).values():
        query = q()
        native_db_time = np.min(benchmarker.n_raw_runs(db, query, n_runs))
        native_db_times.append(native_db_time)
    # print(
    #     f"sum cout {sum(cout_times):.3f}, sum model {sum(model_times):.3f}, speedup {sum(cout_times) / sum(model_times):.4f}"
    # )
    # print(f"sum native db: {sum(native_db_times):.3f}")
    write_latex_file(
        f"""\\begin{{tabular}}{{r|r}}
Model & Execution Time\\\\
    \\hline
$\\text{{C}}_{{\\text{{out}}}}$ & {sum(cout_times):.3f}s\\\\
    T3 & {sum(model_times):.3f}s\\\\
    Native DB & {sum(native_db_times):.3f}s\\\\
    \\end{{tabular}}
""",
        "tbl_join_order_execution_times",
    )


if __name__ == "__main__":
    benchmark_dp_queries()

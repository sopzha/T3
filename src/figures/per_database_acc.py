import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database import Database
from src.database_manager import DatabaseManager
from src.figures.infra import get_figure_path, setup_matplotlib_latex_font, get_figure_format
from src.metrics import q_error
from src.model import Model
from src.optimizer import optimize_per_tuple_tree_model


def optimize_without_db() -> list[tuple[Model, list[Database], str]]:
    result = []
    special_databases = DatabaseManager.get_databases(
        ["tpcdsSf1", "tpcdsSf10", "tpcdsSf100", "tpchSf1", "tpchSf10", "tpchSf100"]
    )
    for db, name in (
        (DatabaseManager.get_databases(["tpchSf1", "tpchSf10", "tpchSf100"]), "TPC-H"),
        (DatabaseManager.get_databases(["tpcdsSf1", "tpcdsSf10", "tpcdsSf100"]), "TPC-DS"),
    ):
        current_train_databases = [x for x in DatabaseManager.get_all_databases() if x not in db]
        benchmarks = DataCollector.collect_benchmarks(current_train_databases, False)
        result.append((optimize_per_tuple_tree_model(benchmarks), db, name))

    dbs = DatabaseManager.get_all_databases()
    dbs.sort(key=lambda x: x.schema.name)
    for db in dbs:
        if db in special_databases:
            continue
        current_train_databases = [x for x in DatabaseManager.get_all_databases() if x != db]
        benchmarks = DataCollector.collect_benchmarks(current_train_databases, False)
        result.append((optimize_per_tuple_tree_model(benchmarks), [db], db.schema.name.capitalize()))
    return result


def eval_dbs(dbs: list[tuple[Model, list[Database], str]]):
    setup_matplotlib_latex_font()

    names = []
    p50s = []
    p90s = []
    avgs = []
    for model, db, name in dbs:
        names.append(name)
        benchmarks = DataCollector.collect_benchmarks(db, False)
        runtimes = [b.get_total_runtime() for b in benchmarks]
        estimates = [model.estimate_runtime(b) for b in benchmarks]
        q_errors = [q_error(e, r) for e, r in zip(estimates, runtimes)]

        p50s.append(np.quantile(q_errors, 0.5))
        p90s.append(np.quantile(q_errors, 0.9))
        avgs.append(np.average(q_errors))

    plt.figure(figsize=(6, 2.5))

    bar_width = 0.3
    x = np.arange(len(names))
    plt.bar(x - bar_width, p50s, width=bar_width, label="p50", edgecolor="black")
    plt.bar(x, p90s, width=bar_width, label="p90", edgecolor="black", hatch="//")
    plt.bar(x + bar_width, avgs, width=bar_width, label="Avg", edgecolor="black", hatch="\\\\")
    plt.ylim(1, None)

    # naming
    plt.ylabel("Q-Error")
    plt.xticks(x, names, rotation=37.5, ha="right", fontsize=10)

    plt.savefig(f"{get_figure_path()}/per_database_acc.{get_figure_format()}", bbox_inches="tight")


def create_per_db_figure():
    dbs = optimize_without_db()
    eval_dbs(dbs)


def main():
    create_per_db_figure()


if __name__ == "__main__":
    main()

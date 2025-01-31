import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache
from src.figures.infra import setup_matplotlib_latex_font, get_figure_path, get_figure_format
from src.metrics import q_error
from src.optimizer import QueryCategory
from src.train import optimize_all


def get_error_by_query_hist(estimation_cache: QueryEstimationCache):
    setup_matplotlib_latex_font()

    dbs = DatabaseManager.get_test_databases()
    benchmarks = DataCollector.collect_benchmarks(dbs, False)

    categories = {c: [] for c in QueryCategory}
    for b in benchmarks:
        if b.query_category not in categories:
            categories[b.query_category] = []
        categories[b.query_category].append(b)

    names = []
    p50s = []
    p90s = []
    avgs = []
    for category, benchmarks in categories.items():
        names.append(category.get_name())
        runtimes = [b.get_total_runtime() for b in benchmarks]
        estimates = [estimation_cache.queries[q.name].estimated_time for q in benchmarks]
        q_errors = [q_error(e, r) for e, r in zip(estimates, runtimes)]
        p50s.append(np.quantile(q_errors, 0.5))
        p90s.append(np.quantile(q_errors, 0.9))
        avgs.append(np.average(q_errors))

    plt.figure(figsize=(6, 2.5))

    bar_width = 0.3
    x = np.arange(len(names))
    plt.bar(x - bar_width, p50s, width=bar_width, label="p50", edgecolor="black", hatch="//")
    plt.bar(x, p90s, width=bar_width, label="p90", edgecolor="black", hatch="//")
    plt.bar(x + bar_width, avgs, width=bar_width, label="Avg", edgecolor="black", hatch="\\\\")
    plt.ylim(1, None)

    # naming
    plt.xlabel("Query Type")
    plt.ylabel("Q-Error")
    plt.xticks(x, names, rotation=37.5, ha="right", fontsize=10)

    plt.savefig(f"{get_figure_path()}/error_by_category.{get_figure_format()}", bbox_inches="tight")


def main():
    model = optimize_all()
    estimation_cache = QueryEstimationCache(model, False)
    get_error_by_query_hist(estimation_cache)


if __name__ == "__main__":
    main()

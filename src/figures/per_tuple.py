import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.figures.infra import get_figure_path, setup_matplotlib_latex_font, get_hex_colors, get_figure_format
from src.metrics import q_error
from src.optimizer import (
    optimize_per_tuple_tree_model,
    optimize_tree_model,
    optimize_flat_tree_model,
)


def get_test_numbers(model, benchmarks, runtimes):
    estimates = [model.estimate_runtime(b) for b in benchmarks]
    q_errors = [q_error(e, r) for e, r in zip(estimates, runtimes)]
    return {"Avg": np.average(q_errors), "p50": np.quantile(q_errors, 0.5), "p90": np.quantile(q_errors, 0.9)}


def benchmark_size_reports() -> list[tuple[str, dict]]:
    result = []
    benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), False)
    eval_queries = DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), False)
    eval_runtimes = [b.get_total_runtime() for b in eval_queries]

    for model, name in (
        (optimize_per_tuple_tree_model(benchmarks), "T3 (Per Tuple)"),
        (optimize_tree_model(benchmarks), "Per Pipeline Vector"),
        (optimize_flat_tree_model(benchmarks), "Flat Query Vector"),
    ):
        report = get_test_numbers(model, eval_queries, eval_runtimes)
        result.append((name, report))

    return result


def eval_benchmarks(results):
    setup_matplotlib_latex_font()

    names = [n for n, _ in results]
    reports = [r for _, r in results]

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))
    x = np.arange(len(names))
    n = 0
    for ax, metric in ((axs[0], "p50"), (axs[1], "p90"), (axs[2], "Avg")):
        values = [r[metric] for r in reports]
        bars = ax.bar(
            names,
            values,
            edgecolor="black",
            color=get_hex_colors(["my_blue", "my_red", "my_green"]),
            hatch=["", "//", "\\\\"],
        )

        ax.set_ylim(1, None)

        # Plot numbers on bars
        for bar in bars:
            y_val = bar.get_height()
            ax.annotate(
                f"{y_val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, y_val),
                xycoords="data",
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # naming
        if n == 1:
            ax.set_xlabel("")
        ax.set_ylabel(f"{metric} Q-Error")
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)

        n += 1

    plt.savefig(f"{get_figure_path()}/ablation_study.{get_figure_format()}", bbox_inches="tight")


def per_tuple_prediction_figure():
    results = benchmark_size_reports()
    eval_benchmarks(results)


def main():
    per_tuple_prediction_figure()


if __name__ == "__main__":
    main()

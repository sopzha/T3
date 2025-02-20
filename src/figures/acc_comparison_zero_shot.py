import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.figures.infra import get_figure_path, setup_matplotlib_latex_font, get_hex_colors, get_figure_format
from src.metrics import q_error
from src.optimizer import optimize_per_tuple_tree_model, QueryCategory


def get_zero_shot_exact_numbers():
    return {"Avg": 1.5979, "p50": 1.1483, "p90": 2.4392}


def get_zero_shot_pred_numbers():
    return {"Avg": 1.7568, "p50": 1.3359, "p90": 2.9053}


def get_test_numbers(predicted_cardinalities: bool):
    """
    For this comparison we use estimated cardinalities!
    :param estimation_cache:
    :return:
    """
    job_db = DatabaseManager.get_database("job")
    train_databases = [x for x in DatabaseManager.get_all_databases() if x != job_db]
    assert len(train_databases) == len(DatabaseManager.get_all_databases()) - 1
    benchmarks = DataCollector.collect_benchmarks(train_databases, predicted_cardinalities)
    model = optimize_per_tuple_tree_model(benchmarks)

    benchmarks = DataCollector.collect_benchmarks(
        [job_db], predicted_cardinalities, query_category=[QueryCategory.fixed]
    )
    runtimes = [b.get_total_runtime() for b in benchmarks]
    estimates = [model.estimate_runtime(q) for q in benchmarks]
    q_errors = [q_error(e, r) for e, r in zip(estimates, runtimes)]
    return {"Avg": np.average(q_errors), "p50": np.quantile(q_errors, 0.5), "p90": np.quantile(q_errors, 0.9)}


def comparison_zero_shot_plot():
    setup_matplotlib_latex_font()
    zero_shot = get_zero_shot_exact_numbers()
    t3 = get_test_numbers(False)
    names = ["T3", "Zero Shot"]
    reports = [t3, zero_shot]

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))
    x = np.arange(len(names))
    n = 0
    for ax, metric in ((axs[0], "p50"), (axs[1], "p90"), (axs[2], "Avg")):
        values = [r[metric] for r in reports]
        bars = ax.bar(
            names,
            values,
            edgecolor="black",
            color=get_hex_colors(["my_blue", "my_red"]),
            hatch=["", "//"],
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
            ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric} Q-Error")
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)

        n += 1

    plt.savefig(f"{get_figure_path()}/acc_comparison_zero_shot.{get_figure_format()}", bbox_inches="tight")


def main():
    comparison_zero_shot_plot()


if __name__ == "__main__":
    main()

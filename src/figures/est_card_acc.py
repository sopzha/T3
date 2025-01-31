import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache
from src.figures.infra import setup_matplotlib_latex_font, get_figure_path, get_hex_colors, get_figure_format
from src.metrics import q_error
from src.model import Model
from src.optimizer import optimize_per_tuple_tree_model
from src.train import optimize_all


def get_card_models() -> list[Model]:
    """
    get two models:
    1. regular model trained with perfect cardinalities
    2. model trained with estimated cardinalities
    """
    dbs = DatabaseManager.get_train_databases()
    qs = [DataCollector.collect_benchmarks(dbs, False), DataCollector.collect_benchmarks(dbs, True)]
    result = []
    for q in qs:
        result.append(optimize_per_tuple_tree_model(q))
    return result


def eval_card_est(estimation_caches: list[QueryEstimationCache]):
    setup_matplotlib_latex_font()

    names = [
        "perfect train\nperfect test",
        "perfect train\nestimated test",
        "estimated train\nestimated test",
    ]
    p50s = []
    p90s = []
    avgs = []
    benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), False)
    for estimation_cache in estimation_caches:
        runtimes = [b.get_total_runtime() for b in benchmarks]
        estimates = [estimation_cache.queries[b.name].estimated_time for b in benchmarks]
        q_errors = [q_error(e, r) for e, r in zip(estimates, runtimes)]

        p50s.append(np.quantile(q_errors, 0.5))
        p90s.append(np.quantile(q_errors, 0.9))
        avgs.append(np.average(q_errors))

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))
    x = np.arange(len(names))
    n = 0
    for ax, values, metric in ((axs[0], p50s, "p50"), (axs[1], p90s, "p90"), (axs[2], avgs, "Avg")):
        # plt.subplot(1, 2, col)
        bars = ax.bar(
            names,
            values,
            edgecolor="black",
            color=get_hex_colors(["my_blue", "my_red", "my_green"]),
            hatch=["", "//", "\\\\"],
        )

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
        ax.set_ylim(1, None)

        # naming
        if n == 1:
            ax.set_xlabel("Cardinalities")
        ax.set_ylabel(f"{metric} Q-Error")
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
        n += 1

    plt.savefig(f"{get_figure_path()}/card_est_acc.{get_figure_format()}", bbox_inches="tight")


def main():
    exact_model = optimize_all(False)
    pred_model = optimize_all(True)
    print("Evaluating models")
    exact_exact_eval = QueryEstimationCache(exact_model, False)
    exact_pred_eval = QueryEstimationCache(exact_model, True)
    pred_pred_eval = QueryEstimationCache(pred_model, True)
    estimation_caches = [exact_exact_eval, exact_pred_eval, pred_pred_eval]
    eval_card_est(estimation_caches)


if __name__ == "__main__":
    main()

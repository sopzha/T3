import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache
from src.figures.acc_comparison_zero_shot import get_zero_shot_pred_numbers
from src.figures.infra import get_figure_path, setup_matplotlib_latex_font, get_hex_colors, get_figure_format
from src.metrics import q_error
from src.train import optimize_all


def get_auto_wlm_pred_numbers():
    return {"Avg": 171.8, "p50": 4.08, "p90": 135.7}


def get_stage_pred_numbers():
    return {"Avg": 54.57, "p50": 1.6, "p90": 19.0}


def get_competitor_numbers():
    # the following results are taken from https://dl.acm.org/doi/pdf/10.1145/3626246.3653391
    auto_wlm = get_auto_wlm_pred_numbers()
    stage = get_stage_pred_numbers()
    # the following results are reproduced using https://github.com/DataManagementLab/zero-shot-cost-estimation
    # we are using job-full with estimated cardinalities
    zero_shot = get_zero_shot_pred_numbers()
    return auto_wlm, stage, zero_shot


def get_test_numbers(estimation_cache: QueryEstimationCache):
    """
    For this comparison we use estimated cardinalities!
    Also we only use the dataset TPC-DS sf100 but all queries
    :param estimation_cache:
    :return:
    """

    benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_databases(["tpcdsSf100"]), True)
    runtimes = [b.get_total_runtime() for b in benchmarks]
    estimates = [estimation_cache.queries[q.name].estimated_time for q in benchmarks]
    q_errors = [q_error(e, r) for e, r in zip(estimates, runtimes)]
    return {"Avg": np.average(q_errors), "p50": np.quantile(q_errors, 0.5), "p90": np.quantile(q_errors, 0.9)}


def comparison_plot(estimation_cache: QueryEstimationCache):
    setup_matplotlib_latex_font()
    auto_wlm, stage, zero_shot = get_competitor_numbers()
    t3 = get_test_numbers(estimation_cache)
    names = ["T3", "Zero Shot", "Stage", "AutoWLM"]
    reports = [t3, zero_shot, stage, auto_wlm]

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))
    x = np.arange(len(names))
    n = 0
    for ax, metric in ((axs[0], "p50"), (axs[1], "p90"), (axs[2], "Avg")):
        values = [r[metric] for r in reports]
        bars = ax.bar(
            names,
            values,
            edgecolor="black",
            color=get_hex_colors(["my_blue", "my_red", "my_green", "my_yellow"]),
            hatch=["", "//", "\\\\", "xx"],
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

    plt.savefig(f"{get_figure_path()}/acc_comparison.{get_figure_format()}", bbox_inches="tight")


def main():
    model = optimize_all(True)
    estimation_cache = QueryEstimationCache(model, True)
    comparison_plot(estimation_cache)


if __name__ == "__main__":
    main()

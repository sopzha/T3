import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache
from src.figures.infra import setup_matplotlib_latex_font, get_figure_path, get_figure_format
from src.metrics import q_error
from src.train import optimize_all


def get_error_histogram(estimation_cache: QueryEstimationCache):
    setup_matplotlib_latex_font()

    dbs = DatabaseManager.get_test_databases()
    benchmarks = DataCollector.collect_benchmarks(dbs, False)
    runtimes = [b.get_total_runtime() for b in benchmarks]

    estimates = [estimation_cache.queries[q.name].estimated_time for q in benchmarks]
    q_errors = [q_error(e, r) for e, r in zip(estimates, runtimes)]
    data = np.array(q_errors)
    data1 = data[data < 20]
    data2 = data[data >= 20]

    # break space into two plots
    start, center, stop = 1, 5, 40
    n_bins1 = (center - start) * 4
    bin_edges1 = np.linspace(start, center, num=n_bins1 + 1)
    bin_width1 = (center - start) / n_bins1
    n_bins2 = (stop - center) // 5
    bin_edges2 = np.linspace(center, stop, num=n_bins2 + 1)
    bin_width2 = (stop - center) / n_bins2

    # Calculate histogram with logarithmic bins
    hist1, bins1 = np.histogram(data1, bins=bin_edges1)
    hist2, bins2 = np.histogram(data2, bins=bin_edges2)

    # Plot the histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    plt.subplots_adjust(wspace=0.001)

    # Plot the histograms
    ax1.hist(bins1[:-1], bins1, weights=hist1, edgecolor="black")
    ax2.hist(bins2[:-1], bins2, weights=hist2, edgecolor="black")
    # plt.hist(bins[:-1], bins, weights=hist, edgecolor="black")

    # Plot numbers on bars
    for bins, hist, ax, bw in ((bins1, hist1, ax1, bin_width1), (bins2, hist2, ax2, bin_width2)):
        for bin_boundary, freq in zip(bins, hist):
            if 10 > freq > 0:
                ax.annotate(
                    f"{freq:.0f}",
                    xy=(bin_boundary + bw / 2, freq),
                    xycoords="data",
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # naming
    ax1.set_xlim(start, center)
    ax2.set_xlim(center, stop)
    ax1.set_xticks(bin_edges1[::4])
    ax2.set_xticks(bin_edges2)
    ax1.set_xlabel(" ")

    ax1.set_ylabel("Frequency")
    ax2.tick_params(axis="y", which="both", length=0)

    plt.savefig(f"{get_figure_path()}/test_accuracy_histogram.{get_figure_format()}", bbox_inches="tight")


def main():
    model = optimize_all()
    estimation_cache = QueryEstimationCache(model, False)
    get_error_histogram(estimation_cache)


if __name__ == "__main__":
    main()

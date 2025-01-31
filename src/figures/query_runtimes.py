import math

import numpy as np
from matplotlib import pyplot as plt

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.figures.infra import setup_matplotlib_latex_font, get_figure_path, get_figure_format


def get_benchmark_variance():
    setup_matplotlib_latex_font()

    dbs = DatabaseManager.get_all_databases()
    benchmarks = DataCollector.collect_benchmarks(dbs, False)
    runtimes = [b.get_total_runtime() for b in benchmarks]
    data = np.array(runtimes)
    # Create logarithmically spaced bins
    n_bins = 30
    bin_edges = np.logspace(np.log10(data.min()), np.log10(data.max()), num=n_bins)

    # Calculate histogram with logarithmic bins
    hist, bins = np.histogram(data, bins=bin_edges)

    # Plot the histogram
    plt.figure(figsize=(6, 3))
    plt.hist(bins[:-1], bins, weights=hist, edgecolor="black")

    # naming
    plt.xscale("log")
    plt.xlabel("Running Time of Query in Seconds")
    plt.ylabel("Frequency")

    plt.savefig(f"{get_figure_path()}/query_runtime_distribution.{get_figure_format()}", bbox_inches="tight")


def main():
    get_benchmark_variance()


if __name__ == "__main__":
    main()

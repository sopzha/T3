from dataclasses import dataclass
from typing import Optional

import numpy as np
from tabulate import tabulate

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.model import Model
from src.optimizer import BenchmarkedQuery, QueryCategory


@dataclass
class EstimatedQuery:
    query: BenchmarkedQuery
    estimated_time: float
    pipeline_times: list[float]


class QueryEstimationCache:
    def __init__(self, model: Model, predicted_cardinalities):
        print("evaluating model on all queries... ", end="")
        # benchmarks = DataCollector.collect_benchmarks(DatabaseManager.get_all_databases(), predicted_cardinalities)
        benchmarks = DataCollector.collect_benchmarks(
            DatabaseManager.get_databases(["tpchSf10"]), predicted_cardinalities
        )
        self.queries: dict[str, EstimatedQuery] = {}
        for b in benchmarks:
            estimate = model.estimate_runtime(b)
            pipeline_estimate = model.estimate_pipeline_runtime(b)
            self.queries[b.name] = EstimatedQuery(b, estimate, pipeline_estimate)
        print("done")


def stringify(v) -> str:
    if isinstance(v, float):
        if v < 10:
            return f"{v:.3f}"
        elif 10 <= v < 100:
            return f"{v:.2f}"
        elif 100 <= v < 1000:
            return f"{v:.1f}"
        else:
            return f"{v:.0f}"
    else:
        return f"{v}"


def statistics_with_error_function(
    errors: list[tuple[float, BenchmarkedQuery, float]],
    dataset: str,
    error_name: str,
    verbose: bool,
) -> dict:
    if len(errors) == 0:
        return {}
    errors.sort(key=lambda x: x[0])
    report = {
        f"{dataset} dataset, metric": error_name,
        "Avg": np.average([e[0] for e in errors]),
        "p10": np.quantile([e[0] for e in errors], 0.1),
        "p50": np.median([e[0] for e in errors]),
        "p90": np.quantile([e[0] for e in errors], 0.90),
        "p95": np.quantile([e[0] for e in errors], 0.95),
        "Max": errors[-1][0],
    }
    if verbose:
        headers = list(report.keys())
        table = [[stringify(report[c]) for c in headers]]
        print(tabulate(table, headers, tablefmt="github"))
        print(
            f"max_name: {errors[-1][1].name}, "
            f"max_true: {errors[-1][1].get_total_runtime()}, "
            f"max_estimated: {errors[-1][2]}"
        )
    return report


def get_errors(
    queries: list[BenchmarkedQuery],
    results: list[float],
    function: callable,
    query_category: Optional[QueryCategory],
    excluded_query_category: Optional[QueryCategory],
) -> list[tuple[float, BenchmarkedQuery, float]]:
    assert query_category is None or excluded_query_category is None
    errors = []
    for bench, result in zip(queries, results):
        if (query_category is None or bench.query_category == query_category) and (
            excluded_query_category is None or bench.query_category != excluded_query_category
        ):
            errors.append((function(bench.get_total_runtime(), result), bench, result))
    return errors

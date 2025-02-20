import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.database import Database
from src.metrics import q_error
from src.optimizer import BenchmarkedQuery, QueryCategory
from src.query_plan import QueryPlan
from src.redshift_util import get_query_plan, get_query_runtimes
from src.util import fifo_cache


def arg_median(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l, r = len(a) // 2 - 1, len(a) // 2
        left = np.partition(a, l)[l]
        # right = np.partition(a, r)[r]
        return np.where(a == left)[0][0]  # , np.where(a == right)[0][0]


class DataCollector:
    @staticmethod
    def read_runtime(file: Path) -> float:
        with open(file, "r") as benchmark_json:
            benchmark_json = json.load(benchmark_json)
        runtimes = [b["executionTime"] for b in benchmark_json["benchmarks"]]
        return np.median(runtimes)

    @staticmethod
    def read_query(file: Path) -> str:
        with open(file, "r") as benchmark_json:
            benchmark_json = json.load(benchmark_json)
        query = benchmark_json["plan"]["query_text"]
        return query

    @staticmethod
    def get_type(file: Path) -> QueryCategory:
        name = str(file.parent.name)
        return next(x for x in QueryCategory if x.name == name)

    @staticmethod
    def read_analyzed_plan(file: Path, db: Database, predicted_cardinalities: bool) -> BenchmarkedQuery:
        with open(file, "r") as benchmark_json:
            benchmark_json = json.load(benchmark_json)
        plan = QueryPlan(benchmark_json["plan"]["plan"], db, predicted_cardinalities)
        plan.build_pipelines(benchmark_json["plan"]["plan"]["analyzePlanPipelines"])
        runtimes = [b["executionTime"] for b in benchmark_json["benchmarks"]]
        # runtimes.sort()
        query_text = benchmark_json["plan"]["query_text"]
        return BenchmarkedQuery(plan, runtimes, file.name, query_text, DataCollector.get_type(file))

    @staticmethod
    def read_analyzed_plan_redshift(file: Path, db: Database, predicted_cardinalities: bool) -> BenchmarkedQuery:
        with open(file, "r") as benchmark_json:
            benchmark_json = json.load(benchmark_json)
        query_text = benchmark_json["plan"]["query_text"]
        query_plan = get_query_plan(query_text)

        plan = QueryPlan(query_plan, db, predicted_cardinalities)
        plan.build_pipelines(query_plan["analyzePlanPipelines"])

        runtimes = get_query_runtimes(query_text)

        return BenchmarkedQuery(plan, runtimes, file.name, query_text, DataCollector.get_type(file))

    @staticmethod
    def group_by_multiple_runs(benchmarks: list[BenchmarkedQuery]) -> dict[str, list[BenchmarkedQuery]]:
        result = {}
        regex = re.compile("(.*\\d+[a-z]?)\\.json$")
        for b in benchmarks:
            match = regex.match(b.name)
            if match:
                query_name = match.group(1)
                # print(query_name)
                if query_name not in result:
                    result[query_name] = []
                result[query_name].append(b)
            else:
                print(f"could not categorize benchmarked query: {b.name}")
        return result

    @staticmethod
    def select_representative_queries(benchmarks: dict[str, list[BenchmarkedQuery]]) -> list[BenchmarkedQuery]:
        result = []
        for group in benchmarks.values():
            runtimes = np.array([b.total_runtime for b in group])
            med = arg_median(runtimes)
            result.append(group[med])
        return result

    @staticmethod
    @fifo_cache
    def collect_db_benchmark_runs(db: Database, predicted_cardinalities) -> list[BenchmarkedQuery]:
        result = []
        files = [f for f in Path(f"data/{db.get_path()}").rglob("*.json")]
        files.sort()
        for file in files:
            # result.append(DataCollector.read_analyzed_plan(file, db, predicted_cardinalities))
            try:
                result.append(DataCollector.read_analyzed_plan_redshift(file, db, predicted_cardinalities))
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        return result

    @staticmethod
    def collect_benchmarks(
        dbs: list[Database],
        predicted_cardinalities: bool,
        query_category: list[QueryCategory] = [],
        exclude_query_category: list[QueryCategory] = [],
    ) -> list[BenchmarkedQuery]:
        benchmarks = []
        for db in dbs:
            benchmarks += DataCollector.collect_db_benchmark_runs(db, predicted_cardinalities)
        if len(query_category) != 0:
            benchmarks = [b for b in benchmarks if b.query_category in query_category]
        if len(exclude_query_category) != 0:
            benchmarks = [b for b in benchmarks if b.query_category not in exclude_query_category]
        return benchmarks

    @staticmethod
    def check_runtimes_integrity(benchmark: BenchmarkedQuery) -> bool:
        result = True

        # bound in seconds, errors below are ignored, errors above are checked with q-error
        acceptable_absolute_error = 0.002
        acceptable_q_error = 1.10
        acceptable_fraction_of_outliers = 1 / 3
        minimal_number_of_non_outliers = 2
        minimal_runs = 3

        n_runs = len(benchmark.total_runtimes)
        if n_runs < minimal_runs:
            print(f"found {n_runs} runs for {benchmark.name}, but expected at least {minimal_runs}")
            return False

        times = np.array(benchmark.total_runtimes)
        i_med = arg_median(times)
        med: float = times[i_med]
        q_errors = np.array([q_error(med, x) for x in times])
        q_errors_to_high_mask = q_errors > acceptable_q_error
        absolute_errors = np.abs(times - med)
        absolute_errors_low_mask = absolute_errors < acceptable_absolute_error
        outlier_mask = q_errors_to_high_mask & (~absolute_errors_low_mask)
        n_outliers = np.sum(outlier_mask)
        n_non_outliers = n_runs - n_outliers
        current_minimal_number_of_non_outliers = max(
            minimal_number_of_non_outliers, int((1 - acceptable_fraction_of_outliers) * n_runs)
        )
        if n_non_outliers < current_minimal_number_of_non_outliers:
            print(
                f"insuffiecient number of non-outliers "
                f"({n_non_outliers}/{current_minimal_number_of_non_outliers}) in query {benchmark.name}\n"
                f" q_errors: {sorted(q_errors)}\n"
                f" absolute errors: {sorted(absolute_errors)}\n"
                f" runtimes: {sorted(times)}\n"
                f" text:\n{benchmark.query_text}"
            )
            result = False
        return result

    @staticmethod
    def check_analyze_plan_duration_integrity(benchmark: BenchmarkedQuery, verbose: bool = True) -> bool:
        result = True
        med = benchmark.get_total_runtime()
        acceptable_analyze_plan_duration_q_error = 1.2
        if q_error(med, benchmark.get_analyze_plan_runtime()) > acceptable_analyze_plan_duration_q_error:
            if verbose:
                print(
                    f"analyze plan duration is way off of benchmarked time in query {benchmark.name}\n"
                    f" benchmarked: {med}\n"
                    f" analyze plan: {benchmark.get_analyze_plan_runtime()}\n"
                    f" q_error: {q_error(med, benchmark.get_analyze_plan_runtime())}"
                )
            result = False
        return result

    @staticmethod
    def check_single_integrity(benchmark: BenchmarkedQuery) -> bool:
        a = DataCollector.check_runtimes_integrity(benchmark)
        b = DataCollector.check_analyze_plan_duration_integrity(benchmark)
        return a and b

    @staticmethod
    def check_benchmark_integrity(dbs: list[Database]) -> bool:
        """
        check that:
        - each run actually executes the same query
        - non outlier runs vary by at most 10%
        - there is a sufficient number of non-outliers
        - the number of outliers is bounded
        - analyze_plan pipeline execution duration is not too far off
        """
        result = True

        benchmarks = DataCollector.collect_benchmarks(dbs)
        for benchmark in benchmarks:
            current = DataCollector.check_single_integrity(benchmark)
            result = result and current

        return result

    @staticmethod
    def get_benchmark_q_errors(dbs: list[Database]) -> np.ndarray:
        """
        all deviations of the benchmark runs in a two-dimensional array
        """
        result = []
        abs_result = []
        benchmarks = DataCollector.collect_benchmarks(dbs)
        for benchmark in benchmarks:
            times = np.array(benchmark.total_runtimes)
            i_med = arg_median(times)
            med: float = times[i_med]
            q_errors = np.array([q_error(med, x) for x in times])
            q_errors.sort()
            result.append(q_errors)
            abs_errors = np.array([abs(med - x) for x in times])
            abs_errors.sort()
            abs_result.append(abs_errors)
        result = np.array(result)
        abs_result = np.array(abs_result)

        cut_off_index = int(result.shape[1] * 2 / 3)
        for result in (result, abs_result):
            x = result[:, cut_off_index]
            print(f"worst: {benchmarks[np.argmax(x)].name}")
            x.sort()
            print(f"accuracy of {len(x)} benchmarks:")
            report = {
                "10\\%": np.quantile(x, 0.1),
                "50\\%": np.median(x),
                "90\\%": np.quantile(x, 0.9),
                "95\\%": np.quantile(x, 0.95),
                "99\\%": np.quantile(x, 0.99),
                "avg": np.average(x),
                # "max": x[-1],
            }
            print(" & ".join(str(x) for x in report))
            print(" & ".join(f"{x:.3f}" for x in report.values()))
            # print(" " + ", ".join(f"{k} {v:.3f}" for k, v in report.items()))

        return result

    @staticmethod
    def inspect_runtime_statistics(dbs: list[Database]):
        def print_runtimes_stats(runtimes: list[float]):
            runtimes = [r * 1000 for r in runtimes]
            n_over_1_ms = len([r for r in runtimes if r > 1])
            n_over_10_ms = len([r for r in runtimes if r > 10])
            print(
                f" min {min(runtimes):.2f}, max {max(runtimes):.2f}, "
                f"med {np.median(runtimes):.2f}, avg {np.average(runtimes):.2f} (in milliseconds)\n"
                f" {n_over_1_ms / len(runtimes) * 100:.2f}% are over 1 ms"
                f" {n_over_10_ms / len(runtimes) * 100:.2f}% are over 10 ms"
            )

        per_type_runtimes = {}
        per_db_runtimes = {}
        all_runtimes = []
        for db in dbs:
            subdirs = [d for d in Path(f"data/{db.get_path()}").iterdir() if d.is_dir()]
            for dir in subdirs:
                print(dir)
                files = [f for f in dir.rglob("*.json")]
                files.sort()
                runtimes = [DataCollector.read_runtime(f) for f in files]
                if len(runtimes) > 0:
                    if dir.name not in per_type_runtimes:
                        per_type_runtimes[dir.name] = []
                    per_type_runtimes[dir.name] += runtimes
                    if db.schema.name not in per_db_runtimes:
                        per_db_runtimes[db.schema.name] = []
                    per_db_runtimes[db.schema.name] += runtimes
                    all_runtimes += runtimes
                    print_runtimes_stats(runtimes)
                else:
                    print(" empty")

        print()
        print("Per Query Type")
        for q_type, runtimes in per_type_runtimes.items():
            print(f"{q_type} ({len(runtimes)})")
            print_runtimes_stats(runtimes)

        print()
        print("Per Database")
        for db in sorted(per_db_runtimes):
            runtimes = per_db_runtimes[db]
            print(f"{db} ({len(runtimes)})")
            print_runtimes_stats(runtimes)

        print()
        print("All:")
        print_runtimes_stats(all_runtimes)

    @staticmethod
    def save_queries(dbs: list[Database], file: Path, filter: Optional[str] = None):
        per_db_queries = {}
        for db in dbs:
            subdirs = [d for d in Path(f"data/{db.get_path()}").iterdir() if d.is_dir()]
            for dir in subdirs:
                if filter is not None and dir.name != filter:
                    continue
                print(dir)
                files = [f for f in dir.rglob("*.json")]
                files.sort()
                queries = [DataCollector.read_query(f) for f in files]
                if len(queries) > 0:
                    if db.schema.name not in per_db_queries:
                        per_db_queries[db.schema.name] = []
                    per_db_queries[db.schema.name] += queries
                else:
                    print(" empty")
        print(per_db_queries)
        with open(file, "w") as fd:
            json.dump(per_db_queries, fd)

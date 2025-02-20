import json
import re
import subprocess
from pathlib import Path
from time import sleep
from typing import Callable

import shlex
import requests

from src.data_collection import DataCollector
from src.database import Database
from src.optimizer import BenchmarkedQuery, QueryCategory
from src.query_generation.aggregations import sample_group_by_query
from src.query_generation.join_agg import generate_join_agg_query, generate_join_simple_agg_query
from src.query_generation.join_graph import generate_join_query
from src.query_generation.selections import SelectionFactory, sample_complex_selection_query
from src.query_generation.window_function import WindowFunctionFactory
from src.query_plan import QueryPlan
from src.server import start_webserver_new

SERVER_PATH = "./webserver"
DB_PATH = "benchmark_setup/db/all.db"


def read_file(file: Path) -> str:
    with open(file, "r") as fd:
        return fd.read()


def format_numbers_with_zeros(string: str, width: int = 3) -> str:
    pattern = r"\b(\d+)\b"
    formatted_string = re.sub(pattern, lambda match: match.group(0).zfill(width), string)
    return formatted_string


class QueryRuntimeExceededException(Exception):
    def __init__(self, message="Query ran into Database time limit"):
        self.message = message
        super().__init__(self.message)


class AnalyzePlanNotPlausibleException(Exception):
    def __init__(self, message="Query returned times for AnalyzePlan that do not match benchmark times"):
        self.message = message
        super().__init__(self.message)


class Benchmarker:
    plan_verbose_analyze: str

    def __init__(self, server: str):
        self.plan_verbose_analyze = f"{server}/planVerboseAnalyze"
        self.benchmark_url = f"{server}/benchmark"

    def _benchmark(self, db: Database, query: str) -> str:
        query_text = f"set search_path = {db.get_search_path()}, public;\n\n{query}"
        query_text = query_text.encode("utf-8")
        response = requests.post(self.benchmark_url, query_text)
        return response.text

    def _query(self, db: Database, query: str) -> str:
        query_text = f"set search_path = {db.get_search_path()}, public;\n\n{query}"
        query_text = query_text.encode("utf-8")
        response = requests.post(self.plan_verbose_analyze, query_text)
        return response.text

    def analyze_query(self, db: Database, q: str) -> dict:
        result = self._query(db, q)
        result = json.loads(result)
        if (
            "status" in result
            and result["status"] == "exception"
            and result["exception"] == "query could not be processed due to a CPU time limitation"
        ):
            raise QueryRuntimeExceededException()
        assert "optimizersteps" in result, f"Query failed!\nQuery:\n{q}\nresponse:\n{result}"
        result = result["optimizersteps"][-1]
        result["query_text"] = q
        return result

    def run_query(self, db: Database, q: str) -> dict:
        result = self._benchmark(db, q)
        result = json.loads(result)
        if (
            "status" in result
            and result["status"] == "exception"
            and result["exception"] == "query could not be processed due to a CPU time limitation"
        ):
            raise QueryRuntimeExceededException()
        assert "results" in result, f"Query failed!\nQuery:\n{q}\nresponse:\n{result}"
        result = result["results"][0]
        result["query_text"] = q
        return result

    def n_raw_runs(self, db: Database, q: str, n: int) -> list[float]:
        return [self.run_query(db, q)["executionTime"] for _ in range(n)]

    @staticmethod
    def store(outfile: Path, result: dict):
        with open(outfile, "w") as outfile:
            json.dump(result, outfile)

    @staticmethod
    def get_fixed_queries(db: Database) -> dict[str, Callable[[], str]]:
        if db.fixedQueryPath is not None:
            result = {}
            for file in sorted(db.fixedQueryPath.glob("*.sql")):
                q_name = file.name[: len(file.name) - len(".sql")]
                q = read_file(file)
                result[q_name] = lambda r=q: r
            return result
        else:
            return {}

    @staticmethod
    def get_category_dict(db: Database) -> dict[QueryCategory, Callable[[], str]]:
        selection_factory = SelectionFactory(db.schema)
        window_factory = WindowFunctionFactory(db.schema)
        return {
            QueryCategory.select: lambda: selection_factory.sample_selection_query(),
            QueryCategory.join: lambda: generate_join_query(db.schema, False, False),
            QueryCategory.select_join: lambda: generate_join_query(db.schema, True, False),
            QueryCategory.pseudo_aggregate: lambda: sample_group_by_query(db.schema, False, pseudo_group_by=True),
            QueryCategory.aggregate: lambda: sample_group_by_query(db.schema, False),
            QueryCategory.select_aggregate: lambda: sample_group_by_query(db.schema, True),
            QueryCategory.join_agg: lambda: generate_join_agg_query(db.schema, False, False),
            QueryCategory.select_join_agg: lambda: generate_join_agg_query(db.schema, True, False),
            QueryCategory.join_simple_agg: lambda: generate_join_simple_agg_query(db.schema, False, False),
            QueryCategory.select_join_simple_agg: lambda: generate_join_simple_agg_query(db.schema, True, False),
            QueryCategory.complex_select: lambda: sample_complex_selection_query(db.schema),
            QueryCategory.complex_select_agg: lambda: sample_group_by_query(
                db.schema, select_input=True, pseudo_group_by=True, complex_select=True
            ),
            QueryCategory.complex_select_join: lambda: generate_join_query(db.schema, True, True),
            QueryCategory.complex_select_join_agg: lambda: generate_join_agg_query(db.schema, True, True),
            QueryCategory.complex_select_join_simple_agg: lambda: generate_join_simple_agg_query(db.schema, True, True),
            QueryCategory.window: lambda: window_factory.get_query(),
        }

    @staticmethod
    def get_queries(db: Database, n_queries: int) -> dict[QueryCategory, dict[str, Callable[[], str]]]:
        result = {}
        category_dict = Benchmarker.get_category_dict(db)
        for category, generator in category_dict.items():
            current_result = {}
            for i in range(1, n_queries + 1):
                name = f"{category.name}_{i:03d}"
                current_result[name] = generator
            result[category] = current_result
        return result

    @staticmethod
    def mock_benchmarked_query(
        plan_json: dict,
        runtimes: list[float],
        query_name: str,
        query_text: str,
        db: Database,
        query_category: QueryCategory,
    ) -> BenchmarkedQuery:
        plan = QueryPlan(plan_json, db, False)
        plan.build_pipelines(plan_json["analyzePlanPipelines"])
        return BenchmarkedQuery(plan, runtimes, query_name, query_text, query_category)

    def retry_analyze(
        self,
        analyzed_query: dict,
        db: Database,
        query_name: str,
        query: str,
        benchmark_times: list[float],
        query_category: QueryCategory,
        n_tries: int = 5,
        ignore_error: bool = False,
    ) -> dict:
        if n_tries <= 0:
            raise AnalyzePlanNotPlausibleException()
        bench_query = Benchmarker.mock_benchmarked_query(
            analyzed_query["plan"],
            benchmark_times,
            "",
            "",
            db,
            query_category,
        )
        if not DataCollector.check_analyze_plan_duration_integrity(bench_query, False) and not ignore_error:
            return self.retry_analyze(
                self.analyze_query(db, query),
                db,
                query_name,
                query,
                benchmark_times,
                query_category,
                n_tries - 1,
            )
        else:
            return analyzed_query

    def get_n_runs(
        self,
        db: Database,
        n: int,
        q: Callable[[], str],
        query_name: str,
        query_category: QueryCategory,
    ) -> tuple[dict, list[dict]]:
        err_counter = 0
        query = ""
        while True:
            try:
                query = q()
                analyzed_query = self.analyze_query(db, query)
                benchmarks = [self.run_query(db, query) for _ in range(n)]
                benchmark_times = [float(b["executionTime"]) for b in benchmarks]
                analyzed_query = self.retry_analyze(
                    analyzed_query,
                    db,
                    query_name,
                    query,
                    benchmark_times,
                    query_category,
                    # ignore_error=err_counter > 10,
                    ignore_error=True,
                )
                return analyzed_query, benchmarks
            except QueryRuntimeExceededException:
                print("t", end="")
            except AnalyzePlanNotPlausibleException:
                print(f"x", end="")
                err_counter += 1
            except Exception:
                print(f"Failed Query:\n {query}")
                sleep(2)
                start_webserver_new()

    def get_all_queries(self, db: Database, n_queries: int) -> dict[QueryCategory, dict[str, [Callable[[], str]]]]:
        all_queries: dict[QueryCategory, dict[str, Callable[[], str]]] = {
            QueryCategory.fixed: self.get_fixed_queries(db),
        }
        all_queries.update(self.get_queries(db, n_queries))
        return all_queries

    def run_database(
        self,
        db: Database,
        n_runs: int,
        n_queries: int,
        verbose: bool = False,
    ):
        # Flexible callable to get queries, so random sampling can be repeated
        all_queries = self.get_all_queries(db, n_queries)
        for query_category, bench in all_queries.items():
            bench_name = query_category.name
            if verbose:
                print(f" {bench_name}", end="")
            out_path = Path(f"data/{db.get_path()}/{bench_name}")
            out_path.mkdir(parents=True, exist_ok=True)
            for query_name, get_query in bench.items():
                filename = f"{db.get_search_path()}_q{query_name}.json"
                outfile = out_path / filename
                if outfile.exists():
                    bench_query = DataCollector.read_analyzed_plan(outfile, db, False)
                    # here we can toggle re-running queries that do not fit the integrity requirements
                    if True or DataCollector.check_analyze_plan_duration_integrity(bench_query, False):
                        print("°", end="", flush=True)
                        continue

                plan, benchmarks = self.get_n_runs(db, n_runs, get_query, query_name, query_category)

                result = {"plan": plan, "benchmarks": benchmarks}
                self.store(outfile, result)
                if verbose:
                    print(".", end="", flush=True)
            if verbose:
                print()

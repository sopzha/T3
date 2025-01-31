from tabulate import tabulate

from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.evaluation import QueryEstimationCache, get_errors, statistics_with_error_function
from src.figures.acc_comparison import get_stage_pred_numbers, get_auto_wlm_pred_numbers
from src.figures.acc_comparison_zero_shot import get_zero_shot_exact_numbers, get_zero_shot_pred_numbers
from src.figures.accuracy_table import latex_accuracy_table
from src.figures.cardinality_degradation import split_databases
from src.metrics import q_error
from src.optimizer import optimize_per_tuple_tree_model, QueryCategory
from src.train import optimize_all


def detailed_accuracy_table():
    exact_model = optimize_all(False)
    pred_model = optimize_all(True)
    exact_exact_cache = QueryEstimationCache(exact_model, False)
    # exact_pred_cache = QueryEstimationCache(exact_model, True)
    pred_pred_cache = QueryEstimationCache(pred_model, True)

    tpcds_sf100_fixed = DataCollector.collect_benchmarks(
        DatabaseManager.get_databases(["tpcdsSf100"]), True, query_category=[QueryCategory.fixed]
    )
    tpcds_sf100_all = DataCollector.collect_benchmarks(DatabaseManager.get_databases(["tpcdsSf100"]), True)
    tpcds_full_fixed = DataCollector.collect_benchmarks(
        DatabaseManager.get_test_databases(), True, query_category=[QueryCategory.fixed]
    )
    tpcds_full_all = DataCollector.collect_benchmarks(DatabaseManager.get_test_databases(), True)

    job_train_dbs, job_test_dbs = split_databases("job")
    job_benchs = DataCollector.collect_benchmarks(job_test_dbs, False, [QueryCategory.fixed])
    job_exact_train_benchs = DataCollector.collect_benchmarks(job_train_dbs, False)
    job_pred_train_benchs = DataCollector.collect_benchmarks(job_train_dbs, True)
    exact_job_model = optimize_per_tuple_tree_model(job_exact_train_benchs)
    pred_job_model = optimize_per_tuple_tree_model(job_pred_train_benchs)
    exact_exact_job_cache = QueryEstimationCache(exact_job_model, False)
    pred_pred_job_cache = QueryEstimationCache(pred_job_model, True)

    data = [
        (
            "T3 Train Queries True Card",
            DataCollector.collect_benchmarks(DatabaseManager.get_train_databases(), False),
            exact_exact_cache,
        ),
        ("T3 TPC-DS Test Queries True Card", tpcds_full_all, exact_exact_cache),
        ("T3 TPC-DS Bench Queries True Card", tpcds_full_fixed, exact_exact_cache),
        ("T3 TPC-DS sf100 Test Queries True Card", tpcds_sf100_all, exact_exact_cache),
        ("T3 TPC-DS sf100 Bench Queries True Card", tpcds_sf100_fixed, exact_exact_cache),
        # ("T3 TPC-DS Test Queries Est Card", tpcds_full_all, exact_pred_cache),
        # ("T3 TPC-DS Bench Queries Est Card", tpcds_full_fixed, exact_pred_cache),
        # ("T3 TPC-DS sf100 Test Queries Est Card", tpcds_sf100_all, exact_pred_cache),
        # ("T3 TPC-DS sf100 Bench Queries Est Card", tpcds_sf100_fixed, exact_pred_cache),
        ("T3 TPC-DS Test Queries Est Card", tpcds_full_all, pred_pred_cache),
        ("T3 TPC-DS Bench Queries Est Card", tpcds_full_fixed, pred_pred_cache),
        ("T3 TPC-DS sf100 Test Queries Est Card", tpcds_sf100_all, pred_pred_cache),
        ("T3 TPC-DS sf100 Bench Queries Est Card", tpcds_sf100_fixed, pred_pred_cache),
        ("T3 JOB Bench Queries True Card", job_benchs, exact_exact_job_cache),
        ("T3 JOB Bench Queries Est Card", job_benchs, pred_pred_job_cache),
    ]

    report = {}
    for n, queries, estimation_cache in data:
        estimates = [estimation_cache.queries[q.name].estimated_time for q in queries]
        q_errors = get_errors(queries, estimates, q_error, None, None)
        report[n] = statistics_with_error_function(q_errors, n, "Q-Error", False)

    report.update(
        {
            "Zero Shot JOB True Card": get_zero_shot_exact_numbers(),
            "Zero Shot JOB Est Card": get_zero_shot_pred_numbers(),
            "Stage Redshift Trace Est Card": get_stage_pred_numbers(),
            "AutoWLM Redshift Trace Est Card": get_auto_wlm_pred_numbers(),
        }
    )

    columns = ["Workload", "p50", "p90", "Avg"]
    report_table = [columns] + [[n] + [f"{s[c]:.2f}" for c in columns[1:]] for n, s in report.items()]
    print(tabulate(report_table, colalign=["right"] * 4))

    print(latex_accuracy_table(report))


def main():
    detailed_accuracy_table()


if __name__ == "__main__":
    main()

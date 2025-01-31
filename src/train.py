from src.data_collection import DataCollector
from src.database_manager import DatabaseManager
from src.model import Model
from src.optimizer import optimize_per_tuple_tree_model


def optimize_all(predicted_cardinalities: bool = False) -> Model:
    excluded_from_train = [
        # QueryCategory.fixed,
        # QueryCategory.select,
        # QueryCategory.aggregate,
        # QueryCategory.pseudo_aggregate,
        # QueryCategory.select_aggregate,
        # QueryCategory.join,
        # QueryCategory.select_join,
        # QueryCategory.select_join_agg,
        # QueryCategory.join_simple_agg,
        # QueryCategory.select_join_simple_agg,
        # QueryCategory.complex_select,
        # QueryCategory.complex_select_join,
        # QueryCategory.complex_select_agg,
        # QueryCategory.complex_select_join_agg,
        # QueryCategory.complex_select_join_simple_agg,
    ]
    benchmarks = DataCollector.collect_benchmarks(
        DatabaseManager.get_train_databases(), predicted_cardinalities, exclude_query_category=excluded_from_train
    )
    return optimize_per_tuple_tree_model(benchmarks)

from src.benchmark import Benchmarker
from src.database_manager import DatabaseManager


def update_schema(server: str):
    for db in DatabaseManager.get_all_databases():
        db.query_missing_table_sizes(server)
    for db in DatabaseManager.get_all_databases():
        db.query_missing_column_sizes(server)
    for db in DatabaseManager.get_all_databases():
        db.query_missing_column_statistics(server)
    for db in DatabaseManager.get_all_databases():
        db.query_missing_column_samples(server)


def benchmark(address: str = "http://127.0.0.1:8000"):
    n_iterations = 10
    n_random_queries = 40
    print("updating schema")
    update_schema(address)
    print("done")
    benchmarker = Benchmarker(address)
    for i, db in enumerate(DatabaseManager.get_all_databases()):
        print(f"running benchmarks for {db.schema.name} ({i + 1}/{len(DatabaseManager.get_all_databases())})")
        benchmarker.run_database(db, n_iterations, n_random_queries, verbose=True)

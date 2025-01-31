from pathlib import Path
from typing import Optional

from src.database import Database

DATABASE_DICT: Optional[dict[str, Database]] = None


def get_database_dict() -> dict[str, Database]:
    global DATABASE_DICT
    if DATABASE_DICT is None:
        DATABASE_DICT = {
            "tpchSf1": Database.get_database(
                "tpchSf1", "benchmark_setup/schemata/01-tpchSf1-schema.sql", Path("queries/tpch")
            ),
            "tpchSf10": Database.get_database(
                "tpchSf10", "benchmark_setup/schemata/01-tpchSf10-schema.sql", Path("queries/tpch")
            ),
            "tpchSf100": Database.get_database(
                "tpchSf100", "benchmark_setup/schemata/01-tpchSf100-schema.sql", Path("queries/tpch")
            ),
            "tpcdsSf1": Database.get_database(
                "tpcdsSf1", "benchmark_setup/schemata/02-tpcdsSf1-schema.sql", Path("queries/tpcds")
            ),
            "tpcdsSf10": Database.get_database(
                "tpcdsSf10", "benchmark_setup/schemata/02-tpcdsSf10-schema.sql", Path("queries/tpcds")
            ),
            "tpcdsSf100": Database.get_database(
                "tpcdsSf100", "benchmark_setup/schemata/02-tpcdsSf100-schema.sql", Path("queries/tpcds")
            ),
            "job": Database.get_database("job", "benchmark_setup/schemata/03-job-schema.sql", Path("queries/job")),
            "airline": Database.get_database("airline", "benchmark_setup/schemata/04-airline-schema.sql"),
            "ssb": Database.get_database("ssb", "benchmark_setup/schemata/05-ssb-schema.sql"),
            "walmart": Database.get_database("walmart", "benchmark_setup/schemata/06-walmart-schema.sql"),
            "financial": Database.get_database("financial", "benchmark_setup/schemata/07-financial-schema.sql"),
            "basketball": Database.get_database("basketball", "benchmark_setup/schemata/08-basketball-schema.sql"),
            "accident": Database.get_database("accident", "benchmark_setup/schemata/09-accident-schema.sql"),
            "movielens": Database.get_database("movielens", "benchmark_setup/schemata/10-movielens-schema.sql"),
            "baseball": Database.get_database("baseball", "benchmark_setup/schemata/11-baseball-schema.sql"),
            "hepatitis": Database.get_database("hepatitis", "benchmark_setup/schemata/12-hepatitis-schema.sql"),
            "tournament": Database.get_database("tournament", "benchmark_setup/schemata/13-tournament-schema.sql"),
            "credit": Database.get_database("credit", "benchmark_setup/schemata/14-credit-schema.sql"),
            "employee": Database.get_database("employee", "benchmark_setup/schemata/15-employee-schema.sql"),
            "consumer": Database.get_database("consumer", "benchmark_setup/schemata/16-consumer-schema.sql"),
            "geneea": Database.get_database("geneea", "benchmark_setup/schemata/17-geneea-schema.sql"),
            "genome": Database.get_database("genome", "benchmark_setup/schemata/18-genome-schema.sql"),
            "carcinogenesis": Database.get_database(
                "carcinogenesis", "benchmark_setup/schemata/19-carcinogenesis-schema.sql"
            ),
            "seznam": Database.get_database("seznam", "benchmark_setup/schemata/20-seznam-schema.sql"),
            "fhnk": Database.get_database("fhnk", "benchmark_setup/schemata/21-fhnk-schema.sql"),
        }
    return DATABASE_DICT


class DatabaseManager:
    @staticmethod
    def get_database(name: str) -> Database:
        return get_database_dict()[name]

    @staticmethod
    def get_databases(names: list[str]) -> list[Database]:
        return [get_database_dict()[name] for name in names]

    @staticmethod
    def get_train_databases() -> list[Database]:
        return DatabaseManager.get_databases(
            [
                "tpchSf1",
                "tpchSf10",
                "tpchSf100",
                "job",
                "airline",
                "ssb",
                "walmart",
                "financial",
                "basketball",
                "accident",
                "movielens",
                "baseball",
                "hepatitis",
                "tournament",
                "credit",
                "employee",
                "consumer",
                "geneea",
                "genome",
                "carcinogenesis",
                "seznam",
                "fhnk",
            ]
        )

    @staticmethod
    def get_test_databases() -> list[Database]:
        return DatabaseManager.get_databases(
            [
                "tpcdsSf1",
                "tpcdsSf10",
                "tpcdsSf100",
            ]
        )

    @staticmethod
    def get_all_databases() -> list[Database]:
        return list(get_database_dict().values())

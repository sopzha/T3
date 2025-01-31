import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jsonpickle
import requests

from src.schemata import Schema, Type, load_schema


COLUMN_SAMPLE_SIZE = 100


@dataclass
class FixedQueries:
    # location of query files
    path: Path


@dataclass
class Database:
    schema: Schema
    fixedQueryPath: Optional[Path]

    def get_search_path(self) -> str:
        return self.schema.name

    def get_path(self) -> str:
        return self.get_search_path()

    def query_missing_table_sizes(self, server: str):
        url = f"{server}/query"
        for table_name, table in self.schema.tables.items():
            if table.size is None:
                query_text = (
                    f"set search_path = {self.get_search_path()}, public;\n\n"
                    f"select count(*)\n"
                    f"from {table_name};"
                )
                response = requests.post(url, query_text)
                result = json.loads(response.text)
                result = result["results"][0]["result"][0][0]
                table.size = result
        self.write_to_cache()

    def query_missing_column_sizes(self, server: str):
        url = f"{server}/query"
        for table in self.schema.tables.values():
            for column in table.columns.values():
                if column.size is None:
                    assert column.type in (
                        Type.Varchar,
                        Type.Text,
                        Type.CharArray,
                    ), f"columns of type {column.type} should have a size"
                    query_text = (
                        f"set search_path = {self.get_search_path()}, public;\n\n"
                        f"select avg(length({column.name}))\n"
                        f"from {table.table_name};"
                    )
                    response = requests.post(url, query_text)
                    result = json.loads(response.text)
                    result = result["results"][0]["result"][0][0]
                    if result is None:
                        result = 0
                    column.size = result
        self.write_to_cache()

    def query_column_boundaries(self, server: str, table_name: str, column_name: str):
        url = f"{server}/query"
        query_text = (
            f"set search_path = {self.get_search_path()}, public;\n\n"
            f"select count(distinct {column_name}),"
            f"  min({column_name}),"
            f"  max({column_name}) "
            f"from {table_name}"
        )
        response = requests.post(url, query_text)
        result = json.loads(response.text)
        result = result["results"][0]["result"]
        result = [r[0] for r in result]
        return result

    def query_column_distinct_count(self, server: str, table_name: str, column_name: str):
        url = f"{server}/query"
        query_text = (
            f"set search_path = {self.get_search_path()}, public;\n\n"
            f"select count(distinct {column_name}) "
            f"from {table_name}"
        )
        response = requests.post(url, query_text)
        result = json.loads(response.text)
        result = result["results"][0]["result"]
        result = [r[0] for r in result]
        return result

    def query_column_sample_count(self, server: str, table_name: str, column_name: str):
        url = f"{server}/query"
        query_text = (
            f"set search_path = {self.get_search_path()}, public;\n\n"
            f"select {column_name} "
            f"from {table_name} "
            f"order by RANDOM() "
            f"limit {COLUMN_SAMPLE_SIZE}"
        )
        response = requests.post(url, query_text)
        result = json.loads(response.text)
        result = result["results"][0]["result"]
        result = result[0]
        return result

    def query_missing_column_statistics(self, server: str):
        for table in self.schema.tables.values():
            for column in table.columns.values():
                if column.statistics_missing():
                    s = self.query_column_boundaries(server, table.table_name, column.name)
                    column.distinct_count, column.min_val, column.max_val = s
                if column.distinct_missing():
                    s = self.query_column_distinct_count(server, table.table_name, column.name)
                    column.distinct_count = s[0]

        self.write_to_cache()

    def query_missing_column_samples(self, server: str):
        for table in self.schema.tables.values():
            for column in table.columns.values():
                if column.samples is None or len(column.samples) < min(COLUMN_SAMPLE_SIZE, table.size):
                    samples = self.query_column_sample_count(server, table.table_name, column.name)
                    column.samples = samples

        self.write_to_cache()

    @staticmethod
    def get_cache_path(db_name: str):
        cache_path = f"data/schema_cache/{db_name}.json"
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path

    def write_to_cache(self):
        cache_path = self.get_cache_path(self.get_search_path())
        with open(cache_path, "w") as f:
            f.write(jsonpickle.encode(self))

    @staticmethod
    def get_database(name: str, schema_file: str, fixed_query_path: Optional[Path] = None) -> "Database":
        cache_path = Database.get_cache_path(name)
        if cache_path.exists():
            with open(cache_path, "r") as f:
                json_result = f.read()
            result = jsonpickle.loads(json_result)
        else:
            with open(schema_file, "r") as f:
                schema_query = f.read()
            result = Database(load_schema(schema_query), fixed_query_path)
            result.write_to_cache()
        assert result.schema.name == name, f"expected name {name}, but found {result.schema.name} in schema file"
        return result

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import sqlparse

from src.util import AutoNumber


class Type(AutoNumber):
    Integer = ()
    Bigint = ()
    Decimal = ()
    Double = ()
    Varchar = ()
    CharArray = ()
    Text = ()
    Date = ()
    Time = ()

    def is_string_like(self) -> bool:
        return self in (Type.Varchar, Type.CharArray, Type.Text)


@dataclass
class Column:
    name: str
    type: Type
    size: Optional[float]  # size of one element in the column
    # statistics for numerical columns only
    distinct_count: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    samples: Optional[list]

    def can_have_statistics(self) -> bool:
        return self.type in {Type.Integer, Type.Decimal, Type.Bigint, Type.Double}

    def statistics_missing(self) -> bool:
        return self.can_have_statistics() and any(
            s is None for s in (self.distinct_count, self.min_val, self.max_val, self.samples)
        )

    def has_statistics(self) -> bool:
        return self.can_have_statistics() and not any(
            s is None for s in (self.distinct_count, self.min_val, self.max_val, self.samples)
        )

    def distinct_missing(self) -> bool:
        return self.distinct_count is None

    def min_max_missing(self) -> bool:
        return self.can_have_statistics() and not any(s is None for s in (self.min_val, self.max_val))

    def simple_print(self) -> str:
        name = self.name.replace('"', "")
        return (
            f'{{"name":"{name}","type":"{self.type.name}","byte_size":{self.size},"distinct_count":{self.distinct_count},'
            f"\"min_val\":{self.min_val if self.min_val is not None else 'null'},"
            f"\"max_val\":{self.max_val if self.max_val is not None else 'null'}}}"
        )


@dataclass
class Table:
    table_name: str
    columns: dict[str, Column]
    size: Optional[int]  # number of elements in table

    def simple_print(self) -> str:
        name = self.table_name.replace('"', "")
        return f"{{\"name\":\"{name}\",\"size\":{self.size},\"columns\":[{','.join(c.simple_print() for c in self.columns.values())}]}}"


# mapping from column name to tuple other table name and column name
JoinColumns = dict[str, [(str, str)]]


class TableParser:
    table_name: str
    columns: list[Column]

    def __init__(self):
        self.columns = []

    def get_table(self) -> Table:
        return Table(self.table_name, {c.name: c for c in self.columns}, None)

    @staticmethod
    def get_type(tokens: list[sqlparse.sql.Token]) -> tuple[list[sqlparse.sql.Token], Type]:
        res: Optional[Type]
        if tokens[0].match(sqlparse.tokens.Name.Builtin, ("integer", "int")):
            res = Type.Integer
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Name.Builtin, "BIGINT"):
            res = Type.Bigint
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Name, "varchar"):
            res = Type.Varchar
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Name, "char"):
            res = Type.CharArray
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Keyword, "character"):
            res = Type.Varchar
            tokens = tokens[2:]
        elif tokens[0].match(sqlparse.tokens.Name, "decimal"):
            res = Type.Decimal
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Name.Builtin, "date"):
            res = Type.Date
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Keyword, "time"):
            res = Type.Time
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Name.Builtin, "text"):
            res = Type.Text
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Keyword, "bytea"):
            res = Type.Text
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Name.Builtin, "float"):
            res = Type.Double
            tokens = tokens[1:]
        elif tokens[0].match(sqlparse.tokens.Name.Builtin, "double precision"):
            res = Type.Double
            tokens = tokens[1:]
        else:
            assert False, f"could not parse type {tokens[0]}"
        return tokens, res

    def get_size(self, tokens, type) -> tuple[list[sqlparse.sql.Token], Optional[float]]:
        known_sizes = {
            Type.Integer: 4,
            Type.Date: 4,
            Type.Time: 8,
            Type.Decimal: 8,
            Type.Double: 8,
            Type.Bigint: 8,
        }

        size: Optional[int] = None
        if type in known_sizes:
            size = known_sizes[type]

        if type in (Type.Varchar, Type.CharArray):
            size = int(tokens[1].normalized)
            tokens = tokens[3:]
        elif type == Type.Decimal:
            tokens = tokens[5:]
        return tokens, size

    @staticmethod
    def gobble_nullability(tokens: list[sqlparse.sql.Token]) -> list[sqlparse.sql.Token]:
        if len(tokens) > 0 and tokens[0].match(sqlparse.tokens.Keyword, "not null"):
            tokens = tokens[1:]
        return tokens

    @staticmethod
    def gobble_inline_key(tokens: list[sqlparse.sql.Token]) -> list[sqlparse.sql.Token]:
        if len(tokens) > 0 and tokens[0].match(sqlparse.tokens.Keyword, ("primary", "foreign")):
            tokens = tokens[2:]
        if len(tokens) > 0 and tokens[0].match(sqlparse.tokens.Keyword, ("primary key", "foreign key")):
            tokens = tokens[1:]
        return tokens

    def consume_column(self, tokens: list[sqlparse.sql.Token]) -> list[sqlparse.sql.Token]:
        name = tokens[0].normalized
        tokens, type = self.get_type(tokens[1:])
        tokens, size = self.get_size(tokens, type)

        tokens = self.gobble_nullability(tokens)
        tokens = self.gobble_inline_key(tokens)
        if len(tokens) > 0 and tokens[0].match(sqlparse.tokens.Punctuation, ","):
            tokens = tokens[1:]

        self.columns.append(Column(name, type, size, None, None, None, None))
        return tokens

    @staticmethod
    def gobble_key_constraint(tokens: list[sqlparse.sql.Token]) -> list[sqlparse.sql.Token]:
        for i, token in enumerate(tokens):
            if token.match(sqlparse.tokens.Punctuation, ")"):
                if len(tokens) > i + 1 and tokens[i + 1].match(sqlparse.tokens.Punctuation, ","):
                    return tokens[i + 2 :]
                else:
                    return tokens[i + 1 :]
        return []

    @staticmethod
    def gobble_until_next_column(tokens: list[sqlparse.sql.Token]) -> list[sqlparse.sql.Token]:
        for i, token in enumerate(tokens):
            if token.match(sqlparse.tokens.Punctuation, ","):
                return tokens[i + 1 :]
            if token.match(sqlparse.tokens.Punctuation, ")"):
                return tokens[i:]
        return []

    def add_table(self, query: str):
        stmt = sqlparse.parse(query)[0]
        assert stmt.get_type() == "CREATE"
        tokens = [t for t in stmt.tokens if not t.is_whitespace]
        identifier = tokens[2]
        assert type(identifier) == sqlparse.sql.Identifier
        self.table_name = str(identifier.tokens[-1])
        assert type(self.table_name) == str, f"incorrect table name type {type(self.table_name)}"
        assert type(tokens[3]) == sqlparse.sql.Parenthesis
        definition_tokens = [t for t in tokens[3].flatten() if not t.is_whitespace][1:][:-1]
        while len(definition_tokens) > 0:
            if definition_tokens[0].match(
                sqlparse.tokens.Keyword, ("primary", "primary key", "foreign", "foreign key", "references")
            ):
                definition_tokens = self.gobble_key_constraint(definition_tokens)
            elif definition_tokens[0].match(sqlparse.tokens.Keyword, ("default")):
                definition_tokens = self.gobble_until_next_column(definition_tokens)
            else:
                definition_tokens = self.consume_column(definition_tokens)


def common_suffix_length(str1, str2):
    i = 1
    while i <= min(len(str1), len(str2)) and str1[-i] == str2[-i]:
        i += 1
    return i - 1


similar_column_name_blacklist = {
    "name",
    "md5sum",
    "note",
}
similar_column_suffix_blacklist = {
    "name",
    "comment",
}
similar_column_type_blacklist = {
    Type.Decimal,
}


def columns_are_similar(c1: Column, c2: Column) -> bool:
    if c1.name in similar_column_name_blacklist and c2.name in similar_column_name_blacklist:
        return False
    if c1.type in similar_column_type_blacklist:
        return False
    for suff in similar_column_suffix_blacklist:
        if c1.name.endswith(suff):
            return False
    suffix_len = common_suffix_length(c1.name, c2.name)
    return c1.type == c2.type and suffix_len >= 3 and suffix_len >= 2 / 3 * max(len(c1.name), len(c2.name))


def collect_join_columns(tables: dict[str, Table]) -> dict[str, JoinColumns]:
    result = {name: {c_n: [] for c_n in table.columns} for name, table in tables.items()}
    table_list = [e for e in tables.items()]
    for i1, (name, table) in enumerate(table_list):
        columns_1 = [c for c in table.columns.values()]
        for i2 in range(i1):
            (other_name, other_table) = table_list[i2]
            for c1 in columns_1:
                for c2 in other_table.columns.values():
                    if columns_are_similar(c1, c2):
                        result[name][c1.name].append((other_name, c2.name))
                        result[other_name][c2.name].append((name, c1.name))
    result = {n: {c: l for c, l in t.items() if l != []} for n, t in result.items()}
    return result


@dataclass
class Schema:
    tables: dict[str, Table]
    join_columns: dict[str, JoinColumns]
    name: str

    def quote_table_name(self, table_name: str) -> str:
        if table_name not in self.tables and f'"{table_name}"' in self.tables:
            return f'"{table_name}"'
        elif table_name in self.tables:
            return table_name
        else:
            assert False, f"could not find table {table_name} in {self.name}"

    def get_table_scan_size(self, table_name: str) -> int:
        return self.tables[table_name].size

    def simple_print(self) -> str:
        return f"{{\"name\":\"{self.name}\",\"tables\":[{','.join(t.simple_print() for t in self.tables.values())}]}}"

    def __repr__(self) -> str:
        return f"{self.name}"


def load_schema(query: str):
    name_found = False
    name = ""
    tables = {}
    for table_query in sqlparse.split(query):
        # gobble comment lines
        table_query = "\n".join(line for line in table_query.splitlines() if not line.lstrip().startswith("--"))
        if len(table_query) == 0:
            # empty query
            continue
        if table_query.lower().lstrip().startswith("create schema"):
            # get schema name
            match = re.search(r'create\s+schema\s+"?(\w+)"?\s*;', table_query.lower().lstrip(), re.I)
            assert match, "failed to parse create schema"
            name = table_query.lstrip()[match.start(1) : match.end(1)]
            assert name.lower() == match.group(1), f"name extraction failed for {table_query.lstrip()}"
            name_found = True
            continue
        if table_query.lower().lstrip().startswith("drop table"):
            continue
        parser = TableParser()
        parser.add_table(table_query)
        table = parser.get_table()
        tables[table.table_name] = table
    join_columns = collect_join_columns(tables)
    assert name_found, f"could not find name of schema in query:\n{query}"
    return Schema(tables, join_columns, name)


def main():
    path = Path("../benchmark_setup/schemata")
    schema_filenames = [file for file in path.iterdir() if file.is_file() and file.name.endswith("schema.sql")]
    schema_filenames.sort()
    for filename in schema_filenames:
        with open(filename, "r") as f:
            query = f.read()
        load_schema(query)


if __name__ == "__main__":
    main()

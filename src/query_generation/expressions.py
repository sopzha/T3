from enum import Enum

import numpy as np

from src.database_manager import DatabaseManager
from src.query_generation.query_structures import IntermediateResult, BindingTable, BindingColumn
from src.schemata import Table
from src.util import AutoNumber


class ExpressionType(AutoNumber):
    Compare = ()
    Like = ()
    InExpression = ()
    Between = ()
    Or = ()


class Compare(Enum):
    Equal = "="
    Unequal = "<>"
    Less = "<"
    LessEq = "<="
    Greater = ">"
    GreaterEq = ">="


def get_random_column_value(column: BindingColumn) -> float:
    c_min = column.column.min_val
    c_max = column.column.max_val
    return c_min + (c_max - c_min) * np.random.random()


def get_random_not() -> str:
    if np.random.random() < 0.5:
        return ""
    else:
        return "NOT "


def sample_compare(input: IntermediateResult) -> str:
    qualifying_columns = [c for c in input.columns if c.column.has_statistics()]
    column: BindingColumn = np.random.choice(qualifying_columns)
    compare: Compare = np.random.choice([c for c in Compare])
    value = None
    if compare in (Compare.Equal, Compare.Unequal):
        unique_samples = [e for e in set(column.column.samples) if e != "null" and e is not None]
        value = np.random.choice(unique_samples)
    else:
        value = get_random_column_value(column)
    return f"{column.to_string()} {compare.value} {value}"


def sample_like(input: IntermediateResult) -> str:
    qualifying_columns = [c for c in input.columns if c.column.type.is_string_like() and c.column.size > 1]
    column: BindingColumn = np.random.choice(qualifying_columns)
    pattern: str = np.random.choice(column.column.samples)
    pattern = pattern.strip()
    words = pattern.split()
    words = [w for w in words if len(w) > 2]
    pattern: str = np.random.choice(words)
    split = np.random.randint(1, len(pattern) - 1)
    w1 = pattern[:split]
    w2 = pattern[split:]
    w1 = w1.replace("'", "''")
    w2 = w2.replace("'", "''")
    pattern = f"%{w1}%{w2}%"
    not_str = get_random_not()
    return f"{column.to_string()} {not_str}LIKE '{pattern}'"


def sample_to_string(sample) -> str:
    if isinstance(sample, str):
        sample = sample.replace("'", "''")
        return f"'{sample}'"
    else:
        return str(sample)


def sample_in_expression(input: IntermediateResult) -> str:
    qualifying_columns = [c for c in input.columns if c.column.samples is not None and len(c.column.samples) > 1]
    column: BindingColumn = np.random.choice(qualifying_columns)
    unique_samples = [e for e in set(column.column.samples) if e != "null" and e is not None]
    if len(unique_samples) == 0:
        raise ValueError
    n_samples = np.random.geometric(p=0.1)
    n_samples = min(n_samples, len(unique_samples))
    samples = np.random.choice(unique_samples, n_samples, replace=False)
    samples_str = ", ".join(sample_to_string(s) for s in samples)
    not_str = get_random_not()
    return f"{column.to_string()} {not_str}IN ({samples_str})"


def sample_between(input: IntermediateResult) -> str:
    qualifying_columns = [c for c in input.columns if c.column.has_statistics()]
    column: BindingColumn = np.random.choice(qualifying_columns)
    v1 = get_random_column_value(column)
    v2 = get_random_column_value(column)
    v1, v2 = min(v1, v2), max(v1, v2)
    not_str = get_random_not()
    return f"{column.to_string()} {not_str}BETWEEN {v1} AND {v2}"


def sample_or(input: IntermediateResult) -> str:
    return f"(({sample_expression(input, False)}) OR ({sample_expression(input, False)}))"


def sample_expression(input: IntermediateResult, allow_or: bool = True) -> str:
    while True:
        try:
            sample_dict = {
                ExpressionType.Compare: sample_compare,
                ExpressionType.Like: sample_like,
                ExpressionType.InExpression: sample_in_expression,
                ExpressionType.Between: sample_between,
                ExpressionType.Or: sample_or,
            }
            types = [t for t in ExpressionType if (t != ExpressionType.Or or allow_or)]
            type = np.random.choice(types)
            return sample_dict[type](input)
        except ValueError:
            pass


def main():
    db = DatabaseManager.get_database("tpchSf1")
    for _ in range(100):
        tables = list(db.schema.tables.values())
        start_table: Table = np.random.choice(tables)
        print(sample_expression(BindingTable(start_table, "t0").to_intermediate_result()))


if __name__ == "__main__":
    main()

from enum import Enum
from pathlib import Path


class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__)  # note no + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


def filter_unique_unhashable(original_list: list) -> list:
    unique_list = []
    for item in original_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def fifo_cache(func):
    cache = []
    limit = 100

    def wrapper(*args, **kwargs):
        nonlocal cache
        key = (args, frozenset(kwargs.items()))

        for c in cache:
            if key == c[0]:
                return c[1]

        result = func(*args, **kwargs)
        cache.append((key, result))
        if len(cache) > limit:
            cache = cache[1:]
            print(f"Evicting cache for {func.__name__}")

        return result

    return wrapper


def get_lines(file: Path) -> list[str]:
    with open(file, "r") as fd:
        return fd.readlines()


def _rm_dir_rec(path: Path):
    for sub in path.iterdir():
        if sub.is_dir():
            _rm_dir_rec(sub)
        else:
            sub.unlink()
    path.rmdir()


def rm_rec(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        _rm_dir_rec(path)
    else:
        path.unlink()

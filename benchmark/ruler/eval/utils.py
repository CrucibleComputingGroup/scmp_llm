import json
from collections.abc import Generator
from typing import Any


def dump_jsonl(fname: str, data: list[dict[str, Any]]) -> None:
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def iter_jsonl(fname: str, cnt: int | None = None) -> Generator[dict[str, Any], None, None]:
    i = 0
    with open(fname) as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def load_data(fname: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(fname))

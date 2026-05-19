import json
from typing import Any


def dump_jsonl(fname: str, data: list[dict[str, Any]]) -> None:
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

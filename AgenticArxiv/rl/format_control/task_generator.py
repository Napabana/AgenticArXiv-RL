"""Deterministic tasks. Test inputs and rule combinations are fixed before sampling."""
import json
import os
import random
import string
from collections import Counter

RULES = (
    "把输入 JSON 按给定顺序写成两行。第一行是 @ROW| 后接 key=value，各字段以 | 分隔。"
    "第二行只写 @END。布尔 true/false 写 YES/NO，null 写 NA，整数用十进制，空字符串留空。"
    "字符串原样保留，但反斜杠、竖线和等号前各加一个反斜杠。"
    "不要输出说明、代码围栏、前后空白或末尾换行。"
)
PROFILES = ("text int", "text bool", "unicode null", "empty bool int",
            "escaped int", "escaped bool null", "unicode bool int", "empty escaped null")
UNSEEN = ("escaped unicode int", "empty bool null int")
SIZES = {"train": 24, "dev": 30, "test": 120}


def prompt(row):
    order = ", ".join(row["order"])
    record = json.dumps(row["record"], ensure_ascii=False)
    rules = RULES
    version = os.getenv("FORMAT_PROMPT_VERSION", "rules")
    if version == "example":
        rules += ('\n示例输入：{"flag":true,"code":"az"}\n示例字段顺序：code, flag'
                  '\n示例输出（恰好两行）：\n@ROW|code=az|flag=YES\n@END'
                  '\n现在转换下面的新记录。键和值之间不加空格，每个键取它自己在输入中的值。')
    elif version != "rules":
        raise ValueError(f"Unknown format prompt version: {version}")
    return f"{rules}\n字段顺序：{order}\n输入：{record}"


def dataset(split, seed=42):
    rng = random.Random(seed + list(SIZES).index(split) * 10000)
    profiles = PROFILES + (UNSEEN if split == "test" else ())
    rows = []
    seen = set()
    while len(rows) < SIZES[split]:
        profile = profiles[len(rows) % len(profiles)].split()
        keys = rng.sample(["unit", "tag", "name", "active", "zone", "mark", "batch", "label"], len(profile))
        word = "".join(rng.choices(string.ascii_lowercase, k=4))
        values = {"text": word, "int": rng.randrange(-888, 999), "bool": rng.choice([False, True]),
                  "unicode": rng.choice(["星", "河", "风", "雪"]) + word, "null": None, "empty": "",
                  "escaped": word + rng.choice(["|", "\\", "=", "\\|", "=\\"]) + "x"}
        items = list(zip(keys, [values[k] for k in profile], strict=True))
        rng.shuffle(items)
        record = dict(items)
        identity = json.dumps(record, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        order = list(keys)
        rng.shuffle(order)
        if order == list(record):
            order = order[1:] + order[:1]
        rows.append(dict(id=f"{split}-{len(rows):03}", record=record, order=order,
                         atoms=dict(zip(keys, profile)), combination="+".join(sorted(profile))))
    return rows


def coverage(rows):
    return dict(instances=len(rows), combinations=dict(Counter(r["combination"] for r in rows)),
                atomic_fields=dict(Counter(a for r in rows for a in r["atoms"].values())),
                reordered=sum(r["order"] != list(r["record"]) for r in rows))

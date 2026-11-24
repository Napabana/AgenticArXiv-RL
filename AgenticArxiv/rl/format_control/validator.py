"""A canonical writer and an independent strict parser for the two-line wire format."""
import re


def value_text(value):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value)


def expected(row):
    parts = []
    for key in row["order"]:
        value = value_text(row["record"][key])
        escaped = value.replace("\\", "\\\\").replace("|", "\\|").replace("=", "\\=")
        parts.append(f"{key}={escaped}")
    return "@ROW|" + "|".join(parts) + "\n@END"


def parse(text):
    if not text.startswith("@ROW|") or not text.endswith("\n@END") or text.count("\n") != 1:
        raise ValueError("envelope")
    body = text[5:-5]
    pairs = {}
    pos = 0
    while True:
        key = re.match(r"[a-z]+=", body[pos:])
        if key is None:
            raise ValueError("field_syntax")
        name = key.group()[:-1]
        if name in pairs:
            raise ValueError("duplicate_key")
        pos += key.end()
        chars = []
        while pos < len(body) and body[pos] != "|":
            ch = body[pos]
            if ch == "\\":
                pos += 1
                if pos == len(body) or body[pos] not in "\\|=":
                    raise ValueError("escape")
                ch = body[pos]
            elif ch == "=" or ch == "\r":
                raise ValueError("field_syntax")
            chars.append(ch)
            pos += 1
        pairs[name] = "".join(chars)
        if pos == len(body):
            return pairs
        pos += 1


def score(text, row):
    try:
        parsed = parse(text)
        error = None
    except ValueError as exc:
        parsed = {}
        error = str(exc)
    fields = {k: parsed.get(k) == value_text(v) for k, v in row["record"].items()}
    order_ok = list(parsed) == row["order"]
    result = int(order_ok and all(fields.values()))
    if result != int(text == expected(row)):
        raise RuntimeError("Parser and independent exact-match oracle disagree")
    return dict(strict=result, order=order_ok, fields=fields,
                header=text.startswith("@ROW|"), footer=text.endswith("\n@END"), error=error or
                ("fields_or_order" if not order_ok else "wrong_value" if not result else None),
                extra_text=("@ROW|" in text and "\n@END" in text and
                            (not text.startswith("@ROW|") or not text.endswith("\n@END"))))

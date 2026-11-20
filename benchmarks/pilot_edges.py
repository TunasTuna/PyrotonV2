"""Deterministic MBPP edge proposals from safe, literal calls in published tests."""

import ast
import re
from collections import defaultdict

KINDS = ("empty", "zero", "negative", "singleton", "wrong_type", "unit")
SEQUENCES = (str, bytes, list, tuple, dict, set)


def encode(value, depth=0):
    """Keep Python types intact without pickling or evaluating generated code."""
    if depth > 9:
        raise ValueError("Fixture nesting limit exceeded.")
    kind = type(value)
    if value is None:
        return ["none", None]
    if kind in (bool, int, str):
        return [kind.__name__, value]
    if kind is float:
        return ["float", repr(value)]
    if kind is complex:
        return ["complex", [repr(value.real), repr(value.imag)]]
    if kind is bytes:
        return ["bytes", value.hex()]
    if kind in (list, tuple, set, frozenset):
        if len(value) > 512:
            raise ValueError("Fixture collection limit exceeded.")
        items = [encode(item, depth + 1) for item in value]
        if kind in (set, frozenset):
            items.sort(key=repr)
        return [kind.__name__, items]
    if kind is dict:
        if len(value) > 512:
            raise ValueError("Fixture collection limit exceeded.")
        pairs = [[encode(key, depth + 1), encode(item, depth + 1)] for key, item in value.items()]
        return ["dict", pairs]
    raise ValueError(f"Unsupported fixture type: {kind.__name__}.")


def decode(record):
    kind, content = record
    if kind in ("none", "bool", "int", "str"):
        return content
    if kind == "float":
        return float(content)
    if kind == "complex":
        return complex(*map(float, content))
    if kind == "bytes":
        return bytes.fromhex(content)
    if kind == "dict":
        return {decode(key): decode(item) for key, item in content}
    constructors = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
    return constructors[kind](map(decode, content))


def literal(node):
    """Handle literals and simple builtin constructors with no code execution."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        constructors = {"list": list, "tuple": tuple, "set": set, "dict": dict}
        if node.func.id in constructors and len(node.args) <= 1:
            return constructors[node.func.id](*[literal(arg) for arg in node.args])
        if node.func.id == "range" and 1 <= len(node.args) <= 3:
            args = [literal(arg) for arg in node.args]
            value = range(*args)
            if len(value) <= 100:
                return list(value)
    return ast.literal_eval(node)


def literal_calls(task):
    names = {task["entry_point"]}
    sources = task["checks"] + task.get("challenge_checks", [])
    seen = set()
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in names:
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(not kw.arg for kw in node.keywords):
                continue
            try:
                args = tuple(literal(arg) for arg in node.args)
                kwargs = {kw.arg: literal(kw.value) for kw in node.keywords}
                key = repr((encode(args), encode(kwargs)))
            except (TypeError, ValueError, SyntaxError, OverflowError):
                continue
            if key not in seen:
                seen.add(key)
                yield args, kwargs


def singleton(value):
    if isinstance(value, dict):
        key = next(iter(value))
        return {key: value[key]}
    if isinstance(value, set):
        return {min(value, key=repr)}
    return value[:1]


def scalar_variants(value, negative, zero):
    kind = type(value)
    if kind in (int, float):
        if zero:
            yield "zero", kind(0)
        if negative:
            yield "negative", kind(-1) if kind is int else -0.5
        yield "unit", kind(1)
    elif kind is bool:
        yield ("zero" if value else "unit"), not value


def variants(value, rules, depth=0):
    yield from scalar_variants(value, rules["negative"], rules["zero"])
    if type(value) not in SEQUENCES:
        return
    if rules["empty"]:
        yield "empty", type(value)()
    if value and rules["singleton"]:
        yield "singleton", singleton(value)
    if depth < 2 and type(value) in (list, tuple) and value and not rules["sorted"]:
        for index in range(min(2, len(value))):
            child = value[index]
            for category, replacement in variants(child, rules, depth + 1):
                if category not in ("zero", "negative"):
                    continue
                clone = list(value)
                clone[index] = replacement
                yield category, type(value)(clone)


def edge_rules(task):
    text = task.get("description", "").lower()
    domain = r"\b(?:given|takes?|accepts?|of|must be|is|are|contains?|consists? of)\b[^.\n]{0,20}\b"
    numeric = r"(?:integers?|numbers?|values?|floats?|amount)"
    nonnegative = bool(re.search(domain + r"non[\s-]?negative\s+" + numeric, text))
    positive = bool(re.search(domain + r"positive\s+" + numeric, text))
    nonempty = bool(re.search(r"\bnon[\s-]?empty\b|\bat least (?:one|[2-9]|two|three)\b", text))
    two_or_more = bool(re.search(r"\bat least (?:[2-9]|two|three)\b", text))
    wrong_type = bool(re.search(
        r"\braise[sd]?\b[^.\n]{0,58}\btypeerror\b"
        r"|\binvalid (?:input )?type\b|\bwrong (?:input )?type\b"
        r"|\b(?:is|are) not (?:an? )?(?:int(?:eger)?|str(?:ing)?|list|tuple|bool(?:ean)?)\b"
        r"|\breject\b[^.\n]{0,35}\b(?:booleans?|bool|non[\s-]?integers?|non[\s-]?strings?)\b",
        text,
    ))
    mutation_required = bool(re.search(
        r"\bin[\s-]place\b|\bmodify (?:the )?(?:original|input)\b|\bmutate (?:the )?(?:input|argument)\b", text
    ))
    return {
        "negative": not positive and not nonnegative, "zero": not positive,
        "empty": not nonempty, "singleton": not two_or_more,
        "sorted": bool(re.search(r"\bsorted\b|\bnondecreasing\b|\bincreasing order\b", text)),
        "wrong_type": wrong_type, "mutation_required": mutation_required,
        "expected_exceptions": sorted(set(re.findall(r"\b(?:TypeError|ValueError|IndexError|KeyError|OverflowError|ZeroDivisionError)\b", task.get("description", "")))),
        "invalid_sentinel": bool(re.search(r"\breturn(?:s|ed)?\s+(?:none|false|-1)\b", text)),
        "aligned": bool(re.search(r"\b(?:same|equal|identical) (?:size|length)\b", text)),
    }


def proposals(task, maximum=24):
    rules = edge_rules(task)
    seeds = list(literal_calls(task))
    seen = {repr((encode(args), encode(kwargs))) for args, kwargs in seeds}
    buckets = defaultdict(list)

    def record(category, args, kwargs):
        if len(buckets[category]) >= 8:
            return
        encoded_args, encoded_kwargs = encode(args), encode(kwargs)
        identity = repr((encoded_args, encoded_kwargs))
        if identity not in seen and len(identity) <= 2800:
            seen.add(identity)
            buckets[category].append({"kind": category, "args": encoded_args, "kwargs": encoded_kwargs})

    for args, kwargs in seeds[:5]:
        slots = [(index, value) for index, value in enumerate(args)] + list(kwargs.items())
        aligned = rules["aligned"] and sum(type(value) in SEQUENCES for _, value in slots) >= 2
        for slot, value in slots:
            examples = list(variants(value, rules))
            if aligned:
                examples = [(kind, replacement) for kind, replacement in examples if kind not in ("empty", "singleton")]
            if rules["wrong_type"]:
                examples.extend(("wrong_type", item) for item in (None, "__not_a_number__" if not isinstance(value, str) else 17))
                if type(value) is int:
                    examples.append(("wrong_type", True))
            for category, replacement in examples:
                if isinstance(slot, int):
                    record(category, args[:slot] + (replacement,) + args[slot + 1:], kwargs)
                else:
                    record(category, args, {**kwargs, slot: replacement})
        if aligned:
            for category, transform in (("empty", lambda item: type(item)()), ("singleton", singleton)):
                if not rules[category]:
                    continue
                if category == "singleton" and any(not item for _, item in slots if type(item) in SEQUENCES):
                    continue
                record(category, tuple(transform(item) if type(item) in SEQUENCES else item for item in args),
                       {key: transform(item) if type(item) in SEQUENCES else item for key, item in kwargs.items()})

    candidates = []
    for i in range(8):
        for kind in KINDS:
            if i < len(buckets[kind]):
                candidates.append(buckets[kind][i])
    return candidates[:maximum], rules

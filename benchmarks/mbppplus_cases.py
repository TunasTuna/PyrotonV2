"""Reconstruct typed MBPP+ inputs and its test subset's result comparisons."""

SOURCE = "MBPP+ v0.2.0 test subset"

# Type restoration follows EvalPlus's mbpp_deserialize_inputs:
# https://github.com/evalplus/evalplus/blob/master/evalplus/data/mbpp.py
TUPLE_ARGUMENTS = {2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429, 470,
                   560, 579, 596, 616, 630, 726, 740, 744, 809}
LIST_OF_TUPLES = {63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780}
FIRST_LIST_OF_TUPLES = {75, 413, 444, 753}
FIRST_TUPLE = {250, 405, 446, 617, 720, 763, 808, 307}
TUPLE_OF_TUPLES = {259, 401, 445}
RECURSIVE_TUPLES = {580, 615, 791}


def tuple_tree(value):
    return tuple(map(tuple_tree, value)) if isinstance(value, list) else value


def decode_case(task_id, args):
    number = int(task_id.rsplit("/", 1)[-1])
    if number in TUPLE_ARGUMENTS:
        return [tuple(argument) for argument in args]
    if number in LIST_OF_TUPLES:
        return [list(map(tuple, argument)) for argument in args]
    if number in FIRST_LIST_OF_TUPLES:
        return [list(map(tuple, args[0])), *args[1:]]
    if number in FIRST_TUPLE:
        return [tuple(args[0]), *args[1:]]
    if number in TUPLE_OF_TUPLES:
        return [tuple(map(tuple, argument)) for argument in args]
    if number in RECURSIVE_TUPLES:
        return tuple_tree(args)
    if number in {106, 750}:
        return [args[0], tuple(args[1])]
    if number == 115:
        return [[set(value) if isinstance(value, list) and value else {} for value in args[0]]]
    if number == 124:
        return float(args[0]), complex(args[1])
    if number == 252:
        return [complex(args[0])]
    if number == 278:
        return [tuple(tuple(value) if isinstance(value, list) else value for value in args[0])]
    if number == 722:
        return [{key: tuple(value) for key, value in args[0].items()}, *args[1:]]
    return args


# The complete oracle registry is tracked so unsupported exceptions fail closed.
# https://github.com/evalplus/evalplus/blob/master/evalplus/eval/_special_oracle.py
SET_OUTPUTS = {"similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly",
               "larg_nnum", "intersection_array", "find_dissimilar", "Diff"}
PRESENCE_OUTPUTS = {"check_str", "text_match_three", "text_starta_endb"}
UNSUPPORTED = {"sum_div", "surface_Area", "digit_distance_nums"}


def oracle_for(entry_point):
    if entry_point in SET_OUTPUTS:
        return "set"
    if entry_point in PRESENCE_OUTPUTS:
        return "presence"
    if entry_point == "are_equivalent":
        return "any-output"
    if entry_point in UNSUPPORTED:
        return "unsupported"
    return "exact-with-float-tolerance"


# This definition is appended to the WASI worker program, not evaluated in the
# process reading model generations. Scalar and nested Python outputs are
# supported. The scorer reports Pyroton's sandbox result, rather than a score
# produced by the upstream EvalPlus command.
COMPARATOR = """
def mbpp_approximate(got, wanted, atol):
    if isinstance(wanted, (list, tuple)):
        return len(got) == len(wanted) and all(
            mbpp_approximate(left, right, atol) for left, right in zip(got, wanted)
        )
    if got == wanted:
        return True
    return abs(got - wanted) <= atol + 1e-7 * abs(wanted)

def mbpp_equal(got, wanted, atol, oracle):
    if oracle == 'set':
        return set(got) == set(wanted)
    if oracle == 'presence':
        return (got if isinstance(got, bool) else got is not None) == (wanted is not None)
    if got == wanted:
        return True
    if not atol and (isinstance(wanted, float) or
                     isinstance(wanted, (list, tuple)) and wanted and
                     all(isinstance(value, float) for value in wanted)):
        atol = 1e-6
    return bool(atol) and type(got) is type(wanted) and mbpp_approximate(got, wanted, atol)
"""


def test_program(row):
    oracle = oracle_for(row["entry_point"])
    if oracle != row["oracle"] or oracle in ("unsupported", "any-output"):
        raise ValueError(f"Unsupported or mismatched MBPP+ oracle for {row['id']}.")
    inputs = [decode_case(row["id"], args) for args in (*row["base_input"], *row["plus_input"])]
    return COMPARATOR + f"""
# repr() uses these names for non-finite float and complex fixtures.
inf, nan = math.inf, math.nan
infj, nanj = complex(0, inf), complex(0, nan)
reference = {{}}
exec(compile({row['reference']!r}, '<reference>', 'exec'), reference)
expected_fn = reference[{row['entry_point']!r}]
for index, args in enumerate({inputs!r}, 1):
    expected = expected_fn(*copy.deepcopy(args))
    got = fn(*copy.deepcopy(args))
    assert mbpp_equal(got, expected, {row['atol']!r}, {oracle!r}), f'case {{index}}: expected {{repr(expected)[:150]}}, got {{repr(got)[:150]}}'
"""

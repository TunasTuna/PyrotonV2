"""Contract checks for a conservative, explicitly reviewed subset of MBPP edges."""

from bisect import bisect_left
from collections import Counter
import math
import re


def longest_increasing(values):
    tails = []
    for value in values:
        index = bisect_left(tails, value)
        if index == len(tails):
            tails.append(value)
        else:
            tails[index] = value
    return len(tails)


def capital_endpoints(text):
    if len(text) < 2:
        return text.upper()
    return text[0].upper() + text[1:-1] + text[-1].upper()


def swap_endpoints(values):
    return values if len(values) == 1 else [values[-1], *values[1:-1], values[0]]


def valid_column(rows, index):
    return bool(rows) and index >= 0 and all(index < len(row) for row in rows)


def mixed_tuple(values):
    return any(isinstance(item, tuple) for item in values) and any(not isinstance(item, tuple) for item in values)


def nested_frequency(values):
    return dict(Counter(item for value in values for item in (value if isinstance(value, tuple) else (value,))))


def max_non_adjacent(values):
    selected, skipped = 0, 0
    for value in values:
        selected, skipped = skipped + value, max(selected, skipped)
    return max(selected, skipped)


def always(*args):
    return True


def same_length(left, right):
    return len(left) == len(right)


def odd_subarray_sum(values):
    return sum(value * (((i + 1) * (len(values) - i) + 1) // 2) for i, value in enumerate(values))


def replace_punctuation(text):
    return re.sub(r"[ ,.]", ":", text)


def alphanumeric(text):
    return "".join(char for char in text if char.isalnum())


# Each oracle covers only the stated domain and the published return convention.
# Eligibility never follows from the behavior of the reference alone.
REVIEWED = {
    512: ("Count occurrences in a mixed nested tuple with the published one-level shape.",
          mixed_tuple, nested_frequency),
    520: ("Least common multiple of a nonempty integer array.", bool, lambda values: math.lcm(*values)),
    526: ("Capitalize first and last letters of each word, including a one-letter word.",
          lambda text: not text or text.isascii() and text.isalpha(), capital_endpoints),
    548: ("The longest increasing subsequence of the empty sequence has length zero.", always, longest_increasing),
    551: ("Extract a nonnegative column that exists in every row.", valid_column,
          lambda rows, index: [row[index] for row in rows]),
    591: ("Swap first and last in a nonempty list. A one-item list has the same endpoint twice.",
          bool, swap_endpoints),
    611: ("Take the maximum of an existing nonnegative column in nonempty input.", valid_column,
          lambda rows, index: max(row[index] for row in rows)),
    655: ("Sum the fifth powers of the first n natural numbers for positive n.", lambda n: 1 <= n <= 100,
          lambda n: sum(value**5 for value in range(1, n + 1))),
    665: ("Move the first element to the end of a nonempty list.", bool, lambda values: values[1:] + values[:1]),
    692: ("The final two digits of n factorial, including 0 factorial.", lambda n: 0 <= n <= 100,
          lambda n: math.factorial(n) % 100),
    710: ("Return the first and last values of a nonempty tuple in the published tuple format.",
          bool, lambda values: (values[0], values[-1])),
    741: ("All characters of a nonempty string are the same exactly when it has one distinct character.",
          bool, lambda text: len(set(text)) == 1),
    803: ("A square is an integer times itself. Zero is a square and negatives are outside the squares.",
          always, lambda n: n >= 0 and math.isqrt(n)**2 == n),
    827: ("Sum an existing nonnegative column in nonempty input.", valid_column,
          lambda rows, index: sum(row[index] for row in rows)),
    887: ("Integer parity is defined for zero and negative integers.", always, lambda n: bool(n & 1)),
    889: ("Reverse each inner list and preserve the outer list order.", always,
          lambda rows: [row[::-1] for row in rows]),
    891: ("Count decimal digits of nonnegative integers. Zero occupies one digit.",
          lambda a, b: a >= 0 and b >= 0, lambda a, b: len(str(a)) == len(str(b))),
    895: ("Take the maximum sum of nonadjacent elements when a positive element exists.",
          lambda values: any(value > 0 for value in values), max_non_adjacent),
    913: ("A string ends with a number exactly when its final ASCII character is a digit.",
          lambda text: text.isascii(), lambda text: bool(text and text[-1].isdigit())),
    925: ("The product over the tuple includes its signed numbers and has empty product one.", always,
          math.prod),
}

REVIEWED.update({
    513: ("Insert the supplied string after each tuple element.", always,
          lambda items, value: [field for item in items for field in (item, value)]),
    532: ("Strings are permutations when their character multisets agree.", always,
          lambda a, b: Counter(a) == Counter(b)),
    542: ("Replace each space, comma, or dot with a colon.", always, replace_punctuation),
    546: ("Return the last position of a character that occurs in the string.",
          lambda text, char: len(char) == 1 and char in text, lambda text, char: text.rfind(char)),
    563: ("Extract the content of complete quoted strings.", lambda text: text.count('"') % 2 == 0,
          lambda text: re.findall(r'"([^"]*)"', text)),
    565: ("Split the entire string into its characters.", always, list),
    573: ("Multiply the distinct numbers in the list.", always, lambda values: math.prod(set(values))),
    582: ("An empty dictionary has no keys.", lambda value: isinstance(value, dict), lambda value: not value),
    587: ("Convert the list to a tuple in the same order.", always, tuple),
    588: ("Subtract the minimum from the maximum in a nonempty array.", bool,
          lambda values: max(values) - min(values)),
    600: ("Integer even parity includes zero and negative integers.", lambda n: type(n) is int,
          lambda n: n & 1 == 0),
    604: ("Reverse word order for the published plain space-separated format.",
          lambda text: text == " ".join(text.split()), lambda text: " ".join(reversed(text.split()))),
    606: ("Convert degrees to radians, including negative angles and zero.", always, math.radians),
    625: ("Exchange the first and last positions of a nonempty list.", bool, swap_endpoints),
    637: ("No profit or loss means sale price equals the original cost.",
          lambda cost, price: cost >= 0 and price >= 0, lambda cost, price: cost == price),
    645: ("Multiply entries in an existing nonnegative column.", valid_column,
          lambda rows, index: math.prod(row[index] for row in rows)),
    651: ("Check whether the elements of the second tuple are a subset of the first.", always,
          lambda container, items: set(items) <= set(container)),
    654: ("A rectangle with positive side lengths has perimeter twice their sum.",
          lambda a, b: a > 0 and b > 0, lambda a, b: 2 * (a + b)),
    672: ("Return the greatest of the three numbers.", always, max),
    676: ("Keep the alphanumeric characters in an ASCII string.", str.isascii, alphanumeric),
    682: ("Multiply corresponding elements of equal-length lists.", same_length,
          lambda a, b: [x * y for x, y in zip(a, b)]),
    688: ("Compute the magnitude of the complex number.", always, math.hypot),
    690: ("Multiply each consecutive pair in the list.", always,
          lambda values: [a * b for a, b in zip(values, values[1:])]),
    696: ("Combine corresponding lists using the published concatenation format.", same_length,
          lambda a, b: [x + y for x, y in zip(a, b)]),
    697: ("Count the even integers in the list, including zero.", always,
          lambda values: sum(value % 2 == 0 for value in values)),
    703: ("Check dictionary key membership.", always, lambda values, key: key in values),
    716: ("A rhombus with positive side length has perimeter four times that length.", lambda x: x > 0,
          lambda x: 4 * x),
    718: ("Take every second item, starting at index zero as in the published examples.", always,
          lambda values: values[::2]),
    726: ("Multiply each adjacent pair of tuple elements.", always,
          lambda values: tuple(a * b for a, b in zip(values, values[1:]))),
    727: ("Keep letters and numbers in an ASCII string.", str.isascii, alphanumeric),
    728: ("Sum corresponding elements of equal-length lists.", same_length,
          lambda a, b: [x + y for x, y in zip(a, b)]),
    729: ("Add corresponding elements of equal-length lists.", same_length,
          lambda a, b: [x + y for x, y in zip(a, b)]),
    732: ("Replace every occurrence of a space, comma, or dot.", always, replace_punctuation),
    744: ("Check whether a tuple contains None.", always, lambda values: any(item is None for item in values)),
    766: ("List every pair of consecutive items in order.", always,
          lambda values: list(zip(values, values[1:]))),
    782: ("Sum all odd-length contiguous subarrays.", always, odd_subarray_sum),
    791: ("Remove nested tuples and keep the other entries in order.", always,
          lambda values: tuple(item for item in values if not isinstance(item, tuple))),
    796: ("Sum the numeric values of the dictionary.", always, lambda values: sum(values.values())),
    798: ("Sum all array elements.", always, sum),
    805: ("Choose the list with the highest sum where the maximum is unique.",
          lambda values: bool(values) and sum(sum(row) == max(map(sum, values)) for row in values) == 1,
          lambda values: max(values, key=sum)),
    811: ("Check whether the two lists agree in elements and order.", always, lambda a, b: a == b),
    813: ("Return the length of the string in characters.", always, len),
    815: ("Sort values from zero through two when the declared length matches the array.",
          lambda values, n: n == len(values) and set(values) <= {0, 1, 2}, lambda values, n: sorted(values)),
    817: ("Use nonzero divisors when checking divisibility by either one.", lambda values, a, b: a != 0 and b != 0,
          lambda values, a, b: [value for value in values if value % a == 0 or value % b == 0]),
    818: ("Count lowercase letters in a string.", always, lambda text: sum(char.islower() for char in text)),
    825: ("Read each requested position when all indices are valid and nonnegative.",
          lambda values, indices: all(0 <= index < len(values) for index in indices),
          lambda values, indices: [values[index] for index in indices]),
    828: ("Count letters, digits, and the remaining printable ASCII characters.",
          lambda text: all(char.isascii() and char.isprintable() and not char.isspace() for char in text),
          lambda text: (sum(c.isalpha() for c in text), sum(c.isdigit() for c in text),
                        sum(not c.isalnum() for c in text))),
    833: ("Return the dictionary's keys in iteration order, as in the published examples.", always, list),
    845: ("Use the decimal digit count of a nonnegative integer factorial.", lambda x: 0 <= x <= 100,
          lambda x: len(str(math.factorial(x)))),
    848: ("Use positive trapezium dimensions.", lambda a, b, h: a > 0 and b > 0 and h > 0,
          lambda a, b, h: (a + b) * h / 2),
    858: ("Square the count of lists.", lambda values: all(isinstance(row, list) for row in values),
          lambda values: len(values)**2),
    868: ("Count the characters in the last word when there is a word.", lambda text: bool(text.split()),
          lambda text: len(text.split()[-1])),
    870: ("Sum the positive entries, excluding zero and negatives.", always,
          lambda values: sum(value for value in values if value > 0)),
    876: ("The least common multiple of two positive integers uses only positive inputs.",
          lambda a, b: a > 0 and b > 0, math.lcm),
    877: ("Return the characters in sorted order.", always, lambda text: "".join(sorted(text))),
    883: ("Use nonzero divisors when checking divisibility by both.", lambda values, a, b: a != 0 and b != 0,
          lambda values, a, b: [value for value in values if value % a == 0 and value % b == 0]),
    896: ("Sort tuples by their last value when all tuples are nonempty and keys are distinct.",
          lambda values: all(values) and len({row[-1] for row in values}) == len(values),
          lambda values: sorted(values, key=lambda row: row[-1])),
    901: ("Find the least positive multiple of the integers one through positive n.", lambda n: 1 <= n <= 100,
          lambda n: math.lcm(*range(1, n + 1))),
    904: ("A value is even when it is an integer divisible by two.", lambda x: type(x) is int,
          lambda x: x % 2 == 0),
    919: ("Multiply the entries of the list.", always, math.prod),
    929: ("Count the occurrences of the supplied tuple value.", always, lambda values, item: values.count(item)),
    945: ("Convert the tuple of hashable elements to a set.", always, set),
    959: ("The mean of a nonempty list divides the sum by its length.", bool,
          lambda values: sum(values) / len(values)),
    966: ("Remove empty tuples and preserve the remaining items in order.", always,
          lambda values: [item for item in values if item != ()]),
})

FIXED_ORACLE_TASKS = {803, 889, 891}

KNOWN_DROPS = {
    517: "The largest positive element is undefined when the list contains no positive element.",
    531: "The description gives no sentinel for an unmakeable amount or a truncated coin array.",
    558: "The description does not define signed digit distance.",
    616: "Pointwise modulo has no specified alignment for unequal tuple lengths.",
    644: "The description does not define how position zero is handled. The reference duplicates input.",
    681: "A least prime divisor of zero is outside the scope of the usual positive-integer task.",
    695: "The comparison does not define an alignment for unequal tuple lengths.",
    775: "The reference also constrains even indices, which the description does not require.",
    790: "The reference treats zero inconsistently with even parity and imposes other index constraints.",
}

import os

def parse_int_list(string):
    """
    Parses a string of numbers and ranges into a list of integers. Ranges
    are separated by "-" and inclusive of both the start and end number.

    Example:
        parse_int_list("8,10-12") == [8,10,11,12]
    """
    integers = []
    for substring in string.split(","):
        if "-" in substring:
            left, right = string.split("-")
            left_val = int(left.strip())
            right_val = int(right.strip())
            integers.extend(range(left_val, right_val + 1))
        else:
            integers.append(int(substring.strip()))
    return integers

def str2bool(value):
    return value.lower() in ('yes', 'true', 't', '1')

def env_var(name, converter=None, default=None):
    value = os.environ.get(name)
    if value is None or (isinstance(value, str) and len(value) == 0):
        return default
    elif converter:
        return converter(value)
    else:
        return value

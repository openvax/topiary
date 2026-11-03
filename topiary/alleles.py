"""Allele-name canonicalization shared by Topiary's readers and predictors.

All parsing is delegated to mhcgnomes; this module only fixes the two
policies Topiary needs, so they are not re-spelled at every call site.
"""

import mhcgnomes


def canonical_allele_name(value):
    """Return the mhcgnomes canonical name of one allele.

    Parameters
    ----------
    value : str
        Allele in any spelling mhcgnomes accepts (``HLA-A0201``,
        ``HLA-A*02:01``, ...). Non-strings are converted with ``str``.

    Returns
    -------
    str

    Raises
    ------
    mhcgnomes.errors.ParseError
        If the allele cannot be parsed.
    """
    return mhcgnomes.parse(str(value)).to_string()


def canonical_allele_names_or_raw(values):
    """Canonicalize allele strings, passing through anything unparseable.

    For readers of external files, where one odd allele should not fail the
    whole table. Non-strings and empty strings are returned unchanged, and
    each distinct spelling is parsed once.

    Parameters
    ----------
    values : iterable
        Allele values, typically a pandas Series.

    Returns
    -------
    list
        One entry per input value, in order.
    """
    cache = {}

    def canonicalize(raw):
        if not isinstance(raw, str) or not raw:
            return raw
        if raw not in cache:
            try:
                cache[raw] = canonical_allele_name(raw)
            except Exception:  # noqa: BLE001 - mhcgnomes raises many exception types
                cache[raw] = raw
        return cache[raw]

    return [canonicalize(value) for value in values]

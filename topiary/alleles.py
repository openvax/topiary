"""MHC allele parsing and canonical names for Topiary.

Every allele name Topiary compares, caches or reports goes through this
module. mhcgnomes does the parsing and owns the nomenclature; this module
only decides which kinds of mhcgnomes result Topiary treats as one allele.
``mhcgnomes.parse`` succeeds on many inputs that are not alleles: ``"A2"``
is the serotype HLA-A2, ``"HLA-A"`` is a gene and
``"HLA-A*02:01/HLA-A*02:02"`` is a "pair" of two class I alleles. Taking
the canonical spelling of such a result produced a name that looked like an
allele and was not one.

Accepted
--------
- An ``Allele`` or ``AlleleWithoutGene`` (``HLA-A*02:01``, ``H2-K*b``,
  ``BoLA-T2c``).
- A ``Pair`` whose chains mhcgnomes marks as a class II alpha and a class II
  beta (``HLA-DQA1*01:01/DQB1*06:02``).

Rejected, with :class:`AlleleError`
-----------------------------------
- A missing value, a non-string, or a string mhcgnomes cannot parse.
- Any other mhcgnomes result: serotype, supertype, haplotype, ambiguous
  alleles, gene, class II locus, MHC class or species.
- An allele mhcgnomes marks as null (``annotation_null``): it is not
  expressed, so binding predictions for it are meaningless.
- A ``Pair`` that is not a class II alpha/beta heterodimer, including a
  beta chain written first.

Expression caveats
------------------
An allele mhcgnomes flags for low, questionable or aberrant expression, as
secreted or cytoplasmic, or as a pseudogene (by annotation or because its
gene is curated as one) is accepted with a warning logged once per process:
the molecule exists, so a binding prediction is well defined, but it may
present little or nothing on the cell surface.

Not checked
-----------
- Whether an allele names one molecule or a group of them. mhcgnomes
  returns an ``Allele`` both for the allele group ``HLA-A*02`` and for
  NetMHC's ``Mamu-A01``, which names one molecule, and does not yet say
  which is which (pirl-unc/mhcgnomes#215, #216). Topiary does not guess
  from the number of fields.
- Whether the allele exists (mhcgnomes parses ``HLA-A*99:99``), and whether
  a given predictor supports it; the predictor answers that.
- A class II beta chain alone (``HLA-DQB1*06:02``) stays a single-chain
  allele: no alpha chain is inferred.
"""

import functools
import logging

import mhcgnomes  # type: ignore[import-untyped]  # Upstream does not publish types.
import pandas as pd

logger = logging.getLogger(__name__)

#: A specific allele, shown in messages as an example of what to write.
EXAMPLE_ALLELE = "HLA-A*02:01"

#: A class II alpha/beta pair, shown in messages as an example.
EXAMPLE_CLASS2_PAIR = "HLA-DQA1*01:01/DQB1*06:02"

#: The mhcgnomes result types that can name one allele. ``mhcgnomes.parse``
#: returns other types too (serotype, gene, haplotype, ...); those are refused.
ALLELE_RESULT_TYPES = (mhcgnomes.Allele, mhcgnomes.AlleleWithoutGene, mhcgnomes.Pair)

#: The mhcgnomes allele properties that put cell-surface expression in doubt,
#: with what each means for the logged warning. mhcgnomes sets the flags but
#: does not yet describe them (pirl-unc/mhcgnomes#217).
EXPRESSION_CAVEATS = {
    "annotation_low_expression": "low cell-surface expression",
    "annotation_questionable": "questionable expression",
    "annotation_secreted": "secreted as a soluble molecule, not on the cell surface",
    "annotation_cystosolic": "present in the cytoplasm, not on the cell surface",
    "annotation_aberrant_expression": "aberrant expression",
    "is_pseudogene": "a pseudogene",
}


class AlleleError(ValueError):
    """An input that does not name one specific MHC allele.

    The message names the input, says what mhcgnomes parsed it as (when it
    parsed at all) and why that was refused. It is a ``ValueError``, so code
    that already handles ``ValueError``, such as the command line's one-line
    error report, handles it too.

    Parameters
    ----------
    message : str
        The full explanation.
    value : object, optional
        The input as given.
    parsed : mhcgnomes.Result, optional
        What mhcgnomes parsed *value* as; None when it did not parse.

    Attributes
    ----------
    value : object
        The input as given.
    parsed : mhcgnomes.Result or None
        What mhcgnomes parsed *value* as. Not kept when the error is
        pickled, since mhcgnomes results do not pickle.
    """

    def __init__(self, message, value=None, parsed=None):
        super().__init__(message)
        self.value = value
        self.parsed = parsed

    def __reduce__(self):
        # An exception that cannot be pickled cannot cross a process
        # boundary at all, so keep the message and input and drop `parsed`.
        return (type(self), (str(self), self.value))


def allele_chains(parsed):
    """List the single-chain alleles in an mhcgnomes result.

    Parameters
    ----------
    parsed : mhcgnomes.Result
        Anything ``mhcgnomes.parse`` returns.

    Returns
    -------
    tuple of mhcgnomes.Allele
        ``(alpha, beta)`` for a ``Pair``, ``(parsed,)`` for an ``Allele``,
        and empty for anything else, including ``AlleleWithoutGene``, which
        carries no annotations.
    """
    if isinstance(parsed, mhcgnomes.Pair):
        return (parsed.alpha, parsed.beta)
    if isinstance(parsed, mhcgnomes.Allele):
        return (parsed,)
    return ()


def allele_rejection_reason(parsed):
    """Say why an mhcgnomes result is not one specific allele.

    Parameters
    ----------
    parsed : mhcgnomes.Result
        Anything ``mhcgnomes.parse`` returns.

    Returns
    -------
    str or None
        None when *parsed* is a usable allele (see the module docstring for
        the rules). Otherwise a clause that completes a sentence about the
        input and suggests what to write instead, such as ``"parses as the
        mhcgnomes Serotype HLA-A2, not an allele; give one of its 47
        alleles, such as HLA-A*02:01, ..."``. :func:`parse_allele` puts the
        input in front.
    """
    if not isinstance(parsed, ALLELE_RESULT_TYPES):
        members = getattr(parsed, "alleles", None) or ()
        if members:
            examples = ", ".join(allele.to_string() for allele in members[:3])
            advice = f"give one of its {len(members)} alleles, such as {examples}"
        else:
            advice = f"give one specific allele, such as {EXAMPLE_ALLELE}"
        return f"parses as the mhcgnomes {type(parsed).__name__} {parsed.to_string()}, not an allele; {advice}"
    if isinstance(parsed, mhcgnomes.Pair) and not (parsed.alpha.is_class2_alpha and parsed.beta.is_class2_beta):
        return (
            f"pairs {parsed.alpha.to_string()} with {parsed.beta.to_string()}, "
            f"which is not a class II alpha/beta heterodimer; give one allele, "
            f"or a class II pair written alpha/beta such as {EXAMPLE_CLASS2_PAIR}"
        )
    verb = "contains" if isinstance(parsed, mhcgnomes.Pair) else "is"
    for chain in allele_chains(parsed):
        if chain.annotation_null:
            return (
                f"{verb} the null allele {chain.to_string()}, which is not expressed, "
                f"so binding predictions for it are meaningless"
            )
    return None


def allele_expression_caveats(parsed):
    """List what mhcgnomes says puts an allele's surface expression in doubt.

    Parameters
    ----------
    parsed : mhcgnomes.Result
        Usually a result :func:`parse_allele` accepted.

    Returns
    -------
    tuple of str
        One entry per flag in :data:`EXPRESSION_CAVEATS` that mhcgnomes sets
        on the allele or on either chain of a pair, such as
        ``"HLA-A*24:02:01:02L (low cell-surface expression)"``. Empty for an
        unflagged allele and for anything that is not an Allele or Pair.
    """
    return tuple(
        f"{chain.to_string()} ({description})"
        for chain in allele_chains(parsed)
        for flag, description in EXPRESSION_CAVEATS.items()
        if getattr(chain, flag)
    )


@functools.cache
def log_allele_caveat_once(caveat):
    """Log an expression caveat as a warning, once per process.

    Strict callers canonicalize alleles row by row, so logging on every
    parse would repeat the same warning for every row of a table.

    Parameters
    ----------
    caveat : str
        One entry from :func:`allele_expression_caveats`.

    Returns
    -------
    None
    """
    logger.warning(
        "Accepting allele %s: binding predictions for it are well defined, "
        "but it may present little or nothing on the cell surface", caveat,
    )


def parse_allele(value, species=None):
    """Parse one MHC allele with mhcgnomes, refusing anything else.

    Parameters
    ----------
    value : str
        One allele in any spelling mhcgnomes accepts (``HLA-A0201``,
        ``A*02:01``, ``H-2-Kb``, ``HLA-DQA1*01:01-DQB1*06:02``).
    species : str, optional
        The species every input belongs to, as a latin name, common name or
        MHC prefix (``"Mus musculus"``, ``"mouse"``, ``"H2"``). It is passed
        to ``mhcgnomes.parse(species=...)``, which reads unprefixed names
        (``"Kb"``) as that species and rejects names whose prefix says
        otherwise. Without it, unprefixed names are read as human, except
        where only one species has the gene (``"88*001:01"`` is DLA).

    Returns
    -------
    mhcgnomes.Allele, mhcgnomes.Pair or mhcgnomes.AlleleWithoutGene
        The parsed allele. Expression caveats (``L``, ``Q``, ...) are logged
        once per process, not raised.

    Raises
    ------
    AlleleError
        If *value* is missing (None, NaN, ``pd.NA``, blank, or text such as
        ``"nan"`` that :func:`topiary.is_stated` reads as missing), is not a
        string, cannot be parsed, or parses as something other than one
        specific allele (see the module docstring). Each case has its own
        message naming the input.
    ValueError
        If *species* is not one species mhcgnomes knows. This is a mistake in
        the call rather than in the data, so it is not an AlleleError and
        the lenient :func:`canonical_allele_names_or_raw` does not absorb it.
    """
    # Imported here because topiary.ranking.nodes imports this module.
    from .ranking.nodes import is_stated

    if species is not None and mhcgnomes.Species.get(species) is None:
        raise ValueError(
            f"species {species!r} is not one species mhcgnomes knows; "
            f"give a latin name, common name or MHC prefix such as 'Mus musculus', 'mouse' or 'H2'"
        )
    if pd.api.types.is_scalar(value) and not is_stated(value):
        raise AlleleError(f"missing allele: got {value!r}", value)
    if not isinstance(value, str):
        raise AlleleError(f"an allele must be a string, not {type(value).__name__}: {value!r}", value)
    try:
        parsed = mhcgnomes.parse(value, species=species)
    except mhcgnomes.ParseError as error:
        if species is None:
            raise AlleleError(
                f"could not parse {value!r} as one MHC allele (mhcgnomes: {error}). Give one "
                f"allele per value; names without a species prefix are read as human, so write "
                f"a non-human allele with its prefix, such as H2-K*b or Mamu-A1*001:01, or pass species=",
                value,
            ) from error
        raise AlleleError(
            f"could not parse {value!r} as one MHC allele of species {species!r} (mhcgnomes: {error})",
            value,
        ) from error
    reason = allele_rejection_reason(parsed)
    if reason is not None:
        raise AlleleError(f"{value!r} {reason}", value, parsed)
    for caveat in allele_expression_caveats(parsed):
        log_allele_caveat_once(caveat)
    return parsed


def canonical_allele_name(value, species=None):
    """Return the mhcgnomes canonical name of one allele.

    Parameters
    ----------
    value : str
        One allele in any spelling mhcgnomes accepts.
    species : str, optional
        The species of *value*; see :func:`parse_allele`.

    Returns
    -------
    str
        ``to_string()`` of the parsed allele, such as ``HLA-A*02:01``,
        ``H2-K*b`` or ``HLA-DQA1*01:01/DQB1*06:02``. Canonicalizing a
        canonical name returns it unchanged.

    Raises
    ------
    AlleleError
        If *value* is not one specific allele; see :func:`parse_allele`.
    ValueError
        If *species* is not one species mhcgnomes knows.
    """
    return parse_allele(value, species=species).to_string()


def canonical_allele_names_or_raw(values, species=None):
    """Canonicalize allele values, keeping any that are not alleles as given.

    For readers of external tables (pVACseq, LENS), where one bad cell
    should not fail the whole file. A value that is not one allele is kept
    as given rather than made missing: a missing allele would turn the row
    into allele-free evidence downstream, while the original text stays
    visible and a strict step later (such as :func:`topiary.combine_sources`)
    names it in its error.

    Parameters
    ----------
    values : iterable
        Allele values, typically a pandas Series.
    species : str, optional
        The species of every value; see :func:`parse_allele`.

    Returns
    -------
    list
        One entry per input value, in order. Each allele becomes its
        canonical name. Missing values and non-strings pass through
        unchanged and silently. Any other string that is not one allele
        passes through unchanged, and one warning per distinct value is
        logged with the reason. Each distinct spelling is parsed once.

    Raises
    ------
    ValueError
        If *species* is not one species mhcgnomes knows (checked when the
        first stated string is parsed).
    """
    # Imported here because topiary.ranking.nodes imports this module.
    from .ranking.nodes import is_stated

    names = {}
    result = []
    for value in values:
        if not isinstance(value, str) or not is_stated(value):
            result.append(value)
            continue
        if value not in names:
            try:
                names[value] = canonical_allele_name(value, species=species)
            except AlleleError as error:
                logger.warning("Allele value kept as given: %s", error)
                names[value] = value
        result.append(names[value])
    return result

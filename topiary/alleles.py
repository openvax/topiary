"""MHC allele parsing and canonical names for Topiary.

Every allele name Topiary compares, caches or reports goes through this
module. Parsing is delegated to mhcgnomes. What this module adds is the
decision of which mhcgnomes results count as one allele, because
``mhcgnomes.parse`` succeeds on many inputs that are not alleles: ``"A2"``
is the serotype HLA-A2, ``"HLA-A"`` is a gene, ``"A*02"`` is the allele
group HLA-A*02 and ``"HLA-A*02:01/HLA-A*02:02"`` is a "pair" of two class I
alleles. Taking the canonical spelling of such a result produced a name
that looked like an allele and was not one.

Accepted
--------
- An ``Allele`` with at least two fields (``HLA-A*02:01``), or a
  letter-named allele whose one field names the molecule (mouse
  ``H2-K*b``, rat ``RT1-A*a``, ``SLA-1*CDY``).
- A ``Pair`` that is a class II alpha/beta heterodimer
  (``HLA-DQA1*01:01/DQB1*06:02``), each chain held to the rules above.
- An ``AlleleWithoutGene``: mhcgnomes returns it only for curated names of
  specific molecules whose gene is unassigned, such as ``BoLA-T2c``, which
  NetMHCpan predicts for.

Rejected, with :class:`AlleleError`
-----------------------------------
- A missing value, a non-string, or a string mhcgnomes cannot parse.
- Any other mhcgnomes result: serotype, supertype, haplotype, ambiguous
  alleles, gene, class II locus, MHC class or species.
- An allele group: one numeric field (``HLA-A*02``, ``HLA-DRB1*04``). In
  IPD-IMGT/HLA and IPD-MHC numbering the first field alone names a group
  of proteins; the second field names one. Legacy one-number names such
  as ``Mamu-A*01`` fall under this rule too; write the current two-field
  name (``Mamu-A1*001:01``).
- A null allele (``N``): it is not expressed, so binding predictions for it
  are meaningless.
- A ``Pair`` that is not an alpha/beta heterodimer, including a beta chain
  written first.

Annotations
-----------
``L``, ``Q``, ``S``, ``C``, ``A`` and ``Ps`` are accepted with a logged
warning (see :data:`EXPRESSION_CAVEATS`): the molecule exists, so a binding
prediction is well defined, but whether it reaches the cell surface is
reduced, uncertain or ruled out. ``G`` and ``P`` groups are accepted
without comment: they group alleles that share a peptide-binding domain,
which is what a binding prediction depends on.

Not checked
-----------
Whether the allele exists (mhcgnomes parses ``HLA-A*99:99``), and whether a
given predictor supports it; the predictor answers that. A class II beta
chain alone (``HLA-DQB1*06:02``) stays a single-chain allele: no alpha
chain is inferred.
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

#: mhcgnomes result types that are not one allele, with what each is called
#: in an error message. Checked in order with ``isinstance``.
NON_ALLELE_KINDS = (
    (mhcgnomes.Serotype, "a serotype"),
    (mhcgnomes.Supertype, "a supertype"),
    (mhcgnomes.Haplotype, "a haplotype"),
    (mhcgnomes.AmbiguousAlleles, "an ambiguous set of alleles"),
    (mhcgnomes.Gene, "a gene"),
    (mhcgnomes.Class2Locus, "a class II locus"),
    (mhcgnomes.MhcClass, "an MHC class"),
    (mhcgnomes.Species, "a species"),
)

#: The non-allele result types that list their member alleles in ``.alleles``.
MULTI_ALLELE_TYPES = (
    mhcgnomes.Serotype,
    mhcgnomes.Supertype,
    mhcgnomes.Haplotype,
    mhcgnomes.AmbiguousAlleles,
)

#: Allele-name annotations accepted with a logged warning, and what each
#: means (IPD-IMGT/HLA and IPD-MHC nomenclature). The null annotation ``N``
#: is not here: a null allele is rejected.
EXPRESSION_CAVEATS = {
    "L": "low cell-surface expression",
    "Q": "questionable expression",
    "S": "secreted as a soluble molecule, not on the cell surface",
    "C": "present in the cytoplasm, not on the cell surface",
    "A": "aberrant expression",
    "Ps": "a pseudogene",
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
        input and suggests what to write instead, such as ``"is a serotype
        (HLA-A2), not an allele; give one of its 47 alleles, such as
        HLA-A*02:01, ..."``. :func:`parse_allele` puts the input in front.
    """
    if isinstance(parsed, mhcgnomes.AlleleWithoutGene):
        return None
    if isinstance(parsed, mhcgnomes.Allele):
        chains, verb = (parsed,), "is"
    elif isinstance(parsed, mhcgnomes.Pair):
        if not (parsed.alpha.is_class2_alpha and parsed.beta.is_class2_beta):
            return (
                f"pairs {parsed.alpha.to_string()} with {parsed.beta.to_string()}, "
                f"which is not a class II alpha/beta heterodimer; give one allele, "
                f"or a class II pair written alpha/beta such as {EXAMPLE_CLASS2_PAIR}"
            )
        chains, verb = (parsed.alpha, parsed.beta), "contains"
    else:
        kind = next(
            (label for result_type, label in NON_ALLELE_KINDS if isinstance(parsed, result_type)),
            f"a {type(parsed).__name__}",
        )
        if isinstance(parsed, MULTI_ALLELE_TYPES) and parsed.alleles:
            examples = ", ".join(allele.to_string() for allele in parsed.alleles[:3])
            advice = f"give one of its {len(parsed.alleles)} alleles, such as {examples}"
        else:
            advice = f"give one specific allele, such as {EXAMPLE_ALLELE}"
        return f"is {kind} ({parsed.to_string()}), not an allele; {advice}"
    for chain in chains:
        name = chain.to_string()
        if chain.annotation_null:
            return (
                f"{verb} the null allele {name}, which is not expressed, "
                f"so binding predictions for it are meaningless"
            )
        if len(chain.allele_fields) == 1 and chain.allele_fields[0].isdigit():
            return (
                f"{verb} the allele group {name} (one field) rather than a specific "
                f"allele; give at least two fields, such as {EXAMPLE_ALLELE} rather than HLA-A*02"
            )
    return None


def allele_expression_caveats(parsed):
    """List what an allele's annotations say about its expression.

    Parameters
    ----------
    parsed : mhcgnomes.Result
        Usually a result :func:`parse_allele` accepted.

    Returns
    -------
    tuple of str
        One entry per annotation in :data:`EXPRESSION_CAVEATS` on the allele
        or on either chain of a pair, such as ``"HLA-A*24:02:01:02L,
        annotated L (low cell-surface expression)"``. Empty for an
        unannotated allele and for anything that is not an Allele or Pair.
    """
    if isinstance(parsed, mhcgnomes.Pair):
        chains = (parsed.alpha, parsed.beta)
    elif isinstance(parsed, mhcgnomes.Allele):
        chains = (parsed,)
    else:
        chains = ()
    caveats = []
    for chain in chains:
        for code in chain.annotations:
            if code in EXPRESSION_CAVEATS:
                caveats.append(f"{chain.to_string()}, annotated {code} ({EXPRESSION_CAVEATS[code]})")
    return tuple(caveats)


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

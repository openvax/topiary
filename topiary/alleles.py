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
- An allele mhcgnomes marks as an allele group (``is_allele_group``), such
  as ``HLA-A*02`` or ``HLA-DRB1*04``, or a pair with such a chain. A group
  covers many molecules; G/P binding-domain typings are also refused rather
  than substituted with one full allele identity.
- An allele mhcgnomes marks as null (``annotation_null``): it is not
  expressed, so Topiary excludes it from presentation analysis.
- A ``Pair`` that is not a class II alpha/beta heterodimer, including a
  beta chain written first.

Old names
---------
Names are read with mhcgnomes' defaults, which do not resolve old names.
NetMHC's short macaque names (``Mamu-A01``, ``Mamu-B08``) therefore parse as
the allele groups ``Mamu-A*01`` and ``Mamu-B*008`` and are refused. The
error names the allele mhcgnomes resolves the old name to when asked
(``use_allele_aliases=True``; ``Mamu-A1*001:01``), via
:func:`resolve_old_allele_name`. Topiary does not resolve old names itself,
because that would rename alleles it already accepts.

Expression caveats
------------------
An allele whose name mhcgnomes says bears on cell-surface expression
(``cell_surface_expression``: low, uncertain or absent, such as a secreted,
cytoplasmic or pseudogene allele) is accepted with a warning logged once per
process, worded with mhcgnomes' meaning of each annotation
(``annotation_descriptions``). The molecule exists, so a binding prediction
is well defined, but it may present little or nothing on the cell surface.

Not checked
-----------
- Whether the allele exists (mhcgnomes parses ``HLA-A*99:99``), and whether
  a given predictor supports it; the predictor answers that.
- A class II beta chain alone (``HLA-DQB1*06:02``) stays a single-chain
  allele: no alpha chain is inferred.
"""

import functools
import logging

import mhcgnomes  # type: ignore[import-untyped]  # Upstream does not publish types.
import pandas as pd
from mhcgnomes.allele_annotations import annotation_descriptions

logger = logging.getLogger(__name__)

#: A specific allele, shown in messages as an example of what to write.
EXAMPLE_ALLELE = "HLA-A*02:01"

#: A class II alpha/beta pair, shown in messages as an example.
EXAMPLE_CLASS2_PAIR = "HLA-DQA1*01:01/DQB1*06:02"

#: The mhcgnomes result types that can name one allele. ``mhcgnomes.parse``
#: returns other types too (serotype, gene, haplotype, ...); those are refused.
ALLELE_RESULT_TYPES = (mhcgnomes.Allele, mhcgnomes.AlleleWithoutGene, mhcgnomes.Pair)

#: How a gene-curated pseudogene (``HLA-H``), which carries no ``Ps``
#: annotation, is described in an expression caveat: mhcgnomes' own meaning
#: of the ``Ps`` annotation.
PSEUDOGENE_DESCRIPTION = annotation_descriptions["Ps"]


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
    if isinstance(parsed, mhcgnomes.Pair) and not (
            isinstance(parsed.alpha, mhcgnomes.Allele)
            and isinstance(parsed.beta, mhcgnomes.Allele)
            and parsed.alpha.species == parsed.beta.species
            and parsed.alpha.is_class2_alpha and parsed.beta.is_class2_beta):
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
                f"so Topiary excludes it from presentation analysis"
            )
        if chain.is_allele_group or set(chain.annotations).intersection({"G", "P"}):
            return (
                f"{verb} the allele group {chain.to_string()}, not one allele; "
                f"give one specific allele, such as {EXAMPLE_ALLELE}"
            )
    return None


def resolve_old_allele_name(value, species=None):
    """Name the allele mhcgnomes resolves an old, group-looking name to.

    Some single molecules have old names that now read as allele groups,
    such as NetMHC's ``Mamu-A01``, which mhcgnomes parses by default as the
    group ``Mamu-A*01``. mhcgnomes resolves old names only when asked
    (``use_allele_aliases=True``). Topiary keeps the default names, since
    resolving would rename alleles it already accepts, and uses this only to
    say in an error which allele an old name meant.

    Parameters
    ----------
    value : str
        One allele name as given.
    species : str, optional
        The species of *value*; see :func:`parse_allele`.

    Returns
    -------
    str or None
        The canonical name of the allele mhcgnomes resolves *value* to, such
        as ``"Mamu-A1*001:01"`` for ``"Mamu-A01"``. None unless *value*
        parses by default as an allele group and, with old names resolved,
        as an allele this module accepts. Also None when *value* does not
        parse.
    """
    parsed = mhcgnomes.parse(value, species=species, raise_on_error=False)
    if not isinstance(parsed, mhcgnomes.Allele) or not parsed.is_allele_group:
        return None
    resolved = mhcgnomes.parse(value, species=species, use_allele_aliases=True, raise_on_error=False)
    if resolved is None or allele_rejection_reason(resolved) is not None:
        return None
    return resolved.to_string()


def allele_expression_caveats(parsed):
    """List the chains whose names mhcgnomes says bear on surface expression.

    Which names count, and what each annotation means, are mhcgnomes'
    (``cell_surface_expression`` and ``annotation_descriptions``), so the
    wording follows mhcgnomes rather than a table kept here.

    Parameters
    ----------
    parsed : mhcgnomes.Result
        Usually a result :func:`parse_allele` accepted.

    Returns
    -------
    tuple of str
        One entry per allele, or per chain of a pair, whose
        ``cell_surface_expression`` is not None, naming the allele and what
        its annotations mean, such as ``"HLA-A*24:02:01:02L (low
        cell-surface expression)"``. A pseudogene by gene curation, such as
        ``HLA-H*01:01``, which carries no annotation, is described as
        :data:`PSEUDOGENE_DESCRIPTION`. Empty for an allele whose name says
        nothing about expression and for anything that is not an Allele or
        Pair.
    """
    caveats = []
    for chain in allele_chains(parsed):
        if chain.cell_surface_expression is None:
            continue
        meanings = list(chain.annotation_descriptions.values())
        if chain.is_pseudogene and not chain.annotation_pseudogene:
            meanings.append(PSEUDOGENE_DESCRIPTION)
        caveats.append(f"{chain.to_string()} ({'; '.join(meanings)})")
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
        specific allele, such as an allele group (see the module docstring).
        Each case has its own message naming the input; for an old name that
        reads as a group, it also names the allele mhcgnomes resolves it to
        (:func:`resolve_old_allele_name`).
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
        parsed = mhcgnomes.parse(
            value.strip().replace(";", "/"), species=species,
            use_allele_aliases=False, infer_class2_pairing=False,
            collapse_singleton_haplotypes=True, collapse_singleton_serotypes=False,
            max_allele_fields=None, raise_on_error=True)
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
        message = f"{value!r} {reason}"
        resolved = resolve_old_allele_name(value, species=species)
        if resolved is not None:
            message += f". mhcgnomes resolves {value!r} as an old name of {resolved}; write that if you mean it"
        raise AlleleError(message, value, parsed)
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

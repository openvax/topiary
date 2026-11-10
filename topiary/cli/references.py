"""Resolve CLI source/reference options independently of presenting MHC.

HLA-transgenic mice are a real use case (Pascolo et al., PMID 9182675).
Species selects sequence annotations, never the predictor's allele vocabulary.
"""

from numbers import Integral

import pyensembl


def source_options(args):
    """Normalize the direct-input defaults without opening annotation data."""
    species = getattr(args, "species", None)
    return {
        "species": pyensembl.normalize_species_name(
            "human" if species is None else species
        ),
        "release": getattr(args, "ensembl_release", None),
    }


def _genome_species(genome):
    species = getattr(genome, "species", None)
    name = getattr(species, "latin_name", species)
    if not name:
        raise ValueError(
            "Cannot establish the input annotation species; select an explicit "
            "--exclude-species / --exclude-ensembl-release background, or use "
            "--exclude-fasta."
        )
    return pyensembl.normalize_species_name(name)


def validate_source_options(args, genomes):
    """Check explicit source assertions against annotations actually loaded."""
    options = source_options(args)
    for genome in genomes:
        if getattr(args, "species", None) is not None:
            actual = _genome_species(genome)
            if actual != options["species"]:
                raise ValueError(
                    "--species %s conflicts with input annotation species %s; "
                    "use --exclude-species for a different background."
                    % (options["species"], actual)
                )
        if options["release"] is not None:
            actual = getattr(genome, "release", None)
            if actual != options["release"]:
                raise ValueError(
                    "--ensembl-release %s conflicts with input annotation release %r; "
                    "use --exclude-ensembl-release for a different background."
                    % (options["release"], actual)
                )


def require_human(options, flag):
    """Reject a human-only data source in the context it would read."""
    if options["species"] != "homo_sapiens":
        raise ValueError(
            "%s uses human-only membership data; requested species is %s. "
            "Use species-appropriate sequences instead." % (flag, options["species"])
        )


def require_release(options, flag):
    """Only human has a documented default numbered Ensembl release."""
    if options["release"] is None and options["species"] != "homo_sapiens":
        raise ValueError(
            "Species %s requires an explicit %s." % (options["species"], flag)
        )


def has_builtin_exclusions(args):
    return any(
        getattr(args, name, None)
        for name in ("exclude_ensembl", "exclude_non_cta", "exclude_tissues")
    )


def exclusion_options(args, genomes=()):
    """Resolve one built-in background; explicit species starts its own context.

    With no override, use the actual single input annotation when available.
    File-based sequences have no annotation metadata and use the source flags
    (human by default). An explicit exclusion species does not borrow the source
    release, even if both happen to name the same species.
    """
    species = getattr(args, "exclude_species", None)
    release = getattr(args, "exclude_ensembl_release", None)
    if not has_builtin_exclusions(args):
        if species is not None or release is not None:
            raise ValueError(
                "--exclude-species / --exclude-ensembl-release require a built-in "
                "exclusion (--exclude-ensembl, --exclude-non-cta or --exclude-tissues)."
            )
        return None

    if species is not None:
        options = {
            "species": pyensembl.normalize_species_name(species),
            "release": release,
        }
    elif genomes:
        species_names = {_genome_species(genome) for genome in genomes}
        if len(species_names) != 1:
            raise ValueError(
                "Input annotations span multiple species; select --exclude-species "
                "and its release explicitly, or use --exclude-fasta."
            )
        if release is None:
            releases = {getattr(genome, "release", None) for genome in genomes}
            if len(releases) != 1 or any(
                not isinstance(value, Integral) or isinstance(value, bool)
                for value in releases
            ):
                raise ValueError(
                    "Input annotations do not identify one numbered Ensembl release; "
                    "select --exclude-ensembl-release explicitly, or use --exclude-fasta."
                )
            release = releases.pop()
        options = {"species": species_names.pop(), "release": release}
    else:
        options = source_options(args)
        if release is not None:
            options["release"] = release

    require_release(options, "--exclude-ensembl-release")
    for flag in ("exclude_non_cta", "exclude_tissues"):
        if getattr(args, flag, None):
            require_human(options, "--" + flag.replace("_", "-"))
    return options

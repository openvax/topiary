"""Shared pytest configuration."""

import pytest

from tests.optional_dependencies import require_integration


@pytest.fixture(scope="module")
def cancer_test_variants():
    """Real BRAF V600E and TP53 R248W variants, constructed only on demand."""
    from varcode import Variant, VariantCollection
    from pyensembl import ensembl_grch38

    return VariantCollection([
        # COSMIC mutation entries 476 (BRAF) and 10656 (TP53).
        Variant(7, 140753336, "A", "T", ensembl_grch38),
        Variant(17, 7674221, "G", "A", ensembl_grch38),
    ])


@pytest.fixture(scope="module")
def cancer_test_effects(cancer_test_variants):
    """Compute real variant effects during fixture setup, never collection."""
    return cancer_test_variants.effects()


@pytest.fixture(scope="module")
def gene_expression_dict(cancer_test_variants):
    """Assign 1.0 FPKM to each gene annotated at the test variants."""
    return {
        gene_id: 1.0
        for variant in cancer_test_variants
        for gene_id in variant.gene_ids
    }


@pytest.fixture(scope="module")
def transcript_expression_dict(cancer_test_variants):
    """Assign 1.0 FPKM to each transcript annotated at the test variants."""
    return {
        transcript_id: 1.0
        for variant in cancer_test_variants
        for transcript_id in variant.transcript_ids
    }


def pytest_runtest_setup(item):
    """Require a usable optional import before setting up its fixtures."""
    for dependency in ("isovar", "pirlygenes"):
        if item.get_closest_marker(dependency) is not None:
            require_integration(dependency)

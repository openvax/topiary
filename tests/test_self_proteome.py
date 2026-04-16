"""Tests for topiary.self_proteome.SelfProteome — construction, nearest
lookup, scope filtering, and TopiaryPredictor integration."""

from unittest.mock import MagicMock

import pytest
from mhctools import RandomBindingPredictor

from topiary import SelfProteome, TopiaryPredictor


# ---------------------------------------------------------------------------
# Construction + basic metadata
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_peptides_indexes_by_length(self):
        ref = SelfProteome.from_peptides(
            {"geneA": "MASIINFEKLGGG"},   # length 13 → 5 × 9-mers
            peptide_lengths=[9],
        )
        assert ref.peptide_lengths == [9]
        assert ref.n_reference_peptides == 5

    def test_from_peptides_multiple_lengths(self):
        ref = SelfProteome.from_peptides(
            {"geneA": "MASIINFEKLGGG"},
            peptide_lengths=[8, 9, 10],
        )
        assert ref.peptide_lengths == [8, 9, 10]
        # 13aa → 6 × 8-mers + 5 × 9-mers + 4 × 10-mers
        assert ref.n_reference_peptides == 6 + 5 + 4

    def test_reference_version_embeds_species_scope(self):
        ref = SelfProteome.from_peptides(
            {"g": "SIINFEKLA"}, peptide_lengths=[9],
        )
        assert "ensembl-synthetic" in ref.reference_version
        assert "scope-all" in ref.reference_version

    def test_from_peptides_short_sequence_skips_longer_lengths(self):
        # 5aa sequence — can't produce any 9-mers
        ref = SelfProteome.from_peptides(
            {"geneA": "SIINF"},
            peptide_lengths=[9],
        )
        assert ref.n_reference_peptides == 0


# ---------------------------------------------------------------------------
# FASTA loader
# ---------------------------------------------------------------------------


class TestFastaLoader:
    def test_from_fasta_parses_multi_record(self, tmp_path):
        path = tmp_path / "ref.fa"
        path.write_text(
            ">geneA\n"
            "MASIINFEKLGGG\n"
            ">geneB description line\n"
            "QPRSTVWYACDEFGH\n"
        )
        ref = SelfProteome.from_fasta(path, peptide_lengths=[9])
        # 5 × 9-mers from A + 7 × 9-mers from B = 12 distinct
        assert ref.n_reference_peptides == 12

    def test_from_fasta_rejects_named_scope_without_metadata(self, tmp_path):
        path = tmp_path / "ref.fa"
        path.write_text(">geneA\nSIINFEKLA\n")
        with pytest.raises(ValueError, match="FASTA"):
            SelfProteome.from_fasta(
                path, peptide_lengths=[9], scope="non_cta",
            )

    def test_from_fasta_accepts_callable_scope(self, tmp_path):
        path = tmp_path / "ref.fa"
        path.write_text(
            ">geneA\nMASIINFEKLGGG\n>geneB\nQPRSTVWYACDEF\n",
        )
        # Keep only geneA
        ref = SelfProteome.from_fasta(
            path, peptide_lengths=[9], scope=lambda g: g == "geneA",
        )
        # Only geneA 9-mers: 5 × 9
        assert ref.n_reference_peptides == 5


# ---------------------------------------------------------------------------
# Nearest lookup
# ---------------------------------------------------------------------------


class TestNearest:
    def test_exact_match_distance_zero(self):
        ref = SelfProteome.from_peptides(
            {"g": "MASIINFEKLGGG"}, peptide_lengths=[9],
        )
        out = ref.nearest(["SIINFEKLG"])
        row = out.iloc[0]
        assert row["self_nearest_peptide"] == "SIINFEKLG"
        assert row["self_nearest_edit_distance"] == 0

    def test_one_substitution_distance_one(self):
        ref = SelfProteome.from_peptides(
            {"g": "SIINFEKLA"}, peptide_lengths=[9],
        )
        out = ref.nearest(["XIINFEKLA"])
        assert out.iloc[0]["self_nearest_edit_distance"] == 1
        assert out.iloc[0]["self_nearest_peptide"] == "SIINFEKLA"

    def test_multiple_queries_preserve_order(self):
        ref = SelfProteome.from_peptides(
            {"g": "MASIINFEKLGGGQRSTV"}, peptide_lengths=[9],
        )
        queries = ["SIINFEKLG", "XXXXXXXXX", "ASIINFEKL"]
        out = ref.nearest(queries)
        assert list(out["peptide"]) == queries

    def test_length_mismatch_returns_null_row(self):
        # Reference has 9-mers; query is an 8-mer.
        ref = SelfProteome.from_peptides(
            {"g": "SIINFEKLA"}, peptide_lengths=[9],
        )
        out = ref.nearest(["ABCDEFGH"])
        row = out.iloc[0]
        assert row["peptide"] == "ABCDEFGH"
        assert row["self_nearest_peptide"] is None
        assert row["self_nearest_edit_distance"] is None

    def test_provenance_is_populated(self):
        ref = SelfProteome.from_peptides(
            {"MY_GENE": "SIINFEKLA"}, peptide_lengths=[9],
        )
        out = ref.nearest(["SIINFEKLA"])
        row = out.iloc[0]
        assert row["self_nearest_gene_id"] == "MY_GENE"
        assert row["self_nearest_transcript_id"] == "MY_GENE"
        assert row["self_nearest_reference_offset"] == 0

    def test_reference_version_stamped_on_every_row(self):
        ref = SelfProteome.from_peptides(
            {"g": "SIINFEKLA"}, peptide_lengths=[9],
        )
        out = ref.nearest(["SIINFEKLA", "XXXXXXXXX"])
        assert (out["self_nearest_reference_version"] == ref.reference_version).all()

    def test_mixed_lengths_in_one_call(self):
        ref = SelfProteome.from_peptides(
            {"g": "MASIINFEKLGGG"},
            peptide_lengths=[8, 9],
        )
        out = ref.nearest(["SIINFEKL", "SIINFEKLG"])
        assert out.iloc[0]["self_nearest_peptide_length"] == 8
        assert out.iloc[1]["self_nearest_peptide_length"] == 9


# ---------------------------------------------------------------------------
# Scope resolution for from_ensembl (error paths, no actual Ensembl)
# ---------------------------------------------------------------------------


class TestEnsemblScopeErrors:
    def test_non_human_non_cta_without_source_raises(self, monkeypatch):
        # Stub out pyensembl so we don't hit network / disk.
        fake_release = MagicMock()
        fake_release.protein_ids.return_value = []
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake_release,
        )
        with pytest.raises(ValueError, match="no default registered"):
            SelfProteome.from_ensembl(species="mouse", scope="non_cta")

    def test_pirlygenes_requires_human(self, monkeypatch):
        fake_release = MagicMock()
        fake_release.protein_ids.return_value = []
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake_release,
        )
        with pytest.raises(ValueError, match="human-only"):
            SelfProteome.from_ensembl(
                species="mouse", scope="non_cta", cta_source="pirlygenes",
            )

    def test_protected_tissues_not_yet_implemented(self, monkeypatch):
        fake_release = MagicMock()
        fake_release.protein_ids.return_value = []
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake_release,
        )
        # Protected tissues now implemented — non-human without
        # tissue_gene_ids should raise (human-only default data).
        with pytest.raises(ValueError, match="human-only"):
            SelfProteome.from_ensembl(
                species="mouse", scope="protected_tissues",
            )

    def test_unknown_scope_string_rejects(self, monkeypatch):
        fake_release = MagicMock()
        fake_release.protein_ids.return_value = []
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake_release,
        )
        with pytest.raises(ValueError, match="Unknown scope"):
            SelfProteome.from_ensembl(
                species="human", scope="bananas",
            )

    def test_custom_cta_source_as_set(self, monkeypatch):
        fake_release = MagicMock()
        fake_release.protein_ids.return_value = []
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake_release,
        )
        # Non-human + explicit CTA set should succeed.
        ref = SelfProteome.from_ensembl(
            species="mouse", scope="non_cta",
            cta_source={"ENSMUSG0001", "ENSMUSG0002"},
        )
        assert "non_cta" in ref.reference_version
        assert "sha256:" in ref.reference_version


class TestEnsemblHappyPath:
    """End-to-end construction via a fully mocked pyensembl genome,
    exercising the from_ensembl pipeline (protein iteration, gene
    filtering, indexing, nearest lookup)."""

    def _fake_genome(self):
        # Two genes: KEEP_GENE (stays after CTA filter) and CTA_GENE.
        # Each has one protein-coding transcript.
        proteins = {
            "PROT_KEEP": {
                "gene_id": "KEEP_GENE",
                "transcript_id": "ENST_KEEP",
                "sequence": "MASIINFEKLGGG",  # 5 × 9-mers
            },
            "PROT_CTA": {
                "gene_id": "CTA_GENE",
                "transcript_id": "ENST_CTA",
                "sequence": "QRSTVWYACDEFGH",  # different 9-mers
            },
        }
        fake = MagicMock()
        fake.protein_ids.return_value = list(proteins.keys())
        fake.gene_id_of_protein_id.side_effect = (
            lambda pid: proteins[pid]["gene_id"]
        )
        fake.transcript_id_of_protein_id.side_effect = (
            lambda pid: proteins[pid]["transcript_id"]
        )
        fake.protein_sequence.side_effect = (
            lambda pid: proteins[pid]["sequence"]
        )
        return fake

    def test_from_ensembl_indexes_proteins_and_filters_cta(
        self, monkeypatch,
    ):
        fake = self._fake_genome()
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake,
        )
        ref = SelfProteome.from_ensembl(
            species="human",
            release=93,
            peptide_lengths=[9],
            scope="non_cta",
            cta_source={"CTA_GENE"},  # filter out CTA_GENE
        )
        # Only KEEP_GENE's 9-mers (5 of them) should be indexed.
        assert ref.n_reference_peptides == 5
        assert ref.reference_version == (
            "ensembl-human-93+scope-non_cta+cta-sha256:"
            + __import__("hashlib").sha256(b"CTA_GENE").hexdigest()[:12]
        )

    def test_from_ensembl_nearest_returns_real_provenance(
        self, monkeypatch,
    ):
        fake = self._fake_genome()
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake,
        )
        ref = SelfProteome.from_ensembl(
            species="human",
            release=93,
            peptide_lengths=[9],
            scope="all",
        )
        # SIINFEKLG lives in KEEP_GENE at offset 2
        out = ref.nearest(["SIINFEKLG"])
        row = out.iloc[0]
        assert row["self_nearest_peptide"] == "SIINFEKLG"
        assert row["self_nearest_edit_distance"] == 0
        assert row["self_nearest_gene_id"] == "KEEP_GENE"
        assert row["self_nearest_transcript_id"] == "ENST_KEEP"
        assert row["self_nearest_reference_offset"] == 2


class TestTieBreaking:
    """Pin the documented (and deterministic) tie-breaking rule:
    when multiple reference peptides match at the same minimum
    distance, the first in internal array order wins."""

    def test_tie_at_distance_zero_first_reference_wins(self):
        # Two genes with the same 9-mer — tie at distance 0.
        # from_peptides builds the reference in insertion order, so
        # geneA (declared first) should win.
        ref = SelfProteome.from_peptides(
            {"geneA": "SIINFEKLA", "geneB": "SIINFEKLA"},
            peptide_lengths=[9],
        )
        out = ref.nearest(["SIINFEKLA"])
        row = out.iloc[0]
        assert row["self_nearest_edit_distance"] == 0
        assert row["self_nearest_gene_id"] == "geneA"


# ---------------------------------------------------------------------------
# Integration with TopiaryPredictor
# ---------------------------------------------------------------------------


class TestTopiaryPredictorIntegration:
    def test_self_nearest_columns_appear_on_predict_output(self):
        ref = SelfProteome.from_peptides(
            {"g": "MASIINFEKLGGG"}, peptide_lengths=[9],
        )
        predictor = TopiaryPredictor(
            models=RandomBindingPredictor(
                alleles=["HLA-A*02:01"], default_peptide_lengths=[9],
            ),
            self_proteome=ref,
        )
        df = predictor.predict_from_named_sequences(
            {"prot": "MASIINFEKLGGG"},
        )
        assert "self_nearest_peptide" in df.columns
        assert "self_nearest_edit_distance" in df.columns
        assert "self_nearest_reference_version" in df.columns
        # Every predicted peptide is in the reference → all distances 0.
        assert (df["self_nearest_edit_distance"] == 0).all()

    def test_no_self_proteome_no_columns(self):
        predictor = TopiaryPredictor(
            models=RandomBindingPredictor(
                alleles=["HLA-A*02:01"], default_peptide_lengths=[9],
            ),
        )
        df = predictor.predict_from_named_sequences(
            {"prot": "MASIINFEKLGGG"},
        )
        assert "self_nearest_peptide" not in df.columns

    def test_self_nearest_on_fragment_path(self):
        """Verify self_nearest_* columns appear on predict_from_fragments
        output (the fragment path goes through _finalize_rows →
        _apply_filter → _maybe_attach_self_nearest)."""
        from topiary import ProteinFragment

        ref = SelfProteome.from_peptides(
            {"g": "MASIINFEKLGGG"}, peptide_lengths=[9],
        )
        predictor = TopiaryPredictor(
            models=RandomBindingPredictor(
                alleles=["HLA-A*02:01"], default_peptide_lengths=[9],
            ),
            self_proteome=ref,
        )
        fragments = [
            ProteinFragment(
                fragment_id="test_frag",
                sequence="MASIINFEKLGGG",
            ),
        ]
        df = predictor.predict_from_fragments(fragments)
        assert "self_nearest_peptide" in df.columns
        assert "self_nearest_edit_distance" in df.columns
        assert (df["self_nearest_edit_distance"] == 0).all()


# ---------------------------------------------------------------------------
# scope="protected_tissues"
# ---------------------------------------------------------------------------


class TestProtectedTissuesScope:
    """Test scope="protected_tissues" — human via pirlygenes defaults
    and non-human via explicit tissue_gene_ids."""

    def test_non_human_without_tissue_gene_ids_raises(self, monkeypatch):
        fake = MagicMock()
        fake.protein_ids.return_value = []
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake,
        )
        with pytest.raises(ValueError, match="human-only"):
            SelfProteome.from_ensembl(
                species="mouse", scope="protected_tissues",
            )

    def test_non_human_with_explicit_gene_ids_works(self, monkeypatch):
        """Non-human users pass tissue_gene_ids= to supply their own
        gene set (e.g. from Tabula Muris) — bypasses pirlygenes."""
        fake = MagicMock()
        fake.protein_ids.return_value = ["PROT_A", "PROT_B"]
        fake.gene_id_of_protein_id.side_effect = {
            "PROT_A": "GENE_KEEP",
            "PROT_B": "GENE_DROP",
        }.get
        fake.transcript_id_of_protein_id.side_effect = {
            "PROT_A": "TX_KEEP",
            "PROT_B": "TX_DROP",
        }.get
        fake.protein_sequence.side_effect = {
            "PROT_A": "MASIINFEKLGGG",
            "PROT_B": "QRSTVWYACDEFGH",
        }.get
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake,
        )
        ref = SelfProteome.from_ensembl(
            species="mouse",
            release=102,
            peptide_lengths=[9],
            scope="protected_tissues",
            tissue_gene_ids={"GENE_KEEP"},
        )
        # Only GENE_KEEP's proteins should survive the filter.
        assert ref.n_reference_peptides == 5  # 5 × 9-mers from "MASIINFEKLGGG"
        assert "protected_tissues" in ref.reference_version
        assert "sha256:" in ref.reference_version

    def test_human_default_tissues(self, monkeypatch):
        """Human scope='protected_tissues' with default tissues uses
        pirlygenes' tissue_expressed_gene_ids.  Mock it to avoid
        pirlygenes data load in CI."""
        keep_genes = {"GENE_A", "GENE_B"}
        monkeypatch.setattr(
            "topiary.sources.tissue_expressed_gene_ids",
            lambda tissues, min_ntpm=1.0: keep_genes,
        )
        fake = MagicMock()
        fake.protein_ids.return_value = ["P1", "P2"]
        fake.gene_id_of_protein_id.side_effect = {
            "P1": "GENE_A",
            "P2": "GENE_C",
        }.get
        fake.transcript_id_of_protein_id.side_effect = {
            "P1": "TX_A",
            "P2": "TX_C",
        }.get
        fake.protein_sequence.side_effect = {
            "P1": "MASIINFEKLGGG",
            "P2": "QRSTVWYACDEFGH",
        }.get
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake,
        )
        ref = SelfProteome.from_ensembl(
            species="human",
            scope="protected_tissues",
            peptide_lengths=[9],
        )
        # Only GENE_A's proteins (GENE_C filtered out)
        assert ref.n_reference_peptides == 5
        assert "protected_tissues" in ref.reference_version
        assert "heart_muscle" in ref.reference_version

    def test_human_custom_tissues(self, monkeypatch):
        """Human with explicit tissues= list (not the default)."""
        monkeypatch.setattr(
            "topiary.sources.tissue_expressed_gene_ids",
            lambda tissues, min_ntpm=1.0: {"GENE_X"},
        )
        fake = MagicMock()
        fake.protein_ids.return_value = ["PX"]
        fake.gene_id_of_protein_id.side_effect = lambda pid: "GENE_X"
        fake.transcript_id_of_protein_id.side_effect = lambda pid: "TX_X"
        fake.protein_sequence.side_effect = lambda pid: "SIINFEKLA"
        monkeypatch.setattr(
            "pyensembl.EnsemblRelease",
            lambda release=None, species=None: fake,
        )
        ref = SelfProteome.from_ensembl(
            species="human",
            scope="protected_tissues",
            tissues=["lung", "brain"],
            peptide_lengths=[9],
        )
        assert ref.n_reference_peptides == 1
        assert "lung" in ref.reference_version
        assert "brain" in ref.reference_version

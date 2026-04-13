# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from .filters import (
    apply_effect_expression_filters,
    apply_variant_expression_filters,
    filter_silent_and_noncoding_effects,
)
from .ranking import (
    DSLNode,
    KindAccessor,
    apply_filter,
    apply_sort,
    parse,
)
from .sequence_helpers import (
    check_padding_around_mutation,
    contains_mutant_residues,
    peptide_mutation_interval,
    protein_subsequences_around_mutations,
)


_JOIN_COLUMNS = {
    "gene": "gene_id",
    "transcript": "transcript_id",
    "variant": "variant",
}


def _build_model_lookup():
    """Build a lowercase name → mhctools predictor class mapping."""
    import inspect
    import mhctools

    lookup = {}
    for attr_name, obj in inspect.getmembers(mhctools):
        if inspect.isclass(obj) and hasattr(obj, "predict_peptides_dataframe"):
            lookup[attr_name.lower()] = obj
    return lookup


_MODEL_LOOKUP = None


def _resolve_model_name(name):
    """Resolve a string model name to an mhctools predictor class.

    Supports case-insensitive matching against mhctools class names,
    e.g. ``"netmhcpan41"`` → ``NetMHCpan41``, ``"mhcflurry"`` → ``MHCflurry``.
    """
    global _MODEL_LOOKUP
    if _MODEL_LOOKUP is None:
        _MODEL_LOOKUP = _build_model_lookup()

    key = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    cls = _MODEL_LOOKUP.get(key)
    if cls is None:
        cls = _MODEL_LOOKUP.get(name.lower())
    if cls is None:
        available = sorted(_MODEL_LOOKUP.keys())
        raise ValueError(
            f"Unknown model name {name!r}. Available: {available}"
        )
    return cls


def _transcript_expression_dict_from_data(expression_data):
    """Extract a transcript_id -> expression dict from new-style expression data.

    Uses the first transcript-level source's first value column.
    Returns None if no transcript expression data is available.
    """
    transcript_sources = expression_data.get("transcript", [])
    if not transcript_sources:
        return None
    _name_prefix, id_col, df = transcript_sources[0]
    value_cols = [c for c in df.columns if c != id_col]
    if not value_cols:
        return None
    return dict(zip(df[id_col], df[value_cols[0]]))


def _attach_expression_data(df, expression_data):
    """Join expression DataFrames onto prediction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction DataFrame with gene_id/transcript_id/variant columns.
    expression_data : dict
        Keys: 'gene', 'transcript', 'variant'. Values: list of
        (name_prefix, id_col, DataFrame) tuples from expression_data_from_args.
    """
    for level, join_col in _JOIN_COLUMNS.items():
        for name_prefix, id_col, expr_df in expression_data.get(level, []):
            if join_col not in df.columns:
                logging.warning(
                    "Cannot join %s-level expression: column %r not in "
                    "predictions (available: %s)",
                    level, join_col, sorted(df.columns.tolist()),
                )
                continue
            # Rename ID column in expression data to match the join column
            merge_df = expr_df.rename(columns={id_col: join_col})
            # Aggregate duplicate join keys — sum numeric columns so that
            # e.g. multiple transcripts per gene have their TPM summed.
            if merge_df[join_col].duplicated().any():
                n_dupes = merge_df[join_col].duplicated().sum()
                logging.warning(
                    "%s-level expression (%s) has %d duplicate %s values; "
                    "summing numeric columns per %s",
                    level, name_prefix or "unnamed", n_dupes,
                    join_col, join_col,
                )
                numeric_cols = merge_df.select_dtypes(include="number").columns
                agg = {c: "sum" for c in numeric_cols}
                non_numeric = [
                    c for c in merge_df.columns
                    if c != join_col and c not in numeric_cols
                ]
                for c in non_numeric:
                    agg[c] = "first"
                merge_df = merge_df.groupby(join_col, sort=False).agg(agg).reset_index()
            # Prefix value columns with name_prefix if provided.
            # Column names are lowercased so that e.g. Salmon's "TPM"
            # becomes "gene_tpm", matching the documented ranking syntax.
            if name_prefix:
                for col in merge_df.columns:
                    if col != join_col:
                        col_lower = col.lower()
                        if col_lower.startswith(name_prefix):
                            new_name = col_lower
                        else:
                            new_name = f"{name_prefix}_{col_lower}"
                        merge_df = merge_df.rename(columns={col: new_name})
            # Left join — keep all prediction rows, fill missing with NaN
            n_before = len(df)
            df = df.merge(merge_df, on=join_col, how="left")
            n_matched = df[merge_df.columns[-1]].notna().sum()
            logging.info(
                "Joined %s-level expression (%s): %d/%d rows matched",
                level, name_prefix or "unnamed", n_matched, n_before,
            )
    return df


def _coerce_filter_node(expr):
    """Return a DSLNode for *expr* (string → parsed, KindAccessor → .value)."""
    if expr is None:
        return None
    if isinstance(expr, str):
        return parse(expr)
    if isinstance(expr, KindAccessor):
        return expr.value
    if isinstance(expr, DSLNode):
        return expr
    raise TypeError(
        f"Expected a DSL expression or string, got {type(expr).__name__}"
    )


def _coerce_sort_nodes(expr):
    """Return a list[DSLNode] for *expr*."""
    if expr is None:
        return []
    if isinstance(expr, (list, tuple)):
        return [_coerce_filter_node(e) for e in expr]
    return [_coerce_filter_node(expr)]


class TopiaryPredictor(object):
    def __init__(
        self,
        models=None,
        alleles=None,
        filter_by=None,
        sort_by=None,
        sort_direction="auto",
        padding_around_mutation=None,
        only_novel_epitopes=False,
        min_gene_expression=0.0,
        min_transcript_expression=0.0,
        raise_on_error=True,
        mhc_model=None,
        mhc_models=None,
    ):
        """
        Parameters
        ----------
        models : class, instance, or list
            Predictor model(s). Can be:

            - A model class or list of classes (requires ``alleles``)::

                  TopiaryPredictor(models=[NetMHCpan, MHCflurry], alleles=["A0201"])

            - A model instance or list of instances::

                  TopiaryPredictor(models=NetMHCpan(alleles=["A0201"]))

        alleles : list of str, optional
            HLA alleles. When provided, model classes in ``models`` are
            instantiated with these alleles.

        filter_by : DSLNode or str, optional
            Boolean filter expression. Accepts a parsed DSL node or a
            string that will be parsed::

                filter_by=(Affinity <= 500) | (Presentation.rank <= 2.0)
                filter_by="affinity <= 500 | el.rank <= 2"

        sort_by : DSLNode or list of DSLNode, optional
            Sort expression(s). Multiple expressions act as
            lexicographic tie breakers with missing values falling
            through to later keys::

                sort_by=[Presentation.score, Affinity.score]

            Or a composite::

                sort_by=0.5 * Affinity.score + 0.5 * Presentation.score

        sort_direction : "auto", "asc", or "desc"
            Direction for every sort key; "auto" infers per-key
            (asc for percentile_rank and raw affinity, desc otherwise).

        padding_around_mutation : int, optional
            Residues around a mutation to include in candidate epitopes.

        only_novel_epitopes : bool
            Drop peptides that do not contain mutated residues.

        min_gene_expression : float
            Minimum gene FPKM to keep a variant effect.

        min_transcript_expression : float
            Minimum transcript FPKM to keep a variant effect.

        raise_on_error : bool
            Raise on variant-effect errors vs. skip.

        mhc_model, mhc_models : legacy aliases for ``models``.
        """
        # --- model setup ---
        raw_models = models or mhc_models or (mhc_model and [mhc_model])
        if raw_models is None:
            raise ValueError("Must provide models")
        if not isinstance(raw_models, (list, tuple)):
            raw_models = [raw_models]

        self.models = []
        for m in raw_models:
            if isinstance(m, str):
                m = _resolve_model_name(m)
            if isinstance(m, type):
                if alleles is None:
                    raise ValueError(
                        f"alleles required when passing model class {m.__name__}"
                    )
                self.models.append(m(alleles=alleles))
            else:
                self.models.append(m)

        # Padding uses the union of all models' peptide lengths
        all_lengths = set()
        for m in self.models:
            all_lengths.update(m.default_peptide_lengths)
        self.padding_around_mutation = check_padding_around_mutation(
            given_padding=padding_around_mutation,
            epitope_lengths=sorted(all_lengths),
        )

        # --- filter / sort ---
        self.filter_by = _coerce_filter_node(filter_by)
        self.sort_by = _coerce_sort_nodes(sort_by)
        self.sort_direction = sort_direction

        self.min_transcript_expression = min_transcript_expression
        self.min_gene_expression = min_gene_expression
        self.only_novel_epitopes = only_novel_epitopes
        self.raise_on_error = raise_on_error

    @property
    def mhc_model(self):
        """Backward-compatible access to the first (or only) model."""
        return self.models[0]

    @property
    def mhc_models(self):
        """Backward-compatible alias for ``models``."""
        return self.models

    # ------------------------------------------------------------------
    # Prediction entry-points
    # ------------------------------------------------------------------

    def predict_from_named_sequences(self, name_to_sequence_dict):
        """
        Parameters
        ----------
        name_to_sequence_dict : dict (str -> str)
            Mapping of sequence names to amino acid sequences.

        Returns
        -------
        pandas.DataFrame with columns:
            source_sequence_name, peptide, peptide_offset, peptide_length,
            allele, kind, score, value, affinity, percentile_rank,
            prediction_method_name, predictor_version, n_flank, c_flank
        """
        df = self._predict_raw(name_to_sequence_dict)
        return self._apply_filter(df)

    def predict_from_named_peptides(self, name_to_peptide_dict):
        """
        Parameters
        ----------
        name_to_peptide_dict : dict (str -> str)
            Mapping of peptide names to amino acid sequences.

        Returns
        -------
        pandas.DataFrame with columns:
            source_sequence_name, peptide, peptide_offset, peptide_length,
            allele, kind, score, value, affinity, percentile_rank,
            prediction_method_name, predictor_version, n_flank, c_flank
        """
        df = self._predict_raw_peptides(name_to_peptide_dict)
        return self._apply_filter(df)

    def _predict_raw(self, name_to_sequence_dict):
        """Run models and format output, without applying filter/ranking."""
        dfs = []
        for model in self.models:
            model_df = model.predict_proteins_dataframe(name_to_sequence_dict)
            dfs.append(self._format_prediction_df(model_df))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _predict_raw_peptides(self, name_to_peptide_dict):
        """Run models on peptides as-is, without sliding-window scanning."""
        peptide_names_df = pd.DataFrame(
            {
                "source_sequence_name": list(name_to_peptide_dict.keys()),
                "peptide": list(name_to_peptide_dict.values()),
            }
        )
        if peptide_names_df.empty:
            return pd.DataFrame()

        peptide_list = peptide_names_df["peptide"].drop_duplicates().tolist()
        dfs = []
        for model in self.models:
            if hasattr(model, "predict_dataframe"):
                model_df = model.predict_dataframe(peptide_list)
            else:
                model_df = model.predict_peptides_dataframe(peptide_list)
            expanded_df = self._expand_named_peptide_predictions(
                model_df, peptide_names_df
            )
            dfs.append(self._format_prediction_df(expanded_df))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _expand_named_peptide_predictions(self, model_df, peptide_names_df):
        """Attach the original peptide names to model predictions."""
        if model_df.empty:
            return model_df.copy()

        expanded_df = model_df.drop(
            columns=["source_sequence_name"], errors="ignore"
        ).merge(peptide_names_df, on="peptide", how="inner")

        if "offset" in expanded_df.columns:
            expanded_df["offset"] = 0
        return expanded_df

    def _format_prediction_df(self, df):
        """Normalize mhctools prediction output to Topiary's schema."""
        if df.empty:
            return df.copy()

        df = df.rename(columns={
            "offset": "peptide_offset",
            "predictor_name": "prediction_method_name",
        }).copy()
        if "source_sequence_name" not in df.columns:
            df["source_sequence_name"] = None
        if "peptide_offset" not in df.columns:
            df["peptide_offset"] = 0
        df["peptide_length"] = df["peptide"].str.len()
        if "affinity" not in df.columns:
            df["affinity"] = np.where(
                df["kind"] == "pMHC_affinity", df["value"], np.nan
            )
        return df

    def _apply_filter(self, df):
        """Apply filter and sort if configured."""
        if df.empty:
            return df
        if self.filter_by is not None:
            df = apply_filter(df, self.filter_by)
        if self.sort_by:
            df = apply_sort(df, self.sort_by, sort_direction=self.sort_direction)
        return df

    def predict_from_antigens(self, fragments):
        """Predict MHC binding for peptides derived from a collection of
        :class:`AntigenFragment`.

        Each fragment's ``sequence`` is scanned with the configured
        models' sliding windows.  Fragment-level metadata
        (``source_type``, ``variant``, ``effect``, ``effect_type``,
        ``gene``, ``gene_id``, ``transcript_id``,
        ``gene_expression``, ``transcript_expression``, and every
        annotation key) is propagated onto each prediction row.
        ``fragment_id`` is threaded through so downstream code
        (vaxrank, vaccine-window selection) can group peptides back to
        their source fragment.

        Additional computed columns on the output:
        - ``overlaps_target`` (bool / NaN) — whether the peptide
          overlaps any interval in ``fragment.target_intervals``.
          NaN when ``target_intervals is None``.
        - ``contains_mutant_residues`` (bool / NaN) — backwards-compat
          alias: True iff the fragment's ``source_type`` starts with
          ``variant`` and the peptide overlaps a target interval.
        - ``wt_peptide`` / ``wt_peptide_length`` — derived by slicing
          ``fragment.effective_baseline`` at the mutant peptide's
          offset (germline precedence, falls back to reference).
          Only populated when the baseline is the same length as the
          mutant sequence (substitution-compatible); indels and
          frameshifts yield ``None`` until coordinate remapping lands.
          ``None`` / NaN when no baseline is present.

        WT model predictions (``wt_value``, ``wt_score``, etc.) are
        **not populated** in this release — populate them externally
        or wait for a follow-up PR.  The DSL's ``wt.*`` scope returns
        NaN for those columns until they're written.
        """
        fragments = list(fragments)
        if not fragments:
            return pd.DataFrame()

        name_to_seq = {f.fragment_id: f.sequence for f in fragments}
        df = self._predict_raw(name_to_seq)
        if df.empty:
            return df

        df = df.rename(columns={"source_sequence_name": "fragment_id"})
        # Keep source_sequence_name populated for any code path that
        # still looks at it; fragment_id is the canonical group key.
        df["source_sequence_name"] = df["fragment_id"]

        by_id = {f.fragment_id: f for f in fragments}

        def _map_attr(attr):
            return df["fragment_id"].map(
                lambda fid, a=attr: getattr(by_id[fid], a, None) if fid in by_id else None
            )

        for attr in (
            "source_type", "variant", "effect", "effect_type",
            "gene", "gene_id", "transcript_id",
            "gene_expression", "transcript_expression",
        ):
            df[attr] = _map_attr(attr)

        def _overlaps(row):
            f = by_id.get(row["fragment_id"])
            if f is None or f.target_intervals is None:
                return None
            return f.peptide_overlaps_target(
                int(row["peptide_offset"]), int(row["peptide_length"])
            )

        df["overlaps_target"] = df.apply(_overlaps, axis=1)

        def _contains_mutant(row):
            f = by_id.get(row["fragment_id"])
            if f is None or not f.source_type or not f.source_type.startswith("variant"):
                return None
            if f.target_intervals is None:
                return None
            return f.peptide_overlaps_target(
                int(row["peptide_offset"]), int(row["peptide_length"])
            )

        df["contains_mutant_residues"] = df.apply(_contains_mutant, axis=1)

        def _wt_peptide(row):
            f = by_id.get(row["fragment_id"])
            if f is None:
                return None
            base = f.effective_baseline
            if base is None:
                return None
            # Only meaningful for substitution-compatible fragments where
            # mutant and baseline coordinates align 1:1.  Indels and
            # frameshifts need explicit remapping, which PR A does not do.
            if len(base) != len(f.sequence):
                return None
            start = int(row["peptide_offset"])
            end = start + int(row["peptide_length"])
            if end > len(base):
                return None
            return base[start:end]

        df["wt_peptide"] = df.apply(_wt_peptide, axis=1)
        df["wt_peptide_length"] = df["wt_peptide"].map(
            lambda p: len(p) if isinstance(p, str) else None
        )

        all_annotation_keys = set()
        for f in fragments:
            all_annotation_keys.update(f.annotations.keys())
        for key in sorted(all_annotation_keys):
            if key in df.columns:
                continue
            df[key] = df["fragment_id"].map(
                lambda fid, k=key: by_id[fid].annotations.get(k)
                if fid in by_id else None
            )

        df = self._apply_filter(df)

        if self.only_novel_epitopes:
            df = df[df["contains_mutant_residues"].eq(True)]

        return df.reset_index(drop=True)

    def predict_from_sequences(self, sequences):
        """
        Predict MHC ligands for sub-sequences of each input sequence.

        Parameters
        ----------
        sequences : list of str
        """
        sequence_dict = {seq: seq for seq in sequences}
        df = self.predict_from_named_sequences(sequence_dict)
        return df.rename(columns={"source_sequence_name": "source_sequence"})

    def predict_from_mutation_effects(
        self, effects, transcript_expression_dict=None, gene_expression_dict=None,
        expression_data=None,
    ):
        """Given a Varcode.EffectCollection of predicted protein effects,
        return predicted epitopes around each mutation.

        Parameters
        ----------
        effects : Varcode.EffectCollection

        transcript_expression_dict : dict, optional
            Transcript ID -> RNA expression estimates (deprecated).

        gene_expression_dict : dict, optional
            Gene ID -> RNA expression estimates (deprecated).

        expression_data : dict, optional
            From expression_data_from_args(). Keys: 'gene', 'transcript',
            'variant', each mapping to list of (name, id_col, DataFrame).

        Returns
        -------
        pandas.DataFrame
        """
        effects = filter_silent_and_noncoding_effects(effects)

        effects = apply_effect_expression_filters(
            effects,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=self.min_transcript_expression,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=self.min_gene_expression,
        )

        variant_effect_groups = effects.groupby_variant()

        if len(variant_effect_groups) == 0:
            logging.warning("No candidates for MHC binding prediction")
            return pd.DataFrame()

        # Derive a transcript expression dict from new-style data when
        # the legacy dict is absent, so that transcript selection stays
        # expression-aware after migrating to --transcript-expression.
        effective_transcript_expr = transcript_expression_dict
        if not effective_transcript_expr and expression_data:
            effective_transcript_expr = _transcript_expression_dict_from_data(
                expression_data
            )

        if effective_transcript_expr:
            top_effects = [
                variant_effects.top_expression_effect(effective_transcript_expr)
                for variant_effects in variant_effect_groups.values()
            ]
        else:
            top_effects = [
                variant_effects.top_priority_effect()
                for variant_effects in variant_effect_groups.values()
            ]

        effect_to_subsequence_dict, effect_to_offset_dict = (
            protein_subsequences_around_mutations(
                effects=top_effects,
                padding_around_mutation=self.padding_around_mutation,
            )
        )

        variant_string_to_effect_dict = {
            effect.variant.short_description: effect
            for effect in effect_to_subsequence_dict.keys()
        }
        variant_string_to_subsequence_dict = {
            effect.variant.short_description: subseq
            for (effect, subseq) in effect_to_subsequence_dict.items()
        }
        variant_string_to_offset_dict = {
            effect.variant.short_description: subseq_offset
            for (effect, subseq_offset) in effect_to_offset_dict.items()
        }
        df = self._predict_raw(variant_string_to_subsequence_dict)
        logging.info(
            "MHC predictor returned %d peptide binding predictions" % (len(df))
        )

        # Rename source_sequence_name -> variant
        df = df.rename(columns={"source_sequence_name": "variant"})

        # Adjust offset to be relative to start of protein
        def compute_peptide_offset_relative_to_protein(row):
            subsequence_offset = variant_string_to_offset_dict[row.variant]
            return row.peptide_offset + subsequence_offset

        df["peptide_offset"] = df.apply(
            compute_peptide_offset_relative_to_protein, axis=1
        )

        # --- Annotate with variant/gene/transcript metadata ---
        # (must happen before ranking so expression columns are available)
        extra_columns = OrderedDict(
            [
                ("gene", []),
                ("gene_id", []),
                ("transcript_id", []),
                ("transcript_name", []),
                ("effect", []),
                ("effect_type", []),
                ("contains_mutant_residues", []),
                ("mutation_start_in_peptide", []),
                ("mutation_end_in_peptide", []),
            ]
        )
        if gene_expression_dict is not None:
            extra_columns["gene_expression"] = []
        if transcript_expression_dict is not None:
            extra_columns["transcript_expression"] = []

        for _, row in df.iterrows():
            effect = variant_string_to_effect_dict[row.variant]
            mutation_start_in_protein = effect.aa_mutation_start_offset
            mutation_end_in_protein = effect.aa_mutation_end_offset
            peptide_length = len(row.peptide)
            is_mutant = contains_mutant_residues(
                peptide_start_in_protein=row.peptide_offset,
                peptide_length=peptide_length,
                mutation_start_in_protein=mutation_start_in_protein,
                mutation_end_in_protein=mutation_end_in_protein,
            )
            if is_mutant:
                mutation_start_in_peptide, mutation_end_in_peptide = (
                    peptide_mutation_interval(
                        peptide_start_in_protein=row.peptide_offset,
                        peptide_length=peptide_length,
                        mutation_start_in_protein=mutation_start_in_protein,
                        mutation_end_in_protein=mutation_end_in_protein,
                    )
                )
            else:
                mutation_start_in_peptide = mutation_end_in_peptide = None

            extra_columns["gene"].append(effect.gene_name)
            gene_id = effect.gene_id
            extra_columns["gene_id"].append(gene_id)
            if gene_expression_dict is not None:
                extra_columns["gene_expression"].append(
                    gene_expression_dict.get(gene_id, 0.0)
                )

            transcript_id = effect.transcript_id
            extra_columns["transcript_id"].append(transcript_id)
            extra_columns["transcript_name"].append(effect.transcript_name)
            if transcript_expression_dict is not None:
                extra_columns["transcript_expression"].append(
                    transcript_expression_dict.get(transcript_id, 0.0)
                )

            extra_columns["effect"].append(effect.short_description)
            extra_columns["effect_type"].append(effect.__class__.__name__)
            extra_columns["contains_mutant_residues"].append(is_mutant)
            extra_columns["mutation_start_in_peptide"].append(mutation_start_in_peptide)
            extra_columns["mutation_end_in_peptide"].append(mutation_end_in_peptide)

        for col, values in extra_columns.items():
            df[col] = values

        # --- Join expression data (new-style --gene/transcript/variant-expression) ---
        if expression_data:
            df = _attach_expression_data(df, expression_data)

        # --- Apply filtering + sorting ---
        # (after annotation + expression join so all columns are available)
        if self.filter_by is not None:
            before = len(df)
            df = apply_filter(df, self.filter_by)
            logging.info(
                "Kept %d/%d predictions after applying filter" % (len(df), before)
            )
        if self.sort_by:
            df = apply_sort(df, self.sort_by, sort_direction=self.sort_direction)

        if self.only_novel_epitopes:
            df = df[df.contains_mutant_residues]

        return df

    def predict_from_variants(
        self, variants, transcript_expression_dict=None, gene_expression_dict=None,
        expression_data=None,
    ):
        """
        Predict epitopes from a Variant collection, filtering options, and
        optional gene and transcript expression data.

        Parameters
        ----------
        variants : varcode.VariantCollection

        transcript_expression_dict : dict, optional
            Maps from Ensembl transcript IDs to FPKM expression values
            (deprecated — use expression_data).

        gene_expression_dict : dict, optional
            Maps from Ensembl gene IDs to FPKM expression values
            (deprecated — use expression_data).

        expression_data : dict, optional
            From expression_data_from_args(). Keys: 'gene', 'transcript',
            'variant', each mapping to list of (name, id_col, DataFrame).

        Returns
        -------
        pandas.DataFrame
        """
        variants = apply_variant_expression_filters(
            variants,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=self.min_transcript_expression,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=self.min_gene_expression,
        )

        effects = variants.effects(raise_on_error=self.raise_on_error)

        return self.predict_from_mutation_effects(
            effects=effects,
            transcript_expression_dict=transcript_expression_dict,
            gene_expression_dict=gene_expression_dict,
            expression_data=expression_data,
        )

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
    EpitopeFilter,
    RankingStrategy,
    affinity_filter,
    apply_ranking_strategy,
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
            # Prefix value columns with name_prefix if provided
            if name_prefix:
                for col in merge_df.columns:
                    if col != join_col:
                        new_name = f"{name_prefix}_{col}" if not col.startswith(name_prefix) else col
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


class TopiaryPredictor(object):
    def __init__(
        self,
        models=None,
        alleles=None,
        filter_by=None,
        rank_by=None,
        padding_around_mutation=None,
        only_novel_epitopes=False,
        min_gene_expression=0.0,
        min_transcript_expression=0.0,
        raise_on_error=True,
        # backward-compat aliases
        filter=None,
        mhc_model=None,
        mhc_models=None,
        ic50_cutoff=None,
        percentile_cutoff=None,
        ranking=None,
        ranking_strategy=None,
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

        filter_by : EpitopeFilter, RankingStrategy, or str
            Which peptide-allele groups to keep. Accepts expression objects
            or a string that will be parsed::

                filter_by=(Affinity <= 500) | (Presentation.rank <= 2.0)
                filter_by="affinity <= 500 | el.rank <= 2"

        rank_by : Expr or list of Expr, optional
            How to sort surviving groups. First non-NaN wins::

                rank_by=[Presentation.score, Affinity.score]

            Or a composite expression::

                rank_by=0.5 * Affinity.score + 0.5 * Presentation.score

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

        filter : deprecated alias for ``filter_by``
        mhc_model : deprecated alias for ``models``
        mhc_models : deprecated alias for ``models``
        ic50_cutoff : deprecated, use ``filter_by=Affinity <= X``
        percentile_cutoff : deprecated, use ``filter_by=Affinity.rank <= X``
        ranking : deprecated alias for ``filter_by``
        ranking_strategy : deprecated alias for ``filter_by``
        """
        # --- model setup ---
        raw_models = models or mhc_models or (mhc_model and [mhc_model])
        if raw_models is None:
            raise ValueError("Must provide models")
        if not isinstance(raw_models, (list, tuple)):
            raw_models = [raw_models]

        self.models = []
        for m in raw_models:
            if isinstance(m, type):
                # It's a class — instantiate with alleles
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

        # --- filter / ranking ---
        effective_filter = filter_by or filter or ranking or ranking_strategy
        if isinstance(effective_filter, str):
            from .ranking import parse_ranking
            effective_filter = parse_ranking(effective_filter)
        if isinstance(effective_filter, EpitopeFilter):
            effective_filter = RankingStrategy(filters=[effective_filter])
        if effective_filter is not None:
            self.ranking_strategy = effective_filter
        elif ic50_cutoff or percentile_cutoff:
            self.ranking_strategy = RankingStrategy(
                filters=[affinity_filter(ic50_cutoff, percentile_cutoff)],
            )
        else:
            self.ranking_strategy = None

        # Attach rank_by to strategy if provided separately
        if rank_by is not None:
            if not isinstance(rank_by, (list, tuple)):
                rank_by = [rank_by]
            if self.ranking_strategy is None:
                self.ranking_strategy = RankingStrategy(sort_by=list(rank_by))
            else:
                self.ranking_strategy = RankingStrategy(
                    filters=list(self.ranking_strategy.filters),
                    require_all=self.ranking_strategy.require_all,
                    sort_by=list(rank_by),
                )

        self.ic50_cutoff = ic50_cutoff
        self.percentile_cutoff = percentile_cutoff
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
        """Apply ranking strategy filter/sort if configured."""
        if self.ranking_strategy and not df.empty:
            df = apply_ranking_strategy(df, self.ranking_strategy)
        return df

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

        if transcript_expression_dict:
            top_effects = [
                variant_effects.top_expression_effect(transcript_expression_dict)
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

        # --- Apply ranking/filtering ---
        # (after annotation + expression join so all columns are available)
        if self.ranking_strategy:
            df = apply_ranking_strategy(df, self.ranking_strategy)
            logging.info(
                "Kept %d predictions after applying ranking strategy" % len(df)
            )

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

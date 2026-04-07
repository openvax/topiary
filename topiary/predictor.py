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


class TopiaryPredictor(object):
    def __init__(
        self,
        mhc_model=None,
        mhc_models=None,
        padding_around_mutation=None,
        ic50_cutoff=None,
        percentile_cutoff=None,
        ranking_strategy=None,
        min_gene_expression=0.0,
        min_transcript_expression=0.0,
        only_novel_epitopes=False,
        raise_on_error=True,
    ):
        """
        Parameters
        ----------
        mhc_model : mhctools.BasePredictor, optional
            A single MHC binding predictor. Mutually exclusive with
            ``mhc_models``.

        mhc_models : list of mhctools.BasePredictor, optional
            Multiple MHC binding predictors whose results will be
            concatenated. Useful for combining e.g. NetMHCpan + MHCflurry.

        padding_around_mutation : int, optional
            How many residues surrounding a mutation to consider including
            in a candidate epitope. Default is the minimum size necessary
            for the epitope lengths of the model(s).

        ic50_cutoff : float, optional
            Maximum predicted IC50 value (nM) for a peptide to be kept.
            Ignored when ``ranking_strategy`` is provided.

        percentile_cutoff : float, optional
            Maximum percentile rank for a peptide to be kept.
            Ignored when ``ranking_strategy`` is provided.

        ranking_strategy : RankingStrategy, optional
            Rich filtering/ranking specification. When provided,
            ``ic50_cutoff`` and ``percentile_cutoff`` are ignored.

        min_gene_expression : float
            Minimum gene FPKM to keep a variant effect.

        min_transcript_expression : float
            Minimum transcript FPKM to keep a variant effect.

        only_novel_epitopes : bool
            If True, drop peptides that do not contain mutated residues.

        raise_on_error : bool
            Raise on variant-effect errors vs. skip.
        """
        # --- model setup ---
        if mhc_models is not None:
            self.mhc_models = list(mhc_models)
        elif mhc_model is not None:
            self.mhc_models = [mhc_model]
        else:
            raise ValueError("Must provide mhc_model or mhc_models")

        # Padding uses the union of all models' peptide lengths
        all_lengths = set()
        for m in self.mhc_models:
            all_lengths.update(m.default_peptide_lengths)
        self.padding_around_mutation = check_padding_around_mutation(
            given_padding=padding_around_mutation,
            epitope_lengths=sorted(all_lengths),
        )

        # --- ranking / filtering ---
        if ranking_strategy is not None:
            self.ranking_strategy = ranking_strategy
        elif ic50_cutoff or percentile_cutoff:
            self.ranking_strategy = RankingStrategy(
                filters=[affinity_filter(ic50_cutoff, percentile_cutoff)],
            )
        else:
            self.ranking_strategy = None

        self.ic50_cutoff = ic50_cutoff
        self.percentile_cutoff = percentile_cutoff
        self.min_transcript_expression = min_transcript_expression
        self.min_gene_expression = min_gene_expression
        self.only_novel_epitopes = only_novel_epitopes
        self.raise_on_error = raise_on_error

    @property
    def mhc_model(self):
        """Backward-compatible access to the first (or only) model."""
        return self.mhc_models[0]

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
        dfs = []
        for model in self.mhc_models:
            model_df = model.predict_proteins_dataframe(name_to_sequence_dict)
            dfs.append(model_df)
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)

        # Backward-compatible column renames / additions
        df = df.rename(columns={
            "offset": "peptide_offset",
            "predictor_name": "prediction_method_name",
        })
        df["peptide_length"] = df["peptide"].str.len()
        # "affinity" = IC50 value for pMHC_affinity rows, NaN otherwise
        df["affinity"] = np.where(
            df["kind"] == "pMHC_affinity", df["value"], np.nan
        )
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
        self, effects, transcript_expression_dict=None, gene_expression_dict=None
    ):
        """Given a Varcode.EffectCollection of predicted protein effects,
        return predicted epitopes around each mutation.

        Parameters
        ----------
        effects : Varcode.EffectCollection

        transcript_expression_dict : dict, optional
            Transcript ID -> RNA expression estimates.

        gene_expression_dict : dict, optional
            Gene ID -> RNA expression estimates.

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
            return []

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
        df = self.predict_from_named_sequences(variant_string_to_subsequence_dict)
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

        # --- Apply ranking/filtering ---
        if self.ranking_strategy:
            df = apply_ranking_strategy(df, self.ranking_strategy)
            logging.info(
                "Kept %d predictions after applying ranking strategy" % len(df)
            )

        # --- Annotate with variant/gene/transcript metadata ---
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

        if self.only_novel_epitopes:
            df = df[df.contains_mutant_residues]

        return df

    def predict_from_variants(
        self, variants, transcript_expression_dict=None, gene_expression_dict=None
    ):
        """
        Predict epitopes from a Variant collection, filtering options, and
        optional gene and transcript expression data.

        Parameters
        ----------
        variants : varcode.VariantCollection

        transcript_expression_dict : dict, optional
            Maps from Ensembl transcript IDs to FPKM expression values.

        gene_expression_dict : dict, optional
            Maps from Ensembl gene IDs to FPKM expression values.

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
        )

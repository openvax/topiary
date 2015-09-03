# Copyright (c) 2015. Mount Sinai School of Medicine
#
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

"""
Helper functions for filtering variants, effects, and epitope predictions
"""

from __future__ import print_function, division, absolute_import
import logging


from varcode import NonsilentCodingMutation

def apply_filter(
        filter_fn,
        collection,
        result_fn=None,
        filter_name="",
        collection_name=""):
    """
    Apply filter to effect collection and print number of dropped elements

    Parameters
    ----------
    """
    n_before = len(collection)
    filtered = [x for x in collection if filter_fn(x)]
    n_after = len(filtered)
    if not collection_name:
        collection_name = collection.__class__.__name__
    logging.info(
        "%s filtering removed %d/%d entries of %s",
        filter_name,
        (n_before - n_after),
        n_before,
        collection_name)
    return result_fn(filtered) if result_fn else collection.__class__(filtered)

def filter_silent_and_noncoding_effects(effects):
    """
    Keep only variant effects which result in modified proteins.

    Parameters
    ----------
    effects : varcode.EffectCollection
    """
    return apply_filter(
        filter_fn=lambda effect: isinstance(effect, NonsilentCodingMutation),
        collection=effects,
        result_fn=effects.clone_with_new_elements,
        filter_name="Silent mutation")


def apply_variant_expression_filters(
        variants,
        gene_expression_dict,
        gene_expression_threshold,
        transcript_expression_dict,
        transcript_expression_threshold):
    """
    Filter a collection of variants by gene and transcript expression thresholds

    Parameters
    ----------
    variants : varcode.VariantCollection

    gene_expression_dict : dict

    gene_expression_threshold : float

    transcript_expression_dict : dict

    transcript_expression_threshold : float
    """
    if gene_expression_dict:
        variants = apply_filter(
            lambda variant: any(
                gene_expression_dict.get(gene_id, 0.0) >=
                gene_expression_threshold
                for gene_id in variant.gene_ids
            ),
            variants,
            result_fn=variants.clone_with_new_elements,
            filter_name="Variant gene expression (min=%0.4f)" % gene_expression_threshold)
    if transcript_expression_dict:
        variants = apply_filter(
            lambda variant: any(
                transcript_expression_dict.get(transcript_id, 0.0) >=
                transcript_expression_threshold
                for transcript_id in variant.transcript_ids
            ),
            variants,
            result_fn=variants.clone_with_new_elements,
            filter_name="Variant transcript expression (min=%0.4f)" % transcript_expression_threshold)
    return variants

def apply_effect_expression_filters(
        effects,
        gene_expression_dict,
        gene_expression_threshold,
        transcript_expression_dict,
        transcript_expression_threshold):
    """
    Filter collection of varcode effects by given gene
    and transcript expression thresholds.

    Parameters
    ----------
    effects : varcode.EffectCollection

    gene_expression_dict : dict

    gene_expression_threshold : float

    transcript_expression_dict : dict

    transcript_expression_threshold : float
    """
    if gene_expression_dict:
        effects = apply_filter(
            lambda effect: (
                gene_expression_dict.get(effect.gene_id, 0.0) >=
                gene_expression_threshold),
            effects,
            result_fn=effects.clone_with_new_elements,
            filter_name="Effect gene expression (min = %0.4f)" % gene_expression_threshold)

    if transcript_expression_dict:
        effects = apply_filter(
            lambda effect: (
                transcript_expression_dict.get(effect.transcript_id, 0.0) >=
                transcript_expression_threshold
            ),
            effects,
            result_fn=effects.clone_with_new_elements,
            filter_name="Effect transcript expression (min=%0.4f)" % transcript_expression_threshold)
    return effects

def apply_epitope_filters(
        epitope_predictions,
        ic50_cutoff,
        percentile_cutoff,
        only_novel_epitopes):
    """
    Apply affinity and wildtype filters and create an EpitopeCollection
    from the remaining binding predictions.

    Parameters
    ----------
    epitope_predictions : mhctools.EpitopeCollection

    ic50_cutoff : float
        Highest allowed IC50 value
        (e.g. 25nM is a stronger binding value than 100nM)

    percentile_cutoff : float
        Highest allowed percentile of IC50 value
        (e.g. 1st percentile is a strong binder than 10th)

    only_novel_epitopes : bool
        If True, only keep epitopes that are mutated and don't appear elsewhere
        in the self-ligandome
    """
    # filter out low binders
    if ic50_cutoff:
        epitope_predictions = apply_filter(
            filter_fn=lambda x: x.value <= ic50_cutoff,
            collection=epitope_predictions,
            filter_name="IC50 nM cutoff",
            collection_name="epitope predictions",
        )
    if percentile_cutoff:
        epitope_predictions = apply_filter(
            filter_fn=lambda x: x.percentile_rank <= percentile_cutoff,
            collection=epitope_predictions,
            filter_name="IC50 percentile rank cutoff",
            collection_name="epitope predictions",
        )

    if only_novel_epitopes:
        epitope_predictions = apply_filter(
            filter_fn=lambda x: x.novel_epitope,
            collection=epitope_predictions,
            filter_name="Novel epitope",
            collection_name="epitope predictions",
        )
    return epitope_predictions

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

import logging


def apply_filter(
    filter_fn, collection, result_fn=None, filter_name="", collection_name=""
):
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
        collection_name,
    )
    return result_fn(filtered) if result_fn else collection.__class__(filtered)


def filter_silent_and_noncoding_effects(effects):
    """
    Keep only variant effects which result in modified proteins.

    Parameters
    ----------
    effects : varcode.EffectCollection
    """
    from varcode import iter_candidate_effects

    return apply_filter(
        filter_fn=lambda effect: any(
            candidate is not None and candidate.modifies_protein_sequence is True
            for candidate in iter_candidate_effects(effect)
        ),
        collection=effects,
        result_fn=effects.clone_with_new_elements,
        filter_name="Silent mutation",
    )

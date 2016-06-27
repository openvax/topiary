
MutantProteinSequence = namedtuple(
        "MutantProteinSequence", [
            # genomic variant that caused a mutant protein to be produced
            "variant",

            # what gene does the variant overlap?
            "gene_name",

            "gene_id"

            # transcript we're choosing to use for this variant
            "transcript_id",
            "transcript_name",

            # varcode Effect associated with the variant/transcript combination
            "effect",


             # where in the protein sequence did our prediction window start?
            "offset_in_protein",

            #
            "mutant_amino_acids",
            "reference_amino_acids",

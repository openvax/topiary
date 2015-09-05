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

from os import listdir
from os.path import join
from mhctools import normalize_allele_name

class AlleleNotFound(Exception):
    pass

class LazyLigandomeDict(object):
    """
    Object which acts like a dictionary mapping MHC allele names to
    sets of peptides. Backed by a directory of files, each of which
    has an allele name (e.g. "A0201") and contains one peptide per line.
    """

    def __init__(self, directory_path):
        """
        Parameters
        ----------
        directory_path : str
            Path to directory containing allele-specific peptide sets, each
            within a file named after its allele (e.g. "A0201")
        """
        self.directory_path = directory_path
        # maps normalize MHC allele names to absolute path of peptide set file
        self.peptide_set_paths = {}
        for filename in listdir(self.directory_path):
            allele_name = normalize_allele_name(filename)
            filepath = join(self.directory_path, filename)
            self.peptide_set_paths[allele_name] = filepath
        # cache peptide sets as we load them
        self.peptide_sets = {}

    def __getitem__(self, allele_name):
        print("Got %s" % allele_name)
        allele_name = normalize_allele_name(allele_name)
        print("Normalized to %s" % allele_name)
        if allele_name in self.peptide_sets:
            return self.peptide_sets[allele_name]

        filepath = self.peptide_set_paths.get(allele_name)
        if not filepath:
            raise AlleleNotFound(allele_name)
        with open(filepath, "r") as f:
            peptides = [l.strip().upper() for l in f.readlines()]
            # drop emptys strings
            peptides = set(p for p in peptides if p)
        self.peptide_sets[allele_name] = peptides
        return peptides

    def get(self, allele_name):
        try:
            return self[allele_name]
        except AlleleNotFound:
            return None


from topiary.lazy_ligandome_dict import LazyLigandomeDict, AlleleNotFound
from nose.tools import eq_, assert_raises

from .data import data_path


def test_lazy_ligandome_dict_allele_normalization():
    dirpath = data_path("tiny_test_ligandome_dir")
    ligandome = LazyLigandomeDict(dirpath)
    peptides_a0201 = ligandome["A0201"]
    peptides_hla_0201 = ligandome["HLA-A*02:01"]
    eq_(peptides_a0201, peptides_hla_0201)

    # tests that normalization works since name of file is different
    peptides_hla_b0704 = ligandome["b*0704"]
    eq_(peptides_hla_b0704, {"RRRRRRRRR"})


def test_lazy_ligandome_dict_missing_allele():
    dirpath = data_path("tiny_test_ligandome_dir")
    ligandome = LazyLigandomeDict(dirpath)
    with assert_raises(AlleleNotFound):
        ligandome["A0202"]


def test_lazy_ligandome_dict_get():
    dirpath = data_path("tiny_test_ligandome_dir")
    ligandome = LazyLigandomeDict(dirpath)
    assert ligandome.get("A*02:01") is not None, \
        "Expected A*02:01 to be in ligandome"
    a0202 = ligandome.get("A*02:02")
    assert a0202 is None, \
        "Did not expected A*02:02 to be in ligandome but got %s : %s" % (
            a0202, type(a0202))

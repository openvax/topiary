from topiary.cli.args import arg_parser
from topiary.cli.outputs import write_outputs
import tempfile
import pandas as pd
from nose.tools import eq_


def test_write_outputs():

    with tempfile.NamedTemporaryFile(mode="r+", delete=False) as f:
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [10, 20, 30]
        })
        args = arg_parser.parse_args([
            "--output-csv", f.name,
            "--subset-output-columns", "x",
            "--rename-output-column", "x", "X",
            "--mhc-predictor", "random",
            "--mhc-alleles", "A0201",
        ])

        write_outputs(
            df,
            args,
            print_df_before_filtering=True,
            print_df_after_filtering=True)
        print("File: %s" % f.name)
        df_from_file = pd.read_csv(f.name, index_col="#")

        df_expected = pd.DataFrame({
            "X": [1, 2, 3]})
        print(df_from_file)
        eq_(len(df_expected), len(df_from_file))
        assert (df_expected == df_from_file).all().all()

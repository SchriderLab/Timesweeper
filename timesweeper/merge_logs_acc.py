import pandas as pd
import argparse


def parse_ua():
    """Read in user arguments and sanitize inputs."""
    uap = argparse.ArgumentParser(
        description="Merges the summary TSV from SLiM logs with test data predictions."
    )

    uap.add_argument(
        "-s",
        "--summary-tsv",
        dest="summary_tsv",
        help="TSV created using `parse_slim_logs.py`.",
        required=True,
    )
    uap.add_argument(
        "-t",
        "--input-test-files",
        dest="test_files",
        help="Files to merge ",
        required=True,
    )
    uap.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT DIR",
        dest="output_dir",
        required=False,
        default=".",
        type=str,
        help="Directory to write images to.",
    )

    ua = uap.parse_args()

    return ua

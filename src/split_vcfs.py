from glob import glob
import argparse
import os
import itertools


def read_multivcf(input_vcf):
    """Reads in file and returns as list of strings."""
    with open(input_vcf, "r") as input_file:
        raw_lines = [i.strip() for i in input_file.readlines()]

    return raw_lines


def split_multivcf(vcf_lines, header):
    """Splits the lines of multi-vcf file into list of vcf entries by <header> using itertools."""
    header_idxs = [i for i in range(len(vcf_lines)) if vcf_lines[i] == header]

    split_vcfs = []
    for idx in range(len(header_idxs[:-1])):
        print(idx)
        split_vcfs.append(vcf_lines[header_idxs[idx] : header_idxs[idx + 1]])

    split_vcfs.append(vcf_lines[header_idxs[-1] :])

    return split_vcfs


def make_vcf_dir(input_vcf):
    """Creates directory named after vcf basename."""
    dirname = os.path.basename(input_vcf).split(".")[0]
    dirpath = os.path.dirname(input_vcf)
    vcf_dir = os.path.join(dirpath, dirname)
    os.makedirs(vcf_dir, exist_ok=True)

    return vcf_dir


def write_vcfs(vcf_lines, vcf_dir):
    """Writes list of vcf entries to numerically-sorted vcf files."""
    for idx, lines in enumerate(vcf_lines):
        with open(os.path.join(vcf_dir, f"{idx}.vcf"), "w") as outfile:
            outfile.writelines("\n".join(lines))


def get_ua():
    ap = argparse.ArgumentParser(
        description="Splits multi-vcf files from SLiM into a directory containing numerically-sorted time series vcf files."
    )
    ap.add_argument(
        "-i",
        "--input-vcf",
        required=True,
        type=str,
        dest="input_vcf",
        help="File containing multiple VCF entries from SLiM's `outputVCFSample` with `append=T`.",
    )
    ap.add_argument(
        "--vcf-header",
        required=False,
        type=str,
        default="##fileformat=VCFv4.2",
        dest="vcf_header",
        help="String that tops VCF header, used to split entries to new files.",
    )

    ap.add_argument(
        "--delete",
        required=False,
        action="store_true",
        dest="delete_vcf",
        help="Whether or not to delete the original multi-vcf file after splitting.",
    )

    return ap.parse_args()


def main():
    ua = get_ua()
    raw_lines = read_multivcf(ua.input_vcf)
    split_lines = split_multivcf(raw_lines, ua.vcf_header)
    # print(len(split_lines), len(split_lines[0]))
    vcf_dir = make_vcf_dir(ua.input_vcf)
    write_vcfs(split_lines, vcf_dir)

    if ua.delete_vcf:
        os.remove(ua.input_vcf)


if __name__ == "__main__":
    main()

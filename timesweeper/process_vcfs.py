import multiprocessing as mp
import os
import subprocess
from glob import glob
from itertools import cycle


# VCF Processing
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
        split_vcfs.append(vcf_lines[header_idxs[idx] : header_idxs[idx + 1]])

    split_vcfs.append(vcf_lines[header_idxs[-1] :])

    return split_vcfs


def write_vcfs(vcf_lines, vcf_dir):
    """Writes list of vcf entries to numerically-sorted vcf files."""
    for idx, lines in enumerate(vcf_lines):
        with open(os.path.join(vcf_dir, f"{idx}.vcf"), "w") as outfile:
            outfile.writelines("\n".join(lines))


def index_vcf(vcf):
    """
    Indexes and sorts vcf file.
    Commands are run separately such that processes complete before the next one starts.
    """
    bgzip_cmd = f"bgzip -c {vcf} > {vcf}.gz"
    tabix_cmd = f"tabix -f -p vcf {vcf}.gz"
    bcftools_cmd = f"bcftools sort -Ov {vcf}.gz | bgzip -f > {vcf}.sorted.gz"
    tabix_2_cmd = f"tabix -f -p vcf {vcf}.sorted.gz"
    subprocess.run(
        bgzip_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        tabix_cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        bcftools_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        tabix_2_cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )


def merge_vcfs(vcf_dir):
    num_files = len(glob(f"{vcf_dir}/*.vcf.sorted.gz"))
    if num_files == 1:
        cmd = f"""zcat {f"{vcf_dir}/0.vcf.sorted.gz"} > {vcf_dir}/merged.vcf"""

    else:
        cmd = f"""bcftools merge -Ov -0 \
                --force-samples --info-rules 'MT:join,S:join' \
                {" ".join([f"{vcf_dir}/{i}.vcf.sorted.gz" for i in range(num_files)])} > \
                {vcf_dir}/merged.vcf \
                """

    subprocess.run(cmd, shell=True)


def get_num_inds(vcf_file):
    num_ind = subprocess.check_output(
        """awk '{if ($1 == "#CHROM"){print NF-9; exit}}' """ + vcf_file, shell=True,
    )
    return int(num_ind)


def cleanup_intermed(vcf_dir):
    for ifile in glob(f"{vcf_dir}/*"):
        if "merged" not in ifile and "final" not in ifile:
            pass
            os.remove(ifile)


def make_vcf_dir(input_vcf):
    """Creates directory named after vcf basename."""
    dirname = os.path.basename(input_vcf).split(".")[0]
    dirpath = os.path.dirname(input_vcf)
    vcf_dir = os.path.join(dirpath, dirname)
    if os.path.exists(vcf_dir):
        for ifile in glob(f"{vcf_dir}/*"):
            os.remove(ifile)

    os.makedirs(vcf_dir, exist_ok=True)

    final_vcf = f"{input_vcf}.final"
    if os.path.exists(final_vcf):
        os.rename(final_vcf, f"{vcf_dir}/{final_vcf.split('/')[-1]}")

    return vcf_dir


def process_vcfs(input_vcf, num_tps):
    try:
        # Split into multiples after SLiM just concats to same file
        raw_lines = read_multivcf(input_vcf)
        split_lines = split_multivcf(raw_lines, "##fileformat=VCFv4.2")
        if len(split_lines) > 0:
            split_lines = split_lines[len(split_lines) - num_tps :]

            # Creates subdir for each rep
            vcf_dir = make_vcf_dir(input_vcf)
            write_vcfs(split_lines, vcf_dir)

            # Now index and merge
            [index_vcf(vcf) for vcf in glob(f"{vcf_dir}/*.vcf")]
            merge_vcfs(vcf_dir)

            cleanup_intermed(vcf_dir)

    except Exception as e:
        print(f"[ERROR] Couldn't process {e}")
        pass


def main(ua):
    vcflist = glob(f"{ua.in_dir}/*/*/*.multivcf")

    with mp.Pool(ua.threads) as p:
        p.map(vcflist, cycle([ua.num_tps]))

    print("[INFO] Done")

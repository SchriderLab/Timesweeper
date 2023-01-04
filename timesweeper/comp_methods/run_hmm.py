import argparse
import pickle as pkl
import subprocess
from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


def make_list_from_str(list_str):
    return [int(i) for i in list_str.strip("[").strip("]").split(", ")]


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p", "--test-preds", dest="test_pred_file", type=str, required=True
    )
    ap.add_argument(
        "-t", "--training-data", dest="training_data_file", type=str, required=True
    )
    ap.add_argument("-o", "--out-prefix", dest="out_prefix", type=str, required=True)
    ua = ap.parse_args()
    return ua


def main():
    """
    Pull in test data
    Pull in training data
    Extract all test data central snp
    Create config file for spectralHMM
    Run and collect results
    Parse results and tabulate

    rep	sweep	selCoeff	sampOffset	numRestarts	numSamples	seed	physLen	sampGens	selAlleleFreq	class	true_sel_coeff	pred_sel_coeff	abs_error
    19212	sdn	0.0110916521803371	2	16	20	711469270847202	5000000	[9998, 10009, 10019, 10030, 10040, 10051, 10061, 10072, 10083, 10093, 10104, 10114, 10125, 10136, 10146, 10157, 10167, 10178, 10188, 10199]	[0.001, 0.019, 0.035, 0.042, 0.066, 0.112, 0.171, 0.176, 0.226, 0.215, 0.195, 0.208, 0.189, 0.229, 0.352, 0.36, 0.44, 0.462, 0.565, 0.597]	sdn	0.0110917	0.04299812	0.031906420000000005

    """

    ua = get_ua()
    with open(ua.training_data_file, "rb") as pklfile:
        train_d = pkl.load(pklfile)

    test_preds = pd.read_csv(ua.test_pred_file, sep="\t")

    yearsPerGen = 25
    mutRate = 1e-7
    popSize = 500
    sampSize = 20

    multiplex_lines = []
    for idx in range(len(test_preds)):

        numDerived = [
            int(round(i * sampSize))
            for i in train_d[test_preds["sweep"][idx]][str(test_preds["rep"][idx])][
                "aft"
            ][:, 25]
        ]

        sampYears = [
            (i - 10000) * yearsPerGen
            for i in make_list_from_str(test_preds["sampGens"][idx])
        ]

        # print(sampYears)
        multiplex_lines.append(
            " ".join(
                [
                    f"({yr}.00, {ss}, {nd});"
                    for yr, ss, nd in zip(sampYears, cycle([sampSize]), numDerived)
                ]
            )
        )

    with open("tmp.config", "w") as tmpfile:
        tmpfile.writelines("\n".join(multiplex_lines))

    cmd = f"""python ../runSpectralHMM.py \
        --multiplex \
        --inputFile tmp.config \
        --mutToBenef {mutRate} \
        --mutFromBenef {mutRate} \
        --effPopSize {popSize} \
        --yearsPerGen {yearsPerGen} \
        --initFrequency 0.1 \
        --initTime -10000 \
        --hetF 0.0 \
        --homF 0.0 \
        --precision 40 \
        --matrixCutoff 150 \
        --maxM 140 \
        --maxN 130"""

    likelihoods = []
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, text=True)
    while (line := p.stdout.readline()) != "":
        line = line.strip()
        if "#" in line:
            pass
        else:
            likelihoods.append(float(line.split("\t")[-1]))

    test_preds["spectral_likelihoods"] = likelihoods

    ax = test_preds.plot(
        x="spectral_likelihoods",
        y="selCoeff",
        kind="scatter",
        title="SpectralHMM Likelihood vs Selection Coefficient",
        xlabel="SpectralHMM Likelihood",
        ylabel="Selection Coefficient",
        ylim=(0.0, 0.25),
    )
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2E"))
    plt.savefig(f"{ua.out_prefix}.spectral_vs_selcoeff.png")
    plt.clf()

    ax = test_preds.plot(
        x="spectral_likelihoods",
        y="abs_error",
        kind="scatter",
        title="Timesweeper Absolute Error vs SpectralHMM Likelihood",
        xlabel="SpectralHMM Likelihood",
        ylabel="Timesweeper Absolute Error",
        ylim=(0.0, 0.25),
    )
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2E"))

    plt.savefig(f"{ua.out_prefix}.tserr_vs_spectral.png")
    plt.clf()

    test_preds.to_csv(ua.out_prefix + ".tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()

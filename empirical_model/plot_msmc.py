import pandas as pd
import matplotlib.pyplot as plt
import sys

mu = 1.25e-8
gen = 30
plot_title = sys.argv[1]
infiles = sys.argv[2:]
for i in infiles:
    popData = pd.read_csv(i, delim_whitespace=True)
    plt.step(
        (popData["left_time_boundary"] / mu) * gen,
        (1 / popData["lambda_00"]) / (2 * mu),
        label=i.split("/")[-1].split(".")[0],
    )

plt.title(plot_title)
plt.legend()
plt.ylim(0, 1e5)
plt.xlim(1e2, 1e8)
plt.xlabel("years ago")
plt.ylabel("effective pop size")
plt.gca().set_xscale("log")
plt.savefig("results/msmc_times.png")

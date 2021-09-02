import pandas as pd
import matplotlib.pyplot as plt
import sys

mu = 1.25e-8
gen = 30
infilename = sys.argv[1]
singlePopData = pd.read_csv(infilename, delim_whitespace=True)
plt.step(
    (singlePopData["left_time_boundary"] / mu) * gen,
    (1 / singlePopData["lambda_00"]) / (2 * mu),
    label="Single Pop Times",
)

plt.title("Pop size over time")
plt.ylim(0, 1e5)
plt.xlim(1e2, 1e8)
plt.xlabel("years ago")
plt.ylabel("effective pop size")
plt.gca().set_xscale("log")
plt.savefig(f"""{infilename.split("/")[-1].split(".")[0]}.png""")
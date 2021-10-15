import numpy as np
from tqdm import tqdm
import sys

outfile = sys.argv[1]
filelist = sys.argv[2:]

# Collect all NPZ entries into a single file for the entire training dataset
data_all = []
for fname in tqdm(filelist, desc="Loading NPZ files for merging"):
    data_all.append(np.load(fname))
print("Data points to be merged:", len(data_all))
merged_data = {}
for data in tqdm(data_all, desc="Merging npz files"):
    for k, v in data.items():
        merged_data.update({k: v})
np.savez(outfile, **merged_data)

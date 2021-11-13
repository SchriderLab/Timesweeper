import numpy as np
from tqdm import tqdm
import sys
from collections import Counter

outfile = sys.argv[1]
filelist = sys.argv[2:]

# Collect all NPZ entries into a single file for the entire training dataset
data_all = []
for fname in tqdm(filelist, desc="Loading NPZ files for merging"):
    data_all.append(np.load(fname))
print("Data points to be merged:", len(data_all))
merged_data = {}
x_sizes = []
y_sizes = []
for data in tqdm(data_all, desc="Merging npz files"):
    for k, v in data.items():
        x_sizes.append(v.shape[0])
        y_sizes.append(v.shape[1])
        # v[0, :] = 1
        if "freq" in fname:
            if v.shape == ((51, 21)):
                merged_data.update({k: v[:, 1:]})
            elif v.shape == ((51, 22)):
                merged_data.update({k: v[:, 2:]})
            elif v.shape == ((51, 20)):
                merged_data.update({k: v})
            else:
                pass
        elif v.shape == ((20, 400)):
            merged_data.update({k: v})
        elif v.shape == ((21, 210)):
            merged_data.update({k: v[1:, 10:]})
        else:
            pass
        # else:
        #    print(f"{k} aint the right shape")

np.savez(outfile, **merged_data)

print("largest smallest x:", max(x_sizes), min(x_sizes))
print("largest smallest y:", max(y_sizes), min(y_sizes))
print(Counter(x_sizes).keys())
print(Counter(x_sizes).values())
print(Counter(y_sizes).keys())
print(Counter(y_sizes).values())


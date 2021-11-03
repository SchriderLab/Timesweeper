import numpy as np
import sys
import os
from tqdm import tqdm
import pandas as pd
from glob import glob

"""
- Take max and avg for FIT values in window
- Bin muts by 11 subwindows, m2 is in the center window
- 3 ROC curves: max across subwindows, mean, just true selected site

mut_ID,mut_type,location,window,fit_t,fit_p,selection_detected
79842,m1,45009,4,-1.6735540243396543,0.12006366972163422,0
"""


"""
Ask Dan about how to handle mean/pval/etc
"""

merged_file = sys.argv[1]
# cnn_preds_file = sys.argv[2]
fit_files = glob(os.path.join(sys.argv[1], "*", "pops", "fit", "*.fit"))
out_dir = sys.argv[2]

# cnn_preds = pd.read_csv(cnn_preds_file, header=0,)
# cnnIDs = list(cnn_preds["id"])

# Only grab fitfiles where sim was completed
# sample_ids = [i for i in np.load(merged_file).files]
# fitIDs = [i.replace("pop", "pop.fit") for i in sample_ids]


samp_dict = {
    "file": [],
    "window": [],
    "min_p_val": [],
    "min_p_detect": [],
    "true_site_soft": [],
    "true_site_hard": [],
}

for i in tqdm(fit_files, desc="Filling dictionary...",):
    try:
        fit_df = pd.read_csv(i, header=0).dropna().reset_index()
    except FileNotFoundError:
        print(i, "doesn't exist. Passing.")
        continue
    fit_df["window"] = fit_df["window"].astype(int)
    for win in pd.unique(fit_df["window"]):
        samp_dict["file"].append(i)
        win_sub = fit_df[fit_df["window"] == win]
        samp_dict["window"].append(win)
        samp_dict["min_p_val"].append(np.min(win_sub["fit_p"]))
        if samp_dict["min_p_val"][-1] <= 0.05:
            samp_dict["min_p_detect"].append(1)
        else:
            samp_dict["min_p_detect"].append(0)
        if "hard" in i and "m2" in list(win_sub["mut_type"]):
            samp_dict["true_site_hard"].append(1)
            samp_dict["true_site_soft"].append(0)
        elif "soft" in i and "m2" in list(win_sub["mut_type"]):
            samp_dict["true_site_hard"].append(0)
            samp_dict["true_site_soft"].append(1)
        else:
            samp_dict["true_site_hard"].append(0)
            samp_dict["true_site_soft"].append(0)

final_df = pd.DataFrame(samp_dict)
final_df.to_csv(os.path.join(out_dir, "fit_sample_dict.csv"), header=True, index=False)

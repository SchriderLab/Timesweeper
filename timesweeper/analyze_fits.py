import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import roc_curve, roc_auc_score
import plotting_utils as pu
import os
import matplotlib.pyplot as plt

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

data_dir = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int/sims/*/muts/*"
samp_dict = {
    "window": [],
    "min_p_val": [],
    "min_p_soft": [],
    "min_p_hard": [],
    "mean": [],
    "mean_detect_soft": [],
    "mean_detect_hard": [],
    "true_site_soft": [],
    "true_site_hard": [],
}

for i in glob(os.path.join(data_dir, "*fit")):
    print(i)

    fit_df = pd.read_csv(i, header=0).dropna().reset_index()

    fit_df["window"] = fit_df["window"].astype(int)

    for win in pd.unique(fit_df["window"]):
        win_sub = fit_df[fit_df["window"] == win]
        samp_dict["window"].append(win)
        samp_dict["min_p_val"].append(np.min(win_sub["fit_p"]))

        if samp_dict["min_p_val"][-1] <= 0.005:
            samp_dict["min_p_hard"].append(1)  # hard
            samp_dict["min_p_soft"].append(0)
        elif (samp_dict["min_p_val"][-1] <= 0.05) and (
            samp_dict["min_p_val"][-1] > 0.005
        ):
            samp_dict["min_p_hard"].append(0)  # soft
            samp_dict["min_p_soft"].append(1)  # soft
        else:
            samp_dict["min_p_hard"].append(0)
            samp_dict["min_p_soft"].append(0)

        samp_dict["mean"].append(np.mean(win_sub["fit_p"]))
        if samp_dict["mean"][-1] <= 0.005:
            samp_dict["mean_detect_hard"].append(1)  # hard
            samp_dict["mean_detect_soft"].append(0)
        elif (samp_dict["mean"][-1] <= 0.05) and (samp_dict["mean"][-1] > 0.005):
            samp_dict["mean_detect_hard"].append(0)  # soft
            samp_dict["mean_detect_soft"].append(1)
        else:
            samp_dict["mean_detect_hard"].append(0)
            samp_dict["mean_detect_soft"].append(0)

        if "hard" in i and "m2" in win["mut_type"]:
            samp_dict["true_site_hard"].append(1)
            samp_dict["true_site_soft"].append(0)
        elif "soft" in i and "m2" in win["mut_type"]:
            samp_dict["true_site_hard"].append(0)
            samp_dict["true_site_soft"].append(1)
        else:
            samp_dict["true_site_hard"].append(0)
            samp_dict["true_site_soft"].append(0)

print(samp_dict)

plt.figure()

fpr, tpr, thresh = roc_curve(samp_dict["true_site_soft"], samp_dict["min_p_soft"])
auc = roc_auc_score(samp_dict["true_site_soft"], samp_dict["min_p_soft"])
plt.plot(fpr, tpr, label="Min p val soft, auc=" + str(auc))

fpr, tpr, thresh = roc_curve(samp_dict["true_site_hard"], samp_dict["min_p_hard"])
auc = roc_auc_score(samp_dict["true_site_hard"], samp_dict["min_p_hard"])
plt.plot(fpr, tpr, label="Min p val hard, auc=" + str(auc))

fpr, tpr, thresh = roc_curve(samp_dict["true_site_soft"], samp_dict["mean_detect_soft"])
auc = roc_auc_score(samp_dict["true_site_soft"], samp_dict["mean_detect_soft"])
plt.plot(fpr, tpr, label="Mean p val soft, auc=" + str(auc))

fpr, tpr, thresh = roc_curve(samp_dict["true_site_hard"], samp_dict["mean_detect_hard"])
auc = roc_auc_score(samp_dict["true_site_hard"], samp_dict["mean_detect_hard"])
plt.plot(fpr, tpr, label="Mean p val hard, auc=" + str(auc))

plt.savefig("fits_rocs.png")
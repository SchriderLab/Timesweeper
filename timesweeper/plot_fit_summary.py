import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

fit_df = pd.read_csv(
    "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int/sample_dict.csv"
)


plt.figure()
plt.title("FIt ROC - All Types")
plt.xlabel("False Positive Rate")
plt.ylabel("True Postitive Rate")

fpr, tpr, thresh = roc_curve(fit_df["true_site_soft"], fit_df["min_p_soft"])
auc = roc_auc_score(fit_df["true_site_soft"], fit_df["min_p_soft"])
plt.plot(fpr, tpr, label="Min p val soft, auc=" + str(auc))

fpr, tpr, thresh = roc_curve(fit_df["true_site_hard"], fit_df["min_p_hard"])
auc = roc_auc_score(fit_df["true_site_hard"], fit_df["min_p_hard"])
plt.plot(fpr, tpr, label="Min p val hard, auc=" + str(auc))

fpr, tpr, thresh = roc_curve(fit_df["true_site_soft"], fit_df["mean_detect_soft"])
auc = roc_auc_score(fit_df["true_site_soft"], fit_df["mean_detect_soft"])
plt.plot(fpr, tpr, label="Mean p val soft, auc=" + str(auc))

fpr, tpr, thresh = roc_curve(fit_df["true_site_hard"], fit_df["mean_detect_hard"])
auc = roc_auc_score(fit_df["true_site_hard"], fit_df["mean_detect_hard"])
plt.plot(fpr, tpr, label="Mean p val hard, auc=" + str(auc))

plt.legend(loc="bottom right")

plt.savefig("fits_rocs.png")
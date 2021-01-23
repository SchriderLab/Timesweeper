import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

fit_df = pd.read_csv(
    "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int/sample_dict.csv"
)

cnn_df = pd.read_csv(
    "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/cnn_preds_rocformat.csv", header=0
)

plt.figure()
plt.title("FIt ROC - All Types")
plt.xlabel("False Positive Rate")
plt.ylabel("True Postitive Rate")


fpr, tpr, thresh = roc_curve(cnn_df["sweep_present"], cnn_df["sweep_predicted"])
print(thresh)
auc = roc_auc_score(cnn_df["sweep_present"], cnn_df["sweep_predicted"])
plt.plot(fpr, tpr, label="CNN Preds, auc=" + str(auc))

fpr, tpr, thresh = roc_curve(fit_df["sweep_present"], fit_df["sweep_predicted"])
print(thresh)
auc = roc_auc_score(fit_df["sweep_present"], fit_df["sweep_predicted"])
plt.plot(fpr, tpr, label="Min p val FIt, auc=" + str(auc))

plt.legend(loc="lower right")

plt.savefig("fits_rocs.png")

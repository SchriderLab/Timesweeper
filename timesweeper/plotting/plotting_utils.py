import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    r2_score,
)
from sklearn.preprocessing import label_binarize

plt.ioff()


def plot_confusion_matrix(
    working_dir, cm, target_names, title="Confusion matrix", cmap=None, normalize=False
):
    """
    Given a sklearn confusion matrix (cm), make a nice plot.

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                f"{cm[i, j]:0.4f}",
                horizontalalignment="center",
                color="black",
                # color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="black",
                # color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")

    plt.savefig(os.path.join(working_dir, title + ".pdf"))
    plt.savefig(os.path.join(working_dir, title + ".png"))
    plt.clf()


def plot_class_training(working_dir, history, model_save_name):
    """
    Plots training and validation accuracies

    Args:
        working_dir (str): Location to save model
        history (Keras history object): Model history after training and validation
        model_save_name (str): Name to use for title and name of plot

    Saves figure to file.
    """
    # Plot accuracy over validation accuracy during training
    plt.plot(history.history["accuracy"], label="class_accuracy")
    plt.plot(history.history["val_accuracy"], label="class_val_accuracy")
    plt.plot(history.history["loss"], label="class_loss")
    plt.plot(history.history["val_loss"], label="class_val_loss")

    plt.plot(history.history["loss"], label="total_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.legend(loc="upper left")
    plt.title(model_save_name)

    imgFile = os.path.join(working_dir, model_save_name + "_training.pdf")
    plt.savefig(imgFile)
    imgFile = os.path.join(working_dir, model_save_name + "_training.png")
    plt.savefig(imgFile)
    plt.clf()


def plot_reg_training(working_dir, history, model_save_name):
    """
    Plots training and validation accuracies

    Args:
        working_dir (str): Location to save model
        history (Keras history object): Model history after training and validation
        model_save_name (str): Name to use for title and name of plot

    Saves figure to file.
    """

    plt.plot(history.history["mse"], label="mse")
    plt.plot(history.history["val_mse"], label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend(loc="upper left")
    plt.title(model_save_name)

    imgFile = os.path.join(working_dir, model_save_name + "_reg_mse_training.pdf")
    plt.savefig(imgFile)
    imgFile = os.path.join(working_dir, model_save_name + "_reg_mse_training.png")
    plt.savefig(imgFile)
    plt.clf()


def print_classification_report(y_true, y_pred):
    """
    Prints classification report to stdout.

    Args:
        y_true (nparray): 1D npy array containing int values for class
        y_pred (nparray): 1D npy array containing int values for predicted class
        train_gen (Keras Generator): Training generator used for model training, used for labels
    """
    print("Classification Report")
    print(classification_report(y_true, y_pred))


def plot_roc(y_true, y_probs, schema, scenarios, outfile):
    for i1, s1 in enumerate(scenarios[1:], 1):
        for i2, s2 in enumerate(scenarios[1:], 1):
            if i1 != i2:
                print("i1", i1, "i2", i2)
                print(y_true.shape)
                # Plot sdn/ssv distinction
                sweep_idxs = np.where(y_true[(y_true == i1) | (y_true == i2)])
                print(sweep_idxs)
                sweep_labs = y_true[sweep_idxs]
                pos_probs = y_probs[sweep_idxs, i2].flatten()
                print(sweep_labs.shape)
                print(pos_probs.shape)

                swp_fpr, swp_tpr, thresh = roc_curve(sweep_labs, pos_probs, pos_label=i2)
                swp_auc_val = auc(swp_fpr, swp_tpr)
                plt.plot(
                    swp_fpr,
                    swp_tpr,
                    label=f"{schema.capitalize()} {s1.upper()} vs {s2.upper()}: {swp_auc_val:.4}",
                )

    # Coerce all ssvs into sweep binary pred
    labs = y_true
    labs[labs > 1] = 1
    pred_probs = np.sum(y_probs[:, 1:], axis=1)

    # Plot ROC Curve
    fpr, tpr, thresh = roc_curve(labs, pred_probs)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{schema.capitalize()} {scenarios[0].upper()} vs Sweep AUC: {auc_val:.2}")
    
    plt.title(f"ROC Curves {schema.upper()}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.savefig(outfile)
    plt.savefig(outfile + ".png")

    plt.clf()


def plot_prec_recall(
    y_true, y_probs, schema, scenarios, outfile
):
    for i1, s1 in enumerate(scenarios[1:], 1):
        for i2, s2 in enumerate(scenarios[1:], 1):
            if i1 != i2:
                print("i1", i1, "i2", i2)
                print(y_true.shape)
                # Plot sdn/ssv distinction
                sweep_idxs = np.where(y_true[(y_true == i1) | (y_true == i2)])
                print(sweep_idxs)
                sweep_labs = y_true[sweep_idxs]
                pos_probs = y_probs[sweep_idxs, i2].flatten()
                print(sweep_labs.shape)
                print(pos_probs.shape)

                swp_prec, swp_rec, thresh = precision_recall_curve(sweep_labs, pos_probs, pos_label=i2)
                swp_auc_val = auc(swp_rec, swp_prec)
                plt.plot(
                    swp_rec,
                    swp_prec,
                    label=f"{schema.capitalize()} {s1.upper()} vs {s2.upper()}: {swp_auc_val:.4}",
                )
    
    # Coerce all ssvs into sweep binary pred
    labs = y_true
    labs[labs > 1] = 1
    pred_probs = np.sum(y_probs[:, 1:], axis=1)

    # Plot ROC Curve
    prec, rec, thresh = precision_recall_curve(labs, pred_probs)
    auc_val = auc(rec, prec)
    plt.plot(rec, prec, label=f"{schema.capitalize()} {scenarios[0].upper()} vs Sweep AUC: {auc_val:.2}")
    
    plt.title(f"PR Curve {schema.upper()}")
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(outfile)
    plt.savefig(outfile + ".png")
    plt.clf()


def plot_sel_coeff_preds(true_class, s_true, s_pred, outfile, scenarios):
    """Plot s predictions against true value, color by scenario."""
    for g in np.unique(true_class):
        i = np.where(true_class == g)
        plt.scatter(s_true[i], s_pred[i], label=scenarios[g].upper())
        plt.annotate(
            f"r^2 of {scenarios[g].upper()}: {np.round(r2_score(s_true[i], s_pred[i]), 2)}",
            (0.05, 0.27),
        )

    plt.legend()
    plt.xlim((0, 0.3))
    plt.ylim((0, 0.3))
    plt.title("Predicted vs True Selection Coefficients")
    plt.ylabel("Predicted S")
    plt.xlabel("True S")
    plt.savefig(outfile)
    plt.savefig(outfile + ".png")
    plt.clf()


def plot_s_vs_freqs(s, freqs, scenario, work_dir, exp_name, mode):
    plt.scatter(s, freqs, label=scenario)
    plt.legend()
    plt.ylim((0, 1))
    plt.title("Freq Change vs S")
    plt.ylabel("Frequency Change")
    plt.xlabel("True S")
    plt.savefig(f"{work_dir}/images/{exp_name}_{scenario}_{mode}_freqchanges.png")
    plt.clf()

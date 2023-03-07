import logging
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import save_model

from timesweeper import models

from timesweeper.plotting import plotting_utils as pu
from timesweeper.utils.gen_utils import read_config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig()
logger = logging.getLogger("nets")
logger.setLevel("INFO")

seed = np.random.randint(1, 1e6)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
logger.info(f"Random seed is: {seed}")


def scale_sel_coeffs(sel_coef_arr):
    """Scale training data, return scaled data and scaler to use for test data."""
    scaler = MinMaxScaler().fit(sel_coef_arr)

    return scaler


def get_data(input_pickle, data_type, win_size):
    """
    Loads data from pickle file and returns as list of labels and data.

    Args:
        input_pickle (str): Path to pickle created with make_training_features module.
        data_type (str): Determines either hfs or aft files to search for.

    Returns:
        list[str]: List of sweep labels for each sample
        np.arr: Array with all data stacked.
    """
    id_list = []
    rep_list = []
    data_list = []
    sel_coeffs = []
    sweep_types = []
    offsets = []
    pikl_dict = pickle.load(open(input_pickle, "rb"))
    for sweep in pikl_dict.keys():
        sweep_types.append(sweep)
        for rep in pikl_dict[sweep].keys():
            try:
                data_list.append(np.array(pikl_dict[sweep][rep][data_type.lower()]))
            except:
                continue

            if sweep != "neut":
                if abs(pikl_dict[sweep][rep]["center_offset"]) > 0:
                    id_list.append("neut")
                else:
                    id_list.append(sweep)

            else:
                id_list.append(sweep)

            rep_list.append(rep)
            sel_coeffs.append(pikl_dict[sweep][rep]["sel_coeff"])
            offsets.append(pikl_dict[sweep][rep]["center_offset"])

    return (
        id_list,
        rep_list,
        np.stack(data_list),
        sweep_types,
        np.array(sel_coeffs).reshape(-1, 1),
        np.array(offsets).reshape(-1, 1),
    )


def split_partitions(data, labs, sel_coeffs, reps, offsets):
    """
        Splits all data and labels into partitions for train/val/testing.

    Args:
        data (np.arr): Data for training model.
        labs (List): List of numeric labels for IDs
        sel_coeffs (List[float]): List of selection coefficients for each rep

    Returns:
        Tuple[List[narr], List[narr], List[narr], List[int], List[int], List[int]]: Train/val/test splits of IDs and labs
    """
    (
        train_data,
        val_data,
        train_labs,
        val_labs,
        train_s,
        val_s,
        train_reps,
        val_reps,
        train_offsets,
        val_offsets,
    ) = train_test_split(
        data, labs, sel_coeffs, reps, offsets, stratify=labs, test_size=0.3
    )
    (
        val_data,
        test_data,
        val_labs,
        test_labs,
        val_s,
        test_s,
        val_reps,
        test_reps,
        val_offsets,
        test_offsets,
    ) = train_test_split(
        val_data,
        val_labs,
        val_s,
        val_reps,
        val_offsets,
        stratify=val_labs,
        test_size=0.5,
    )
    return (
        train_data,
        val_data,
        test_data,
        train_labs,
        val_labs,
        test_labs,
        train_s,
        val_s,
        test_s,
        train_reps,
        val_reps,
        test_reps,
        train_offsets,
        val_offsets,
        test_offsets,
    )


def fit_class_model(
    out_dir,
    model,
    data_type,
    train_data,
    train_labs,
    val_data,
    val_labs,
    experiment_name,
):
    """
    Fits a given model using training/validation data, plots history after done.
    Args:
        out_dir (str): Base directory where data is located, model will be saved here.
        model (Model): Compiled Keras model.
        data_type (str): Whether data is HFS or aft data.
        train_data (np.arr): Training data.
        train_labs (list[int]): OHE labels for training set.
        val_data (np.arr): Validation data.
        val_labs (list[int]): OHE labels for validation set.
        experiment_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.
    Returns:
        Model: Fitted Keras model, ready to be used for accuracy characterization.
    """
    monitor = "val_accuracy"
    train_target = train_labs
    val_target = val_labs

    if not os.path.exists(os.path.join(out_dir, "images")):
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    if not os.path.exists(os.path.join(out_dir, "trained_models")):
        os.makedirs(os.path.join(out_dir, "trained_models"), exist_ok=True)

    checkpoint = ModelCheckpoint(
        os.path.join(out_dir, "trained_models", f"{model.name}_{data_type}"),
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    earlystop = EarlyStopping(
        monitor=monitor,
        min_delta=0.1,
        patience=20,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    history = model.fit(
        x=train_data,
        y=train_target,
        epochs=100,
        verbose=2,
        callbacks=callbacks_list,
        validation_data=(val_data, val_target),
        # class_weight=class_weights,
    )

    pu.plot_class_training(
        os.path.join(out_dir, "images"),
        history,
        f"{experiment_name}_{model.name}_{data_type}",
    )

    # Won't checkpoint handle this?
    save_model(
        model,
        os.path.join(
            out_dir, "trained_models", f"{experiment_name}_{model.name}_{data_type}"
        ),
    )

    return model


def fit_reg_model(
    out_dir, model, data_type, train_data, train_s, val_data, val_s, experiment_name,
):
    """
    Fits a given model using training/validation data, plots history after done.

    Args:
        out_dir (str): Base directory where data is located, model will be saved here.
        model (Model): Compiled Keras model.
        data_type (str): Whether data is HFS or aft data.
        train_data (np.arr): Training data.
        train_labs (list[int]): OHE labels for training set.
        val_data (np.arr): Validation data.
        val_labs (list[int]): OHE labels for validation set.
        experiment_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.

    Returns:
        Model: Fitted Keras model, ready to be used for accuracy characterization.
    """
    monitor = "val_mse"
    train_target = train_s
    val_target = val_s

    if not os.path.exists(os.path.join(out_dir, "images")):
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    if not os.path.exists(os.path.join(out_dir, "trained_models")):
        os.makedirs(os.path.join(out_dir, "trained_models"), exist_ok=True)

    checkpoint = ModelCheckpoint(
        os.path.join(out_dir, "trained_models", f"{model.name}_{data_type}"),
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    earlystop = EarlyStopping(
        monitor=monitor,
        min_delta=0.01,
        patience=40,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    history = model.fit(
        x=train_data,
        y=train_target,
        epochs=100,
        verbose=2,
        callbacks=callbacks_list,
        validation_data=(val_data, val_target),
        # class_weight=class_weights,
    )

    pu.plot_reg_training(
        os.path.join(out_dir, "images"),
        history,
        f"{experiment_name}_{model.name}_{data_type}",
    )

    # Won't checkpoint handle this?
    save_model(
        model,
        os.path.join(
            out_dir, "trained_models", f"REG_{experiment_name}_{model.name}_{data_type}"
        ),
    )

    return model


def evaluate_reg_model(
    model,
    test_data,
    test_labs,
    trans_test_s,
    test_reps,
    test_offsets,
    s_scaler,
    out_dir,
    scenarios,
    experiment_name,
    data_type,
    lab_dict,
):
    """
    Evaluates model using confusion matrices and plots results.

    Args:
        model (Model): Fit Keras model.
        test_data (List[narr]): Testing data.
        test_labs (narr): Testing labels.
        test_s (narr): Selection coefficients to test against.
        out_dir (str): Base directory data is located in.
        scenarios (list[str]): Scenarios defined in config.
        experiment_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.
        data_type (str): Whether data is aft or hfs.
    """
    str_lab_dict = {value: key for key, value in lab_dict.items()}
    trans_pred_s = model.predict(test_data)
    trues = np.argmax(test_labs, axis=1)

    if "log" in experiment_name:
        pred_s = 10 ** (-trans_pred_s)
        test_s = 10 ** (-trans_test_s)
    elif "minmax" in experiment_name:
        pred_s = s_scaler.inverse_transform(trans_pred_s.reshape(-1, 1))
        test_s = s_scaler.inverse_transform(trans_test_s.reshape(-1, 1))
    else:
        pred_s = trans_pred_s
        test_s = trans_test_s

    pred_dict = {
        "rep": test_reps,
        "offset": test_offsets.flatten(),
        "class": [str_lab_dict[i] for i in trues],
        "true_sel_coeff": test_s.flatten(),
        "pred_sel_coeff": pred_s.flatten(),
    }

    pred_df = pd.DataFrame(pred_dict)

    os.makedirs(os.path.join(out_dir, "test_predictions"), exist_ok=True)
    pred_df.to_csv(
        os.path.join(
            out_dir,
            "test_predictions",
            f"{experiment_name}_{model.name}_{data_type}_selcoeff_test_predictions.csv",
        ),
        header=True,
        index=False,
    )

    logger.info(
        f"\nMean absolute error for Sel Coeff predictions: {mean_absolute_error(test_s, pred_s)}"
    )
    pu.plot_sel_coeff_preds(
        trues,
        test_s,
        pred_s,
        os.path.join(
            out_dir,
            "images",
            f"{experiment_name}_{model.name}_{data_type}_selcoeffs.pdf",
        ),
        scenarios,
    )


def evaluate_class_model(
    model,
    test_data,
    test_labs,
    test_reps,
    test_offsets,
    out_dir,
    scenarios,
    experiment_name,
    data_type,
    lab_dict,
):
    """
    Evaluates model using confusion matrices and plots results.

    Args:
        model (Model): Fit Keras model.
        test_data (List[narr]): Testing data.
        test_labs (narr): Testing labels.
        test_s (narr): Selection coefficients to test against.
        out_dir (str): Base directory data is located in.
        scenarios (list[str]): Scenarios defined in config.
        experiment_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.
        data_type (str): Whether data is aft or hfs.
    """
    str_lab_dict = {value: key for key, value in lab_dict.items()}

    class_probs = model.predict(test_data)
    class_predictions = np.argmax(class_probs, axis=1)
    trues = np.argmax(test_labs, axis=1)

    # Cannot for the life of me figure out why memory is shared b/t functions and this
    # So it gets casted twice to break that chain
    roc_trues = np.array(list(trues))
    pr_trues = np.array(list(trues))

    pred_dict = {
        "rep": test_reps,
        "offset": test_offsets.flatten(),
        "true": [str_lab_dict[i] for i in trues],
        "pred": [str_lab_dict[i] for i in class_predictions],
    }
    for str_lab in lab_dict:
        pred_dict[f"{str_lab}_scores"] = class_probs[:, lab_dict[str_lab]]

    pred_df = pd.DataFrame(pred_dict)

    os.makedirs(os.path.join(out_dir, "test_predictions"), exist_ok=True)
    pred_df.to_csv(
        os.path.join(
            out_dir,
            "test_predictions",
            f"{experiment_name}_{model.name}_{data_type}_class_test_predictions.csv",
        ),
        header=True,
        index=False,
    )

    lablist = [i.upper() for i in lab_dict]

    conf_mat = confusion_matrix(trues, class_predictions)

    pu.plot_confusion_matrix(
        os.path.join(out_dir, "images"),
        conf_mat,
        lablist,
        title=f"{experiment_name}_{model.name}_{data_type}_confmat_normed",
        normalize=True,
    )
    pu.plot_confusion_matrix(
        os.path.join(out_dir, "images"),
        conf_mat,
        lablist,
        title=f"{experiment_name}_{model.name}_{data_type}_confmat_unnormed",
        normalize=False,
    )

    pu.print_classification_report(trues, class_predictions)

    pu.plot_roc(
        roc_trues,
        class_probs,
        f"{experiment_name}_{model.name}_{data_type}",
        scenarios,
        os.path.join(
            out_dir, "images", f"{experiment_name}_{model.name}_{data_type}_roc.pdf"
        ),
    )

    pu.plot_prec_recall(
        pr_trues,
        class_probs,
        f"{experiment_name}_{model.name}_{data_type}",
        scenarios,
        os.path.join(
            out_dir, "images", f"{experiment_name}_{model.name}_{data_type}_pr.pdf"
        ),
    )


def main(ua):
    yaml_data = read_config(ua.yaml_file)
    work_dir = yaml_data["work dir"]
    os.makedirs(os.path.join(work_dir, "images"), exist_ok=True)

    # Collect all the data
    logger.info(
        "Starting training process, shoulders are included in the training set."
    )

    ids, raw_reps, raw_ts_data, sweep_types, raw_sel_coeffs, raw_offsets = get_data(
        ua.training_data, ua.data_type, yaml_data["win_size"]
    )
    lab_dict = {str_id: int_id for int_id, str_id in enumerate(sweep_types)}

    # Convert to numerical ohe IDs
    num_ids = np.array([lab_dict[lab] for lab in ids])
    raw_ohe_ids = to_categorical(
        np.array([lab_dict[lab] for lab in ids]), len(set(ids))
    )

    if ua.subsample_amount:
        subsample_amount = ua.subsample_amount * len(set(ids))
        # Subsample to test for training size effects
        (
            ts_data,
            _,
            ohe_ids,
            _,
            sel_coeffs,
            _,
            reps,
            _,
            offsets,
            _,
        ) = train_test_split(
            raw_ts_data,
            raw_ohe_ids,
            raw_sel_coeffs,
            raw_reps,
            raw_offsets,
            train_size=subsample_amount,
            stratify=raw_ohe_ids,
        )
    else:
        ts_data = raw_ts_data
        ohe_ids = raw_ohe_ids
        reps = raw_reps
        sel_coeffs = raw_sel_coeffs
        offsets = raw_offsets

    logger.info(f"Data is subsampled to {len(ts_data)}")

    class_weights = dict(
        enumerate(
            compute_class_weight(
                class_weight="balanced", classes=np.unique(num_ids), y=num_ids
            )
        )
    )
    logger.info(f"Class weights: {class_weights}")

    if ua.data_type == "aft":
        # Needs to be in correct dims order for Conv1D layer
        datadim = ts_data.shape[1:]
        logger.info(
            f"{ua.data_type.upper()} TS Data shape (samples, timepoints, alleles): {ts_data.shape}"
        )
    else:
        logger.info(f"TS Data shape (samples, timepoints, haps): {ts_data.shape}")
        datadim = ts_data.shape[1:]
        logger.info(f"{len(ts_data)} samples in dataset.")

    logger.info("Splitting Partitions for Classification Task")
    (
        ts_train_data,
        ts_val_data,
        ts_test_data,
        train_labs,
        val_labs,
        test_labs,
        train_s,
        val_s,
        test_s,
        _train_reps,
        _val_reps,
        test_reps,
        train_offsets,
        val_offsets,
        test_offsets,
    ) = split_partitions(ts_data, ohe_ids, sel_coeffs, reps, offsets)

    # Time-series model training and evaluation
    logger.info("Training time-series model.")

    # Lazy switch for testing
    model_type = "1dcnn"
    if model_type == "1dcnn":
        class_model = models.create_TS_class_model(datadim, len(lab_dict))  # type: ignore
        reg_model = models.create_TS_reg_model(datadim)  # type: ignore
    elif model_type == "transformer":
        class_model = models.create_transformer_class_model(
            input_shape=ts_train_data.shape[1:],
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
            n_class=len(lab_dict),
        )
        reg_model = models.create_transformer_reg_model(
            input_shape=ts_train_data.shape[1:],
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )
    else:
        logger.error("Need a model")
        sys.exit(1)

    run_modes = ["class", "reg"]
    if "class" in run_modes:
        logger.info(f"\nRunning classification model for {ua.data_type}")

        trained_class_model = fit_class_model(
            work_dir,
            class_model,
            ua.data_type,
            ts_train_data,
            train_labs,
            ts_val_data,
            val_labs,
            ua.experiment_name,
        )
        evaluate_class_model(
            trained_class_model,
            ts_test_data,
            test_labs,
            test_reps,
            test_offsets,
            work_dir,
            yaml_data["scenarios"],
            ua.experiment_name,
            ua.data_type,
            lab_dict,
        )

    if "reg" in run_modes:
        for idx, scenario in enumerate(yaml_data["scenarios"][1:], start=1):
            # print(reg_model.summary())

            mode = "minmax"
            if mode == "log":
                mm_scaler = None
                trvals = -np.log10(train_s)
                vvals = -np.log10(val_s)
                tevals = -np.log10(test_s)
            elif mode == "minmax":
                mm_scaler = scale_sel_coeffs(train_s)
                trvals = mm_scaler.transform(train_s)
                vvals = mm_scaler.transform(val_s)
                tevals = mm_scaler.transform(test_s)
            else:
                mm_scaler = None
                trvals = train_s
                vvals = val_s
                tevals = test_s

            logger.info(
                f"{ua.data_type.upper()} Regression {scenario.upper()} TS Data shape (samples, timepoints, alleles/haplotypes): {ts_train_data.shape} {ts_val_data.shape} {ts_test_data.shape}"
            )

            plot = True
            if plot:
                pu.plot_s_vs_freqs(
                    train_s,
                    ts_train_data[:, -1, int(ts_train_data.shape[-1] / 2)]
                    - ts_train_data[:, 0, int(ts_train_data.shape[-1] / 2)],
                    scenario,
                    work_dir,
                    ua.experiment_name,
                    mode,
                )

            trained_reg_model = fit_reg_model(
                work_dir,
                reg_model,
                ua.data_type,
                ts_train_data,
                trvals,
                ts_val_data,
                vvals,
                ua.experiment_name + f"_{scenario}_{mode}",
            )
            evaluate_reg_model(
                trained_reg_model,
                ts_test_data,
                test_labs,
                tevals,
                test_reps,
                test_offsets,
                mm_scaler,
                work_dir,
                yaml_data["scenarios"],
                ua.experiment_name + f"_{scenario}_{mode}",
                ua.data_type,
                lab_dict,
            )

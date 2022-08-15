import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from .plotting import plotting_utils as pu
from .utils.gen_utils import read_config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig()
logger = logging.getLogger("nets")
logger.setLevel("INFO")

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def scale_sel_coeffs(sel_coef_arr):
    """Scale training data, return scaled data and scaler to use for test data."""
    scaler = MinMaxScaler().fit(sel_coef_arr)

    return scaler.transform(sel_coef_arr), scaler


def get_data(input_pickle, data_type):
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
    data_list = []
    sel_coeffs = []
    sweep_types = []
    pikl_dict = pickle.load(open(input_pickle, "rb"))
    for sweep in pikl_dict.keys():
        sweep_types.append(sweep)
        for rep in pikl_dict[sweep].keys():
            id_list.append(sweep)
            data_list.append(np.array(pikl_dict[sweep][rep][data_type.lower()]))
            sel_coeffs.append(pikl_dict[sweep][rep]["sel_coeff"])

    return id_list, np.stack(data_list), sweep_types, np.array(sel_coeffs).reshape(-1, 1)


def split_partitions(data, labs, sel_coeffs):
    """
        Splits all data and labels into partitions for train/val/testing.

    Args:
        data (np.arr): Data for training model.
        labs (List): List of numeric labels for IDs
        sel_coeffs (List[float]): List of selection coefficients for each rep

    Returns:
        Tuple[List[narr], List[narr], List[narr], List[int], List[int], List[int]]: Train/val/test splits of IDs and labs
    """
    train_data, val_data, train_labs, val_labs, train_s, val_s = train_test_split(
        data, labs, sel_coeffs, stratify=labs, test_size=0.3
    )
    val_data, test_data, val_labs, test_labs, val_s, test_s = train_test_split(
        val_data, val_labs, val_s, stratify=val_labs, test_size=0.5
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
    )


# fmt: off
def create_TS_model(datadim, n_class):
    """
    Creates Time-Distributed SHIC model that uses 3 convlutional blocks with concatenation.

    Returns:
        Model: Keras compiled model.
    """
    model_in = Input(datadim)
    h = Conv1D(64, 3, activation="relu", padding="same")(model_in)
    h = Conv1D(64, 3, activation="relu", padding="same")(h)
    h = MaxPooling1D(pool_size=3, padding="same")(h)
    h = Dropout(0.15)(h)
    h = Flatten()(h)

    h = Dense(264, activation="relu")(h)
    h = Dropout(0.2)(h)        
    h = Dense(264, activation="relu")(h)
    h = Dropout(0.2)(h)
    h = Dense(128, activation="relu")(h)
    h = Dropout(0.1)(h)
    reg_output = Dense(1, activation="relu", name="reg_output")(h)
    class_output = Dense(n_class, activation="softmax", name="class_output")(h)

    model = Model(inputs=[model_in], outputs=[class_output, reg_output], name="Timesweeper")
    model.compile(
        loss={"class_output":"categorical_crossentropy", "reg_output":"mae"},
        optimizer="adam",
        metrics={"class_output": "accuracy", "reg_output": "mae"},
    )

    return model

def create_1Samp_model(datadim, n_class):
    """
    Fully connected net for 1Samp prediction.

    Returns:
        Model: Keras compiled model.
    """
    model_in = Input(datadim)
    h = Dense(512, activation="relu")(model_in)
    h = Dropout(0.2)(h)        
    h = Dense(512, activation="relu")(h)
    h = Dropout(0.2)(h)
    h = Dense(128, activation="relu")(h)
    h = Dropout(0.1)(h)
    reg_output = Dense(1, activation="linear", name="reg_output")(h)
    class_output = Dense(n_class, activation="softmax", name="class_output")(h)
    
    model = Model(inputs=[model_in], outputs=[class_output, reg_output], name="Timesweeper1Samp")
    model.compile(
        loss={"class_output":"categorical_crossentropy", "reg_output":"mae"},
        optimizer="adam",
        metrics={"class_output": "accuracy", "reg_output": "mae"},
    )

    return model

# fmt: on


def fit_model(
    out_dir,
    model,
    data_type,
    train_data,
    train_labs,
    train_s,
    val_data,
    val_labs,
    val_s,
    class_weights,
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

    if not os.path.exists(os.path.join(out_dir, "images")):
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    if not os.path.exists(os.path.join(out_dir, "trained_models")):
        os.makedirs(os.path.join(out_dir, "trained_models"), exist_ok=True)

    checkpoint = ModelCheckpoint(
        os.path.join(out_dir, "trained_models", f"{model.name}_{data_type}"),
        monitor="val_class_output_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    earlystop = EarlyStopping(
        monitor="val_class_output_accuracy",
        min_delta=0.1,
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    history = model.fit(
        x=train_data,
        y=[train_labs, train_s],
        epochs=40,
        verbose=2,
        callbacks=callbacks_list,
        validation_data=(val_data, {"class_output": val_labs, "reg_output": val_s}),
        # class_weight=class_weights,
    )

    pu.plot_training(
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


def evaluate_model(
    model,
    test_data,
    test_labs,
    test_s,
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

    class_probs, pred_s = model.predict(test_data)
    class_predictions = np.argmax(class_probs, axis=1)
    trues = np.argmax(test_labs, axis=1)

    # Cannot for the life of me figure out why memory is shared b/t functions and this
    # So it gets casted twice to break that chain
    roc_trues = np.array(list(trues))
    pr_trues = np.array(list(trues))

    pred_s = s_scaler.inverse_transform(pred_s.reshape(-1, 1))
    test_s = s_scaler.inverse_transform(test_s.reshape(-1, 1))

    pred_dict = {
        "true": trues,
        "pred": class_predictions,
        "true_sel_coeff": list(test_s),
        "pred_sel_coeff": list(pred_s),
    }
    for str_lab in lab_dict:
        pred_dict[f"{str_lab}_scores"] = class_probs[:, lab_dict[str_lab]]

    pred_df = pd.DataFrame(pred_dict)

    os.makedirs(os.path.join(out_dir, "test_predictions"), exist_ok=True)
    pred_df.to_csv(
        os.path.join(
            out_dir,
            "test_predictions",
            f"{experiment_name}_{model.name}_{data_type}_test_predictions.csv",
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

    print(
        f"Mean absolute error for Sel Coeff predictions: {mean_absolute_error(test_s, pred_s)}"
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


def main(ua):
    yaml_data = read_config(ua.yaml_file)
    work_dir = yaml_data["work dir"]

    # Collect all the data
    logger.info("Starting training process.")
    if ua.hft:
        type_list = ["aft", "hft"]
    else:
        type_list = ["aft"]
    for data_type in type_list:
        ids, ts_data, sweep_types, sel_coeffs = get_data(ua.training_data, data_type)
        lab_dict = {str_id: int_id for int_id, str_id in enumerate(sweep_types)}

        # Convert to numerical ohe IDs
        num_ids = np.array([lab_dict[lab] for lab in ids])
        ohe_ids = to_categorical(
            np.array([lab_dict[lab] for lab in ids]), len(set(ids))
        )

        class_weights = dict(
            enumerate(
                compute_class_weight(
                    class_weight="balanced", classes=np.unique(num_ids), y=num_ids
                )
            )
        )
        print(f"Class weights: {class_weights}")

        if data_type == "aft":
            # Needs to be in correct dims order for Conv1D layer
            datadim = ts_data.shape[1:]
            logger.info(
                f"TS Data shape (samples, timepoints, alleles): {ts_data.shape}"
            )
        else:
            logger.info(f"TS Data shape (samples, timepoints, haps): {ts_data.shape}")
            datadim = ts_data.shape[1:]
            logger.info(f"{len(ts_data)} samples in dataset.")

        logger.info("Splitting Partition")
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
        ) = split_partitions(ts_data, ohe_ids, sel_coeffs)

        scaled_train_s, mm_scaler = scale_sel_coeffs(train_s)

        # Time-series model training and evaluation
        logger.info("Training time-series model.")
        model = create_TS_model(datadim, len(lab_dict))

        print(model.summary())

        trained_model = fit_model(
            work_dir,
            model,
            data_type,
            ts_train_data,
            train_labs,
            scaled_train_s,
            ts_val_data,
            val_labs,
            mm_scaler.transform(val_s),
            class_weights,
            ua.experiment_name,
        )
        evaluate_model(
            trained_model,
            ts_test_data,
            test_labs,
            mm_scaler.transform(test_s),
            mm_scaler,
            work_dir,
            yaml_data["scenarios"],
            ua.experiment_name,
            data_type,
            lab_dict,
        )

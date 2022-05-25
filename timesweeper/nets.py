import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.utils import to_categorical

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
    sweep_types = []
    pikl_dict = pickle.load(open(input_pickle, "rb"))
    for sweep in pikl_dict.keys():
        sweep_types.append(sweep)
        for rep in pikl_dict[sweep].keys():
            id_list.append(sweep)
            data_list.append(np.array(pikl_dict[sweep][rep][data_type.lower()]))

    return id_list, np.stack(data_list), sweep_types


def split_partitions(data, labs):
    """
        Splits all data and labels into partitions for train/val/testing.

    Args:
        data (np.arr): Data for training model.
        labs (List): List of numeric labels for IDs

    Returns:
        Tuple[List[narr], List[narr], List[narr], List[int], List[int], List[int]]: Train/val/test splits of IDs and labs
    """
    train_data, val_data, train_labs, val_labs = train_test_split(
        data, labs, stratify=labs, test_size=0.3
    )
    val_data, test_data, val_labs, test_labs = train_test_split(
        val_data, val_labs, stratify=val_labs, test_size=0.5
    )
    return (train_data, val_data, test_data, train_labs, val_labs, test_labs)


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
    output = Dense(n_class, activation="softmax")(h)

    model = Model(inputs=[model_in], outputs=[output], name="Timesweeper")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
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
    output = Dense(n_class,  activation="softmax")(h)

    
    model = Model(inputs=[model_in], outputs=[output], name="Timesweeper1Samp")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model

# fmt: on


def fit_model(
    out_dir,
    model,
    data_type,
    train_data,
    train_labs,
    val_data,
    val_labs,
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
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    earlystop = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.1,
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    history = model.fit(
        x=train_data,
        y=train_labs,
        epochs=40,
        verbose=2,
        callbacks=callbacks_list,
        validation_data=(val_data, val_labs),
        class_weight=class_weights,
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
    model, test_data, test_labs, out_dir, experiment_name, data_type, lab_dict
):
    """
    Evaluates model using confusion matrices and plots results.

    Args:
        model (Model): Fit Keras model.
        test_data (List[narr]): Testing data.
        test_labs (narr): Testing labels.
        out_dir (str): Base directory data is located in.
        experiment_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.
        data_type (str): Whether data is aft or hfs.
    """

    pred = model.predict(test_data)
    predictions = np.argmax(pred, axis=1)
    trues = np.argmax(test_labs, axis=1)

    # Cannot for the life of me figure out why memory is shared b/t functions and this
    # So it gets casted twice to break that chain
    roc_trues = np.array(list(trues))
    pr_trues = np.array(list(trues))

    pred_dict = {
        "true": trues,
        "pred": predictions,
    }
    for str_lab in lab_dict:
        pred_dict[f"{str_lab}_scores"] = pred[:, lab_dict[str_lab]]

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

    conf_mat = confusion_matrix(trues, predictions)

    pu.plot_confusion_matrix(
        os.path.join(out_dir, "images"),
        conf_mat,
        lablist,
        title=f"{experiment_name}_{model.name}_{data_type}_confmat",
        normalize=False,
    )

    pu.print_classification_report(trues, predictions)

    pu.plot_roc(
        roc_trues,
        pred,
        f"{experiment_name}_{model.name}_{data_type}",
        os.path.join(
            out_dir, "images", f"{experiment_name}_{model.name}_{data_type}_roc.png"
        ),
    )

    pu.plot_prec_recall(
        pr_trues,
        pred,
        f"{experiment_name}_{model.name}_{data_type}",
        os.path.join(
            out_dir, "images", f"{experiment_name}_{model.name}_{data_type}_pr.png"
        ),
    )


def main(ua):
    if ua.config_format == "yaml":
        yaml_data = read_config(ua.yaml_file)
        work_dir = yaml_data["work dir"]

    elif ua.config_format == "cli":
        work_dir = ua.work_dir

    # Collect all the data
    logger.info("Starting training process.")
    if ua.hft:
        type_list = ["aft", "hft"]
    else:
        type_list = ["aft"]
    for data_type in type_list:
        ids, ts_data, sweep_types = get_data(ua.training_data, data_type)
        lab_dict = {str_id: int_id for int_id, str_id in enumerate(sweep_types)}

        # Convert to numerical one IDs
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
        ) = split_partitions(ts_data, ohe_ids)

        # Time-series model training and evaluation
        logger.info("Training time-series model.")
        model = create_TS_model(datadim, len(lab_dict))

        trained_model = fit_model(
            work_dir,
            model,
            data_type,
            ts_train_data,
            train_labs,
            ts_val_data,
            val_labs,
            class_weights,
            ua.experiment_name,
        )
        evaluate_model(
            trained_model,
            ts_test_data,
            test_labs,
            work_dir,
            ua.experiment_name,
            data_type,
            lab_dict,
        )

        # Single-timepoint model training and evaluation
        logger.info("Training single-point model.")
        # Use only the final timepoint
        sp_train_data = np.squeeze(ts_train_data[:, -1, :])
        sp_val_data = np.squeeze(ts_val_data[:, -1, :])
        sp_test_data = np.squeeze(ts_test_data[:, -1, :])

        logger.info(f"SP Data shape (samples, haps): {sp_train_data.shape}")

        sp_datadim = sp_train_data.shape[-1]
        model = create_1Samp_model(sp_datadim, len(lab_dict))

        trained_model = fit_model(
            work_dir,
            model,
            data_type,
            sp_train_data,
            train_labs,
            sp_val_data,
            val_labs,
            class_weights,
            ua.experiment_name,
        )
        evaluate_model(
            trained_model,
            sp_test_data,
            test_labs,
            work_dir,
            ua.experiment_name,
            data_type,
            lab_dict,
        )

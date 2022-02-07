import argparse
import os
from glob import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

import plotting.plotting_utils as pu

narr = np.ndarray


def get_data(input_npz):
    """
    Loads data from an NPZ zip.

    Args:
        input_npz (str): Path to NPZ.

    Returns:
        np.arr: Array with all data stacked.
    """
    data_npz = np.load(input_npz)
    id_list = []
    data_list = []
    for i in tqdm(range(len(data_npz.files)), desc="Loading data"):
        if data_npz[data_npz.files[i]].shape[0]:
            id_list.append(data_npz.files[i].split("/")[0])
            data_list.append(data_npz[data_npz.files[i]])

    data_arr = np.stack(padded_data)

    return id_list, data_arr


def split_partitions(data, labs):
    """
        Splits all data and labels into partitions for train/val/testing.

    Args:
        IDs (List): List of files used for training model
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
def create_hapsTS_model(datadim):
    """
    Creates Time-Distributed SHIC model that uses 3 convlutional blocks with concatenation.

    Returns:
        Model: Keras compiled model.
    """
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
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
    output = Dense(3, activation="softmax")(h)

    model = Model(inputs=[model_in], outputs=[output], name="TimeSweeper")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model

def create_haps1Samp_model(datadim):
    """
    Fully connected net for 1Samp prediction.

    Returns:
        Model: Keras compiled model.
    """
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():

    model_in = Input(datadim)
    h = Dense(512, name="512dense", activation="relu")(model_in)
    h = Dropout(0.2, name="drop7")(h)        
    h = Dense(512, name="512dense1", activation="relu")(h)
    h = Dropout(0.2, name="drop71")(h)
    h = Dense(128, name="last_dense", activation="relu")(h)
    h = Dropout(0.1, name="drop8")(h)
    output = Dense(3, name="out_dense", activation="softmax")(h)

    
    model = Model(inputs=[model_in], outputs=[output], name="TimeSweeper1Samp")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model

# fmt: on


def fit_model(base_dir, model, train_data, train_labs, val_data, val_labs, schema_name):
    """
    Fits a given model using training/validation data, plots history after done.

    Args:
        base_dir (str): Base directory where data is located, model will be saved here.
        model (Model): Compiled Keras model.
        train_gen: Training generator
        val_gen: Validation Generator
        schema_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.

    Returns:
        Model: Fitted Keras model, ready to be used for accuracy characterization.
    """

    checkpoint = ModelCheckpoint(
        os.path.join(base_dir, "models", model.name),
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
        train_data,
        train_labs,
        epochs=40,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(val_data, val_labs),
    )

    if not os.path.exists(os.path.join(base_dir, "images")):
        os.makedirs(os.path.join(base_dir, "images"))

    pu.plot_training(
        os.path.join(base_dir, "images"), history, f"{schema_name}_{model.name}",
    )

    # Won't checkpoint handle this?
    save_model(model, os.path.join(base_dir, "models", f"{schema_name}_{model.name}"))

    return model


def evaluate_model(model, test_data, test_labs, base_dir, schema_name):
    """
    Evaluates model using confusion matrices and plots results.

    Args:
        model (Model): Fit Keras model.
        X_test (List[narr]): Testing data.
        Y_test (narr): Testing labels.
        base_dir (str): Base directory data is located in.
        schema_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.
    """

    pred = model.predict(test_data)
    predictions = np.argmax(pred, axis=1)
    trues = np.argmax(test_labs, axis=1)

    pred_dict = {
        "true": trues,
        "pred": predictions,
        "prob_neut": pred[:, 0],
        "prob_hard": pred[:, 1],
        "prob_soft": pred[:, 2],
    }

    pred_df = pd.DataFrame(pred_dict)

    pred_df.to_csv(
        os.path.join(base_dir, f"{schema_name}_{model.name}_predictions.csv"),
        header=True,
        index=False,
    )

    lablist = ["Neut", "Hard", "Soft"]

    conf_mat = confusion_matrix(trues, predictions)
    pu.plot_confusion_matrix(
        os.path.join(base_dir, "images"),
        conf_mat,
        lablist,
        title=f"{schema_name}_{model.name}_confmat",
        normalize=True,
    )
    pu.print_classification_report(trues, predictions)


def train_conductor(base_dir: str, input_npz: str, schema_name: str) -> None:
    """
    Runs all functions related to training and evaluating a model.
    Loads data, splits into train/val/test partitions, creates model, fits, then evaluates.

    Args:
        base_dir (str): Base directory containing subirs with npz files in them. Must have any combo of hard/soft/neut.
        num_timesteps (int): Number of samples in a series of simulation timespan.
        schema_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.
    """

    lab_dict = {"neut": 0, "hard": 1, "soft": 2}
    # Collect all the data
    print("Starting training process.")
    print("Base directory:", base_dir)
    print("Input data file:", input_npz)

    ids, ts_data = get_data(input_npz)
    num_ids = to_categorical(np.array([lab_dict[lab] for lab in ids]), len(set(ids)))

    print(f"{len(ts_data)} samples in dataset.")

    ts_datadim = ts_data.shape[1:]
    print("TS Data shape (samples, timepoints, haps):", ts_data.shape)
    print("\n")

    print("Splitting Partition")
    (
        ts_train_data,
        ts_val_data,
        ts_test_data,
        train_labs,
        val_labs,
        test_labs,
    ) = split_partitions(ts_data, num_ids)

    # Time-series model training and evaluation
    print("Training time-series model.")
    model = create_hapsTS_model(ts_datadim)
    print(model.summary())

    trained_model = fit_model(
        base_dir, model, ts_train_data, train_labs, ts_val_data, val_labs, schema_name
    )
    evaluate_model(trained_model, ts_test_data, test_labs, base_dir, schema_name)

    # Single-timepoint model training and evaluation
    print("Training single-point model.")
    # Use only the final timepoint
    sp_train_data = np.squeeze(ts_train_data[:, -1, :])
    sp_val_data = np.squeeze(ts_val_data[:, -1, :])
    sp_test_data = np.squeeze(ts_test_data[:, -1, :])

    print("SP Data shape (samples, haps):", sp_train_data.shape)

    # print(train_labs)
    # print(test_labs)
    sp_datadim = sp_train_data.shape[-1]
    model = create_haps1Samp_model(sp_datadim)
    print(model.summary())

    trained_model = fit_model(
        base_dir, model, sp_train_data, train_labs, sp_val_data, val_labs, schema_name
    )
    evaluate_model(trained_model, sp_test_data, test_labs, base_dir, schema_name)


def parse_ua():
    argparser = argparse.ArgumentParser(
        description="Handler script for neural network training and prediction for TimeSweeper Package.\
            Will train two models: one for the series of timepoints generated using the hfs vectors over a timepoint and one "
    )

    argparser.add_argument(
        "-i",
        "--input_npz",
        metavar="INPUT_DATA_NPZ",
        dest="input_npz",
        type=str,
        help="NPZ file with arrays in a flat structure, naming scheme for each array should be {sweep_type}_{batch}_{rep}.",
    )

    argparser.add_argument(
        "-n",
        "--schema-name",
        metavar="SCHEMA-NAME",
        dest="schema_name",
        type=str,
        required=False,
        help="Identifier for the sampling schema used to generate the data. Optional, but essential in differentiating runs.",
    )

    user_args = argparser.parse_args()

    return user_args


def main() -> None:
    ua = parse_ua()
    base_dir = os.path.dirname(ua.input_npz)
    print("Input NPZ file:", ua.input_npz)
    print("Saving files to:", base_dir)

    train_conductor(base_dir, ua.input_npz, ua.schema_name)


if __name__ == "__main__":
    main()

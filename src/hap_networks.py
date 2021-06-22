import argparse
import os
from glob import glob
from typing import List, Tuple
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    concatenate,
)
from tensorflow.keras.models import Model, Sequential, load_model, save_model
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

import plotting_utils as pu
import streaming_data as sd


def get_training_data(
    base_dir: str,
    sweep_type: str,
) -> List:
    sample_npz = glob(os.path.join(base_dir, "sims", sweep_type, "muts/*/haps/*.npy"))
    id_list = []

    for i in tqdm(sample_npz, desc="Loading npy files..."):
        id_list.append(i)

    print("\n", len(id_list), "samples in data.")

    return id_list


def prep_data(base_dir: str, timeseries: bool) -> None:
    y_list = []
    id_list = []

    if timeseries:
        sweep_lab_dict = {
            "hard": 0,
            "neut": 1,
            "soft": 2,
        }
        idfile = "haps_IDs.csv"

    else:
        sweep_lab_dict = {
            "hard1Samp": 0,
            "neut1Samp": 1,
            "soft1Samp": 2,
        }
        idfile = "haps_1Samp_IDs.csv"

    for sweep, lab in tqdm(sweep_lab_dict.items(), desc="Loading input data..."):
        id_temp = get_training_data(base_dir, sweep)
        y_list.extend([lab] * len(id_temp))
        id_list.extend(id_temp)

    print("Saving npy files...\n")
    with open(os.path.join(base_dir, idfile), "w") as idfile:
        for (i, j) in zip(id_list, y_list):
            idfile.write(i + "\t" + str(j) + "\n")

    print("Data prepped, you can now train a model using GPU.\n")


def split_partitions(
    IDs: List, labs: List
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
        Splits all data and labels into partitions for train/val/testing.

    Args:
        IDs (List): List of files used for training model
        labs (List): List of numeric labels for IDs

    Returns:
        Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]: Train/val/test splits of IDs and labs
    """
    train_IDs, val_IDs, train_labs, val_labs = train_test_split(
        IDs, labs, stratify=labs, test_size=0.3
    )
    val_IDs, test_IDs, val_labs, test_labs = train_test_split(
        IDs, labs, stratify=labs, test_size=0.5
    )
    return (train_IDs, val_IDs, test_IDs, train_labs, val_labs, test_labs)


# fmt: off
def create_hapsTS_model(datadim: Tuple[int, int]) -> Model:
    """
    Creates Time-Distributed SHIC model that uses 3 convlutional blocks with concatenation.

    Returns:
        Model: Keras compiled model.
    """
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():

    model_in = Input(datadim)
    h = Conv1D(64, 3, activation="relu", padding="same", name="conv1_1")(model_in)
    h = Conv1D(64, 3, activation="relu", padding="same", name="conv1_2")(h)
    h = MaxPooling1D(pool_size=3, name="pool1", padding="same")(h)
    h = Dropout(0.15, name="drop1")(h)
    h = Flatten(name="flatten1")(h)

    h = Dense(264, name="512dense", activation="relu")(h)
    h = Dropout(0.2, name="drop7")(h)        
    h = Dense(264, name="512dense1", activation="relu")(h)
    h = Dropout(0.2, name="drop71")(h)
    h = Dense(128, name="last_dense", activation="relu")(h)
    h = Dropout(0.1, name="drop8")(h)
    output = Dense(3, name="out_dense", activation="softmax")(h)

    model = Model(inputs=[model_in], outputs=[output], name="TimeSweeperHaps")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model

def create_haps1Samp_model(datadim: Tuple[int, int]) -> Model:
    """
    Fully connected net for 1Samp prediction.

    Returns:
        Model: Keras compiled model.
    """
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():

    model_in = Input(datadim)
    h = Dense(264, name="512dense", activation="relu")(model_in)
    h = Dropout(0.2, name="drop7")(h)        
    h = Dense(264, name="512dense1", activation="relu")(h)
    h = Dropout(0.2, name="drop71")(h)
    h = Dense(128, name="last_dense", activation="relu")(h)
    h = Dropout(0.1, name="drop8")(h)
    output = Dense(3, name="out_dense", activation="softmax")(h)

    
    model = Model(inputs=[model_in], outputs=[output], name="TimeSweeperHaps1Samp")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model

# fmt: on


def fit_model(
    base_dir: str,
    model: Model,
    train_gen: sd.DataGenerator,
    val_gen: sd.DataGenerator,
) -> Model:
    """
    Fits a given model using training/validation data, plots history after done.

    Args:
        base_dir (str): Base directory where data is located, model will be saved here.
        model (Model): Compiled Keras model.
        train_gen: Training generator
        val_gen: Validation Generator

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
        patience=3,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    history = model.fit(
        train_gen,
        epochs=40,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=val_gen,
    )

    if not os.path.exists(os.path.join(base_dir, "images")):
        os.makedirs(os.path.join(base_dir, "images"))
    pu.plot_training(os.path.join(base_dir, "images"), history, model.name)

    # Won't checkpoint handle this?
    save_model(model, os.path.join(base_dir, "models", model.name))

    return model


def evaluate_model(
    model: Model,
    ID_test: List[str],
    test_gen: sd.DataGenerator,
    base_dir: str,
) -> None:
    """
    Evaluates model using confusion matrices and plots results.

    Args:
        model (Model): Fit Keras model.
        X_test (List[np.ndarray]): Testing data.
        Y_test (np.ndarray): Testing labels.
        base_dir (str): Base directory data is located in.
        time_series (bool): Whether data is time series or one sample per simulation.
    """

    pred = model.predict(test_gen)
    predictions = np.argmax(pred, axis=1)
    trues = test_gen.labels

    # for i, j, k in zip(test_gen.list_IDs, predictions, trues):
    #    print("\t".join([str(i), str(j), str(k)]))

    pred_dict = {
        "id": test_gen.list_IDs,
        "true": trues,
        "pred": predictions,
        "prob_hard": pred[:, 0],
        "prob_neut": pred[:, 1],
        "prob_soft": pred[:, 2],
    }

    pred_df = pd.DataFrame(pred_dict)

    pred_df.to_csv(
        os.path.join(base_dir, model.name + "_predictions.csv"),
        header=True,
        index=False,
    )

    lablist = ["Hard", "Neut", "Soft"]

    conf_mat = pu.print_confusion_matrix(trues, predictions)
    pu.plot_confusion_matrix(
        os.path.join(base_dir, "images"),
        conf_mat,
        lablist,
        title=model.name,
        normalize=True,
    )
    pu.print_classification_report(trues, predictions)


def train_conductor(
    base_dir: str,
    timeseries: bool,
) -> None:
    """
    Runs all functions related to training and evaluating a model.
    Loads data, splits into train/val/test partitions, creates model, fits, then evaluates.

    Args:
        base_dir (str): Base directory containing data.
        num_timesteps (int): Number of samples in a series of simulation timespan.
        time_series (bool): Whether data is time-series or not, if False num_timesteps must be 1.
    """

    print("Loading previously-prepped data...")

    if timeseries:
        idfile = "haps_IDs.csv"
    else:
        idfile = "haps_1Samp_IDs.csv"

    with open(os.path.join(base_dir, idfile), "r") as idfile:
        rawIDs = [i.strip() for i in idfile.readlines()]
        samps = [i.split("\t")[0] for i in rawIDs]
        labs = [int(i.split("\t")[1]) for i in rawIDs]

    datadim = np.load(samps[0]).shape

    clean_samps = []
    clean_labs = []
    for i in tqdm(range(len(samps))):
        _arr = np.load(samps[i])
        if _arr.shape == datadim:
            clean_samps.append(samps[i])
            clean_labs.append(labs[i])

        if i % 1000 == 0:
            gc.collect()

    print("Data shape:", datadim)

    print("Splitting Partition")
    (train_IDs, val_IDs, test_IDs, train_labs, val_labs, test_labs) = split_partitions(
        clean_samps, clean_labs
    )

    print(train_IDs[:5])
    print(train_labs[:5])

    # fmt: off
    train_gen = sd.DataGenerator(train_IDs, train_labs, 24, datadim, n_classes=3, shuffle=True)
    val_gen = sd.DataGenerator(val_IDs, val_labs, 24, datadim, n_classes=3, shuffle=True)
    test_gen = sd.DataGenerator(test_IDs, test_labs, 24, datadim, n_classes=3, shuffle=False)
    # fmt: on

    print("Creating Model")
    if timeseries:
        model = create_hapsTS_model(datadim)
    else:
        model = create_haps1Samp_model(datadim)

    print(model.summary())

    trained_model = fit_model(base_dir, model, train_gen, val_gen)
    evaluate_model(trained_model, test_IDs, test_gen, base_dir)


def get_pred_data(base_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Stripped down version of the data getter for training data. Loads in arrays and labels for data in the directory.
    TODO Swap this for a method that uses the prepped data. Should just require it to be prepped always.

    Args:
        base_dir (str): Base directory where data is located.

    Returns:
        Tuple[np.ndarray, List[str]]: Array containing all data and list of sample identifiers.
    """
    raise NotImplementedError

    # Input shape needs to be ((num_samps (reps)), num_timesteps, 11(x), 15(y))
    sample_dirs = glob(os.path.join(base_dir, "*"))
    meta_arr_list = []
    sample_list = []
    for i in tqdm(sample_dirs, desc="Loading in data..."):
        sample_files = glob(os.path.join(i, "*.fvec"))
        arr_list = []
        for j in sample_files:
            temp_arr = np.loadtxt(j, skiprows=1)
            arr_list.append(format_arr(temp_arr))
            sample_list.append("-".join(j.split("/")[:-2]))
        one_sample = np.stack(arr_list)
        meta_arr_list.append(one_sample)
    sweep_arr = np.stack(meta_arr_list)
    return sweep_arr, sample_list


def write_predictions(
    outfile_name: str,
    pred_probs: np.ndarray,
    predictions: np.ndarray,
    sample_list: List,
) -> None:
    """
    Writes predictions and probabilities to a csv file.

    Args:
        outfile_name (str): Name of file to write predictions to.
        pred_probs (np.ndarray): Probabilities from softmax output in last layer of model.
        predictions (np.ndarray): Prediction labels from argmax of pred_probs.
        sample_list (List): List of sample identifiers.
    """
    raise NotImplementedError

    classDict = {0: "hard", 1: "neutral", 2: "soft"}

    with open(outfile_name, "w") as outputFile:
        for sample, prediction, prob in zip(
            sample_list, [classDict[i] for i in predictions], pred_probs
        ):
            outputFile.write("\t".join(sample, prediction, prob) + "\n")

    print("{} predictions complete".format(len(sample_list) + 1))


def predict_runner(base_dir: str, model_name: str) -> None:
    """
    Loads a model and data and predicts on samples, writes predictions to csv.
    TODO Make this actually useable.

    Args:
        base_dir (str): Base directory where data is located.
        model_name (str): Name of model being used for predictions.
    """
    raise NotImplementedError

    trained_model = load_model(os.path.join(base_dir, model_name + ".model"))
    pred_data, sample_list = get_pred_data(base_dir)

    pred = trained_model.predict(pred_data)
    predictions = np.argmax(pred, axis=1)
    pred_probs = pred[:, predictions]

    write_predictions(
        model_name + "_predictions.csv", pred_probs, predictions, sample_list
    )


def get_ts_from_dir(dirname: str) -> int:
    """
    Splits directory name to get number of timepoints per replicate.

    Args:
        dirname (str): Directory named after slim parameterization, contains timepoints in name.

    Returns:
        int: Number of timepoints being sampled.
    """
    sampstr = dirname.split("-")[2]

    return int(sampstr.split("Samp")[0])


def parse_ua() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        description="Handler script for neural network training and prediction for TimeSweeper Package."
    )

    argparser.add_argument(
        "mode",
        metavar="RUN_MODE",
        choices=["train", "predict", "prep"],
        type=str,
        default="train",
        help="Whether to train a new model or load a pre-existing one located in base_dir.",
    )

    argparser.add_argument(
        "base_dir",
        metavar="DATA_BASE_DIRECTORY",
        type=str,
        help="Directory containing simulation results and preprocessed files.",
    )

    argparser.add_argument(
        "--time-series",
        action="store_true",
        default=False,
        help="Whether data being trained on/prepped is time-series sampled or single-sampled.",
        dest="time_series",
    )

    user_args = argparser.parse_args()

    return user_args


def main() -> None:
    ua = parse_ua()

    print("Saving files to:", ua.base_dir)
    print("Mode:", ua.mode)
    print("Time series:", str(ua.time_series))

    if ua.mode == "train":
        train_conductor(ua.base_dir, ua.time_series)

    elif ua.mode == "prep":
        # This is so you don't have to prep so much data on a GPU job
        # Run this on CPU first, then train the model on the formatted data
        prep_data(ua.base_dir, ua.time_series)


if __name__ == "__main__":
    main()

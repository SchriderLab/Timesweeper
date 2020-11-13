import argparse
import os
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
    BatchNormalization,
)
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import plotting_utils as pu


def get_training_data(base_dir, sweep_type, num_lab, numSubWins=11):
    # Input shape needs to be ((num_samps (reps)), num_timesteps, 11(x), 15(y))
    meta_arr_list = []
    sample_dirs = glob(os.path.join(base_dir, "sims", sweep_type, "cleaned/*"))

    for i in tqdm(sample_dirs[:20], desc="Loading in data..."):
        sample_files = glob(os.path.join(i, "*.fvec"))
        arr_list = []
        for j in sample_files:
            temp_arr = np.loadtxt(j, skiprows=1)
            arr_list.append(format_arr(temp_arr, numSubWins))
        one_sample = np.stack(arr_list)
        print(one_sample.shape)

        meta_arr_list.append(one_sample)

    print(meta_arr_list)

    sweep_arr = np.stack(meta_arr_list)
    sweep_labs = np.repeat(num_lab, sweep_arr.shape[0])

    return sweep_arr, sweep_labs


def format_arr(sweep_array, numSubWins):
    """Splits fvec into 2D array that is (windows, features) large.

    Args:
        sweep_array (ndarray): 1D np array output by diploSHIC

    Returns:
        2D nparray: 2D representation of SHIC fvec, x axis is windows, y axis is features
    """
    vector = np.array_split(sweep_array, 15)
    stacked = np.vstack(vector)
    stacked = np.reshape(stacked, (numSubWins, 15))
    return stacked


def split_partitions(X, Y):
    (X_train, X_valid, Y_train, Y_valid) = train_test_split(X, Y, test_size=0.3)
    (X_valid, X_test, Y_valid, Y_test) = train_test_split(
        X_valid, Y_valid, test_size=0.5
    )

    return (
        X_train,
        X_valid,
        X_test,
        to_categorical(Y_train),
        to_categorical(Y_valid),
        to_categorical(Y_test),
    )


def create_rcnn_model():
    # https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

    # Build CNN
    rcnn = Sequential(name="TimeSweeperModel")
    rcnn.add(TimeDistributed(Conv1D(128, 3, input_shape=(None, 11, 15))))
    rcnn.add(
        TimeDistributed(
            Conv1D(128, 3, activation="relu", padding="same", name="conv1_1")
        )
    )
    rcnn.add(
        TimeDistributed(
            Conv1D(64, 3, activation="relu", padding="same", name="conv1_2")
        )
    )
    rcnn.add(TimeDistributed(MaxPooling2D(pool_size=3, name="pool1", padding="same")))
    rcnn.add(TimeDistributed(Dropout(0.15, name="drop1")))
    rcnn.add(TimeDistributed(Flatten(name="flatten1")))

    # LSTM Model
    rcnn.add(LSTM(64, return_sequences=False, input_shape=(None, None)))

    # Dense
    # dense.add(Dense(512, name="512dense", activation="relu"))
    # dense.add(Dropout(0.2, name="drop7"))
    # dense.add(Dense(128, name="last_dense", activation="relu"))
    # dense.add(Dropout(0.1, name="drop8"))
    rcnn.add(Dense(3, activation="softmax"))

    rcnn.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return rcnn


def create_cnn3d_model():
    # https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

    # CNN
    cnn3d = Sequential(name="TimeSweeper3D")
    cnn3d.add(Conv2D(128, 3, input_shape=(11, 15, 10)))
    cnn3d.add(MaxPooling2D(pool_size=3, padding="same"))
    cnn3d.add(BatchNormalization())

    cnn3d.add(Conv2D(128, 3, activation="relu", padding="same", name="conv1_1"))
    cnn3d.add(MaxPooling2D(pool_size=3, padding="same"))
    cnn3d.add(BatchNormalization())

    cnn3d.add(Conv2D(64, 3, activation="relu", padding="same", name="conv1_2"))
    cnn3d.add(MaxPooling2D(pool_size=3, padding="same"))
    cnn3d.add(BatchNormalization())

    cnn3d.add(Flatten())

    # Dense
    cnn3d.add(Dense(512, name="512dense", activation="relu"))
    cnn3d.add(Dropout(0.2, name="drop7"))
    cnn3d.add(Dense(128, name="last_dense", activation="relu"))
    cnn3d.add(Dropout(0.1, name="drop8"))
    cnn3d.add(Dense(3, activation="softmax"))

    cnn3d.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return cnn3d


def fit_model(base_dir, model, X_train, X_valid, X_test, Y_train, Y_valid):

    # print(X_train.shape)
    print("Training set has {} examples".format(len(X_train)))
    print("Validation set has {} examples".format(len(X_valid)))
    print("Test set has {} examples".format(len(X_test)))

    checkpoint = ModelCheckpoint(
        model.name + ".model",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    earlystop = EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    history = model.fit(
        x=X_train,
        y=Y_train,
        batch_size=32,
        steps_per_epoch=len(X_train) / 32,
        epochs=40,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(X_valid, Y_valid),
        validation_steps=len(X_valid) / 32,
    )

    pu.plot_training(".", history, "TimeSweeper3D")

    # Won't checkpoint handle this?
    save_model(model, os.path.join(base_dir, model.name + ".model"))

    return model


def evaluate_model(model, X_test, Y_test, base_dir):
    pred = model.predict(X_test)
    predictions = np.argmax(pred, axis=1)

    trues = np.argmax(Y_test, axis=1)

    conf_mat = pu.print_confusion_matrix(trues, predictions)
    pu.plot_confusion_matrix(base_dir, conf_mat, ["Hard", "Neut", "Soft"])
    pu.print_classification_report(trues, predictions)


def train_conductor(base_dir, time_series):

    if time_series:
        sweep_lab_dict = {
            "hard": 0,
            "neut": 1,
            "soft": 2,
        }
    else:
        sweep_lab_dict = {
            "hard1Samp": 0,
            "soft1Samp": 1,
            "neut1Samp": 2,
        }

    X_list = []
    y_list = []
    for sweep, lab in tqdm(sweep_lab_dict.items(), desc="Loading input data..."):
        X_temp, y_temp = get_training_data(base_dir, sweep, lab)
        X_list.extend(X_temp)
        y_list.extend(y_temp)

    X = np.asarray(X_list)  # np.stack(X_list, 0)
    y = y_list

    print(X[0].shape)

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_partitions(X, y)

    model = create_cnn3d_model()
    trained_model = fit_model(
        base_dir, model, X_train, X_valid, X_test, Y_train, Y_valid
    )
    evaluate_model(trained_model, X_test, Y_test, base_dir)


def get_pred_data(base_dir, numSubWins=11):
    # Input shape needs to be ((num_samps (reps)), num_timesteps, 11(x), 15(y))
    sample_dirs = glob(os.path.join(base_dir, "*"))
    meta_arr_list = []
    sample_list = []
    for i in tqdm(sample_dirs[:20], desc="Loading in data..."):
        sample_files = glob(os.path.join(i, "*.fvec"))
        arr_list = []
        for j in sample_files:
            temp_arr = np.loadtxt(j, skiprows=1)
            arr_list.append(format_arr(temp_arr, numSubWins))
            sample_list.append("-".join(j.split("/")[:-2]))
        one_sample = np.stack(arr_list)
        meta_arr_list.append(one_sample)
    sweep_arr = np.stack(meta_arr_list)
    return sweep_arr, sample_list


def write_predictions(outfile_name, pred_probs, predictions, sample_list):
    classDict = {0: "hard", 1: "neutral", 2: "soft"}

    with open(outfile_name, "w") as outputFile:
        for sample, prediction, prob in zip(
            sample_list, [classDict[i] for i in predictions], pred_probs
        ):
            outputFile.write("\t".join(sample, prediction, prob) + "\n")

    print("{} predictions complete".format(len(sample_list) + 1))


def predict_runner(base_dir, model_name="TimeSweeperCNN", numSubWins=11):

    trained_model = load_model(os.path.join(base_dir, model_name + ".model"))
    pred_data, sample_list = get_pred_data(base_dir, numSubWins, "pred")

    pred = trained_model.predict(pred_data)
    predictions = np.argmax(pred, axis=1)
    pred_probs = pred[:, predictions]

    write_predictions(
        model_name + "_predictions.csv", pred_probs, predictions, sample_list
    )


def parse_ua():
    argparser = argparse.ArgumentParser(
        description="Handler script for neural network training and prediction for TimeSweeper Package."
    )

    argparser.add_argument(
        "mode",
        metavar="RUN_MODE",
        choices=["train", "predict"],
        type=str,
        help="Whether to train a new model or load a pre-existing one located in base_dir.",
    )

    argparser.add_argument(
        "base_dir",
        metavar="DATA_BASE_DIRECTORY",
        type=str,
        default="/proj/dschridelab/timeSeriesSweeps/onePop-selectiveSweep-10Samp-20Int",
        help="Directory containing subdirectory structure of base_dir/samples/timestep.fvec.",
    )  # for testing
    user_args = argparser.parse_args()

    return user_args


def main():
    ua = parse_ua()

    print(ua.base_dir)
    print(ua.mode)
    if ua.mode == "train":
        print("hello")
        train_conductor(ua.base_dir, time_series=True)


if __name__ == "__main__":
    main()
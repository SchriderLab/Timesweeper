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
    Input,
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


def get_training_data(base_dir, sweep_type, num_lab):
    # Input shape needs to be ((num_samps (reps)), num_timesteps, 11(x), 15(y))
    sample_dirs = glob(os.path.join(base_dir, "sims", sweep_type, "cleaned/*"))

    """
    File structure:
    - cleaned
        - 0 (samps)
            - rep_#_point_#

    """
    samp_list = []
    lab_list = []
    for samp_dir in tqdm(sample_dirs, desc="Loading in {} data...".format(sweep_type)):
        # cleaned
        for rep in range(1, 101):
            # rep_#_point_*.fvec
            sample_files = glob(
                os.path.join(samp_dir, "rep_{}_point_*.fvec".format(rep))
            )

            rep_list = []
            for point_file in sample_files:
                try:
                    temp_arr = np.loadtxt(point_file, skiprows=1)
                    rep_list.append(format_arr(temp_arr))
                except ValueError:
                    print("! {} couldn't be read!".format(point_file))
                    continue

            try:
                one_rep = np.stack(rep_list).astype(np.float32)
                if one_rep.shape[0] == 10:
                    # samp_list.append(one_rep.reshape(11, 15, one_rep.shape[0])) #Use this if non-TD 2DCNN model

                    samp_list.append(
                        one_rep.reshape(10, 11, 15, 1)
                    )  # Use this if TD 2DCNN model
                    lab_list.append(num_lab)
                else:
                    continue
            except ValueError:
                print("! No replicates found in rep {}".format(rep))
                continue

    sweep_arr = samp_list
    sweep_labs = lab_list

    print("\n", len(samp_list), "\n")

    return sweep_arr, sweep_labs


def format_arr(sweep_array):
    """Splits fvec into 2D array that is (windows, features) large.

    Args:
        sweep_array (ndarray): 1D np array output by diploSHIC

    Returns:
        2D nparray: 2D representation of SHIC fvec, x axis is windows, y axis is features
    """
    # Split into vectors of windows for each feature
    vector = np.array_split(sweep_array, 15)
    # Stack sow that shape is (feats,wins)
    stacked = np.vstack(vector)
    # Transpose so shape is (wins,feats)
    stacked = stacked.T

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

    # Build CNN.
    input_layer = Input((10, 11, 15, 1))
    conv_block_1 = TimeDistributed(Conv2D(32, (3, 3), activation="relu"))(input_layer)
    conv_block_1 = TimeDistributed(MaxPooling2D(3, padding="same"))(conv_block_1)
    conv_block_1 = TimeDistributed(BatchNormalization())(conv_block_1)

    conv_block_2 = TimeDistributed(Conv2D(64, (3, 3), activation="relu"))(conv_block_1)
    conv_block_2 = TimeDistributed(MaxPooling2D(3, padding="same"))(conv_block_2)
    conv_block_2 = TimeDistributed(BatchNormalization())(conv_block_2)

    conv_block_3 = TimeDistributed(Conv2D(128, (3, 3), activation="relu"))(conv_block_1)
    conv_block_3 = TimeDistributed(MaxPooling2D(3, padding="same"))(conv_block_2)
    conv_block_3 = TimeDistributed(BatchNormalization())(conv_block_2)

    flat = TimeDistributed(Flatten())(conv_block_3)

    # LSTM Model
    rnn = LSTM(64, return_sequences=False)(flat)

    # Dense
    dense_block = Dense(128, activation="relu")(rnn)
    dense_block = Dropout(0.2)(dense_block)
    dense_block = Dense(3, activation="softmax")(dense_block)

    rcnn = Model(inputs=input_layer, outputs=dense_block, name="TimeSweeperRCNN")

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


def create_shic_model():

    model_in = Input((11, 15, 10))
    h = Conv2D(128, 3, activation="relu", padding="same", name="conv1_1")(model_in)
    h = Conv2D(64, 3, activation="relu", padding="same", name="conv1_2")(h)
    h = MaxPooling2D(pool_size=3, name="pool1", padding="same")(h)
    h = Dropout(0.15, name="drop1")(h)
    h = Flatten(name="flaten1")(h)

    dh = Conv2D(
        128, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_1"
    )(model_in)
    dh = Conv2D(
        64, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_2"
    )(dh)
    dh = MaxPooling2D(pool_size=2, name="dpool1")(dh)
    dh = Dropout(0.15, name="ddrop1")(dh)
    dh = Flatten(name="dflaten1")(dh)

    dh1 = Conv2D(
        128, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_1"
    )(model_in)
    dh1 = Conv2D(
        64, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_2"
    )(dh1)
    dh1 = MaxPooling2D(pool_size=2, name="d1pool1")(dh1)
    dh1 = Dropout(0.15, name="d1drop1")(dh1)
    dh1 = Flatten(name="d1flaten1")(dh1)

    h = concatenate([h, dh, dh1])
    h = Dense(512, name="512dense", activation="relu")(h)
    h = Dropout(0.2, name="drop7")(h)
    h = Dense(128, name="last_dense", activation="relu")(h)
    h = Dropout(0.1, name="drop8")(h)
    output = Dense(3, name="out_dense", activation="softmax")(h)

    model = Model(inputs=[model_in], outputs=[output], name="TimeSweeperSHIC")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


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
        epochs=100,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(X_valid, Y_valid),
        validation_steps=len(X_valid) / 32,
    )

    pu.plot_training(".", history, model.name)

    # Won't checkpoint handle this?
    save_model(model, os.path.join(base_dir, model.name + ".model"))

    return model


def evaluate_model(model, X_test, Y_test, base_dir):
    pred = model.predict(X_test)
    predictions = np.argmax(pred, axis=1)

    trues = np.argmax(Y_test, axis=1)

    conf_mat = pu.print_confusion_matrix(trues, predictions)
    pu.plot_confusion_matrix(
        base_dir, conf_mat, ["Hard", "Neut", "Soft"], title=model.name
    )
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


def get_pred_data(base_dir):
    # Input shape needs to be ((num_samps (reps)), num_timesteps, 11(x), 15(y))
    sample_dirs = glob(os.path.join(base_dir, "*"))
    meta_arr_list = []
    sample_list = []
    for i in tqdm(sample_dirs[:20], desc="Loading in data..."):
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
    pred_data, sample_list = get_pred_data(base_dir)

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
        train_conductor(ua.base_dir, time_series=True)


if __name__ == "__main__":
    main()

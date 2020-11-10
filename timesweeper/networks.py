import numpy as np
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
    TimeDistributed,
    LSTM,
)
from tensorflow.keras.models import Model, save_model, load_model
import os
from glob import glob
from tqdm import tqdm
import argparse


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
    Y_train = np.to_categorical(Y, 3)
    (X_train, X_valid, Y_train, Y_valid) = train_test_split(X, Y, test_size=0.3)
    (X_valid, X_test, Y_valid, Y_test) = train_test_split(
        X_valid, Y_valid, test_size=0.5
    )

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def create_model(X_train):
    # https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

    model_in = Input(X_train.shape[1:])
    h = Conv1D(128, 3, activation="relu", padding="same", name="conv1_1")(model_in)
    h = Conv1D(64, 3, activation="relu", padding="same", name="conv1_2")(h)
    h = MaxPooling1D(pool_size=3, name="pool1", padding="same")(h)
    h = Dropout(0.15, name="drop1")(h)
    h = Flatten(name="flaten1")(h)

    dh = Conv1D(
        128, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_1"
    )(model_in)
    dh = Conv1D(
        64, 2, activation="relu", dilation_rate=[1, 3], padding="same", name="dconv1_2"
    )(dh)

    dh = MaxPooling1D(pool_size=2, name="dpool1")(dh)
    dh = Dropout(0.15, name="ddrop1")(dh)
    dh = Flatten(name="dflaten1")(dh)

    dh1 = Conv1D(
        128, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_1"
    )(model_in)
    dh1 = Conv1D(
        64, 2, activation="relu", dilation_rate=[1, 4], padding="same", name="dconv4_2"
    )(dh1)

    dh1 = MaxPooling1D(pool_size=2, name="d1pool1")(dh1)
    dh1 = Dropout(0.15, name="d1drop1")(dh1)
    dh1 = Flatten(name="d1flaten1")(dh1)

    h = concatenate([h, dh, dh1])
    h = Dense(512, name="512dense", activation="relu")(h)
    h = Dropout(0.2, name="drop7")(h)
    h = Dense(128, name="last_dense", activation="relu")(h)
    h = Dropout(0.1, name="drop8")(h)

    t = TimeDistributed(h)
    t = LSTM(t)

    output = Dense(3, name="out_dense", activation="softmax")(t)
    model = Model(inputs=[model_in], outputs=[output], name="TimeSweeperCNN")

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def fit_model(base_dir, model, X_train, X_valid, X_test, Y_train, Y_valid):

    # print(X_train.shape)
    print("training set has %d examples" % X_train.shape[0])
    print("validation set has %d examples" % X_valid.shape[0])
    print("test set has %d examples" % X_test.shape[0])

    checkpoint = ModelCheckpoint(
        model.name + ".model",
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    earlystop = EarlyStopping(
        monitor="val_acc",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    callbacks_list = [earlystop, checkpoint]

    model.fit(
        x=X_train,
        y=Y_train,
        batch_size=32,
        steps_per_epoch=len(X_train) / 32,
        epochs=40,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(X_valid, Y_valid),
        validation_steps=len(X_test) / 32,
    )

    # Won't checkpoint handle this?
    save_model(os.path.join(base_dir, model.name + ".model"))

    return model


def evaluate_model(model, X_test, Y_test):

    score = model.evaluate(len(Y_test) / 32, X_test, Y_test, batch_size=32)

    print("Evaluation on test set:")
    print("TimeSweeper loss: %f" % score[0])
    print("TimeSweeper accuracy: %f" % score[1])


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
    for lab, sweep in sweep_lab_dict.items():
        X_temp, y_temp = get_training_data(base_dir, sweep, lab)
        X_list.append(X_temp)
        y_list.append(y_temp)

        print(X_temp.shape)
        print(y_temp.shape)

    X = np.stack(X_list, 0)
    y = np.concatenate(y_list)

    print(X.shape)
    print(y.shape)

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_partitions(X, y)

    model = create_model(X_train)
    trained_model = fit_model(
        base_dir, model, X_train, X_valid, X_test, Y_train, Y_valid
    )
    evaluate_model(trained_model, X_test, Y_test)


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
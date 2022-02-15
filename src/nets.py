import argparse
import os
from glob import glob
import multiprocessing as mp

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.utils import to_categorical

import plotting.plotting_utils as pu


def get_sweep(filepath):
    """Grabs the sweep label from filepaths for easy saving."""
    for sweep in ["neut", "hard", "soft"]:
        if sweep in filepath:
            return sweep


def loader(filename):
    """Quick loader for multiprocessing."""
    sweep = get_sweep(filename)
    raw_npy = np.load(filename)

    return sweep, raw_npy


def get_data(input_dir, output_dir, data_type):
    """
    Loads data from an base directory and grabs labels from filepath.
    Saves npz file of data with structure label[npy data].
    
    Args:
        input_dir (str): Path of base directory to load files from.
        output_dir (str): Path to write output to.
        data_type (str): Determines either hfs or afs files to search for.

    Returns:
        list[str]: List of sweep labels for each sample
        np.arr: Array with all data stacked.
    """
    id_list = []
    data_list = []
    data_dict = {}
    print("Loading data...")
    all_files = glob(f"{input_dir}/**/{data_type.lower()}_center.npy", recursive=True)

    pool = mp.Pool(mp.cpu_count())
    loader_results = pool.map(loader, all_files)

    for sweep, data in loader_results:
        id_list.append(sweep)
        data_list.append(data)

        if not sweep in data_dict.keys():
            data_dict[sweep] = [data]
        else:
            data_dict[sweep].append(data)

    data_arr = np.stack(data_list)

    np.savez(f"{output_dir}/{data_type}.npz", **data_dict)

    return id_list, data_arr


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
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model_in = Input(datadim)
        h = Dense(512, activation="relu")(model_in)
        h = Dropout(0.2)(h)        
        h = Dense(512, activation="relu")(h)
        h = Dropout(0.2)(h)
        h = Dense(128, activation="relu")(h)
        h = Dropout(0.1)(h)
        output = Dense(3,  activation="softmax")(h)

        
        model = Model(inputs=[model_in], outputs=[output], name="TimeSweeper1Samp")
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

    return model

# fmt: on


def fit_model(
    out_dir, model, data_type, train_data, train_labs, val_data, val_labs, schema_name
):
    """
    Fits a given model using training/validation data, plots history after done.

    Args:
        out_dir (str): Base directory where data is located, model will be saved here.
        model (Model): Compiled Keras model.
        data_type (str): Whether data is HFS or AFS data.
        train_data (np.arr): Training data.
        train_labs (list[int]): OHE labels for training set.
        val_data (np.arr): Validation data.
        val_labs (list[int]): OHE labels for validation set.
        schema_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.

    Returns:
        Model: Fitted Keras model, ready to be used for accuracy characterization.
    """

    if not os.path.exists(os.path.join(out_dir, "images")):
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    if not os.path.exists(os.path.join(out_dir, "models")):
        os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    checkpoint = ModelCheckpoint(
        os.path.join(out_dir, "models", f"{model.name}_{data_type}"),
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

    pu.plot_training(
        os.path.join(out_dir, "images"),
        history,
        f"{schema_name}_{model.name}_{data_type}",
    )

    # Won't checkpoint handle this?
    save_model(
        model,
        os.path.join(out_dir, "models", f"{schema_name}_{model.name}_{data_type}"),
    )

    return model


def evaluate_model(model, test_data, test_labs, out_dir, schema_name, data_type):
    """
    Evaluates model using confusion matrices and plots results.

    Args:
        model (Model): Fit Keras model.
        test_data (List[narr]): Testing data.
        test_labs (narr): Testing labels.
        out_dir (str): Base directory data is located in.
        schema_name (str): Descriptor of the sampling strategy used to generate the data. Used to ID the output.
        data_type (str): Whether data is afs or hfs.
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
        os.path.join(
            out_dir, f"{schema_name}_{model.name}_{data_type}_val_predictions.csv"
        ),
        header=True,
        index=False,
    )

    lablist = ["Neut", "Hard", "Soft"]

    conf_mat = confusion_matrix(trues, predictions)
    pu.plot_confusion_matrix(
        os.path.join(out_dir, "images"),
        conf_mat,
        lablist,
        title=f"{schema_name}_{model.name}_{data_type}_confmat",
        normalize=False,
    )
    pu.print_classification_report(trues, predictions)
    pu.plot_roc(
        trues,
        predictions,
        f"{schema_name}_{model.name}_{data_type}",
        os.path.join(
            out_dir, "images", f"{schema_name}_{model.name}_{data_type}_roc.png"
        ),
    )


def parse_ua():
    argparser = argparse.ArgumentParser(
        description="Handler script for neural network training and prediction for TimeSweeper Package.\
            Will train two models: one for the series of timepoints generated using the hfs vectors over a timepoint and one "
    )

    argparser.add_argument(
        "-i",
        "--input-dir",
        metavar="INPUT_DATADIR",
        dest="input_dir",
        type=str,
        help="Directory with hard/soft/neut subdirs containing <afs/hfs>_center.npy for each replicate.",
    )

    argparser.add_argument(
        "-o",
        "--output-dir",
        metavar="OUTPUT_DATADIR",
        dest="output_dir",
        type=str,
        help="Directory to write trained model files and data npz files to.",
    )

    argparser.add_argument(
        "-n",
        "--schema-name",
        metavar="SCHEMA-NAME",
        dest="schema_name",
        type=str,
        required=False,
        help="Identifier for the sampling schema used to generate the data. Optional, but helpful in differentiating runs.",
    )

    argparser.add_argument(
        "-t",
        "--data-type",
        metavar="DATA MODEL",
        dest="data_type",
        type=str,
        required=True,
        choices=["AFS", "HFS"],
        help="Either AFS or HFS, whether to train on AFS or HFS network data.",
    )

    user_args = argparser.parse_args()

    return user_args


def main():
    ua = parse_ua()
    print("Input dir:", ua.input_dir)
    print("Saving files to:", ua.output_dir)
    os.makedirs(ua.output_dir, exist_ok=True)
    print("Data type:", ua.data_type)

    lab_dict = {"neut": 0, "hard": 1, "soft": 2}

    # Collect all the data
    print("Starting training process.")

    ids, ts_data = get_data(ua.input_dir, ua.output_dir, ua.data_type)

    # Convert to numerical one hot encoded IDs
    num_ids = to_categorical(np.array([lab_dict[lab] for lab in ids]), len(set(ids)))

    if ua.data_type == "AFS":
        # Needs to be in correct dims order for Conv1D layer
        ts_data = np.swapaxes(ts_data, 1, 2)

    print(f"{len(ts_data)} samples in dataset.")

    datadim = ts_data.shape[1:]
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
    model = create_hapsTS_model(datadim)
    print(model.summary())

    trained_model = fit_model(
        ua.output_dir,
        model,
        ua.data_type,
        ts_train_data,
        train_labs,
        ts_val_data,
        val_labs,
        ua.schema_name,
    )
    evaluate_model(
        trained_model,
        ts_test_data,
        test_labs,
        ua.output_dir,
        ua.schema_name,
        ua.data_type,
    )

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
        ua.output_dir,
        model,
        ua.data_type,
        sp_train_data,
        train_labs,
        sp_val_data,
        val_labs,
        ua.schema_name,
    )
    evaluate_model(
        trained_model,
        sp_test_data,
        test_labs,
        ua.output_dir,
        ua.schema_name,
        ua.data_type,
    )


if __name__ == "__main__":
    main()

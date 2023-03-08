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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from timesweeper.plotting import plotting_utils as pu
from timesweeper.utils.gen_utils import read_config

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig()
logger = logging.getLogger("nets")
logger.setLevel("INFO")

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



grid = ParameterGrid([{'kernel_size': [16, 32, 64, 128, 256]},
                      {'kernel': ['rbf'], 'gamma': [1, 10]}]

list(ParameterGrid(grid)) == [{'kernel': 'linear'},
                              {'kernel': 'rbf', 'gamma': 1},
                              {'kernel': 'rbf', 'gamma': 10}]



def get_mlp_model(datadim, n_class,
                  kernel_sizeLayerer_a=64,
                  kernel_sizeLayerer_b=64,
	              dropoutCovLayer_a=0.15,
                  learnRate=0.01,
                  add_one_layer=False,
                  add_two_layer=False,
                  dropoutFullLayer_a=0.2,
                  dropoutFullLayer_b=0.2,
                  dropoutFullLayer_c=0.1):
    """
        Create a CNN model based on the hypter parameter input for the whole sample

    Args:
          kernel_sizeLayerer_a (int): the kernel size of the first convolutiona layer in the first convlutional block.
          kernel_sizeLayerer_b (int): the kernel size of the second convolutiona layer in the first convlutional block.
          dropoutCovLayer_a (float): the dropout rate in the first convlutional block.
          learnRate (float): the learning rate of the CNN.
          add_one_layer (boolean): if true, add one additional convlutional block.
          add_two_layer (boolean): if true, add a second additional convlutional block.
          dropoutFullLayer_a (float): the dropout rate of the first layer in the fully connnected block.
          dropoutFullLayer_b (float): the dropout rate of the second layer in the fully connnected block.
          dropoutFullLayer_c (float): the dropout rate of the third layer in the fully connnected block.

    Returns:
        kera.classifier: a cnn fitted to the input hyperparameter.
    """
    ## Convolution Layer One
    model_in = Input(datadim)
    h = Conv1D(kernel_sizeLayerer_a, 3, activation="relu", padding="same")(model_in)
    h = Conv1D(kernel_sizeLayerer_b, 3, activation="relu", padding="same")(h)
    h = MaxPooling1D(pool_size=3, padding="same")(h)
    h = Dropout(dropoutCovLayer_a)(h)
    h = Flatten()(h)

    ## Convolution Layer Two if add_one_layer
    if add_one_layer:
        h = Conv1D(kernel_sizeLayerer_a, 3, activation="relu", padding="same")(h)
        h = Conv1D(kernel_sizeLayerer_b, 3, activation="relu", padding="same")(h)
        h = MaxPooling1D(pool_size=3, padding="same")(h)
        h = Dropout(dropoutCovLayer_a)(h)
        h = Flatten()(h)

    ## Convolution Layer Three if add_two_layer
    if add_two_layer:
        h = Conv1D(kernel_sizeLayerer_a, 3, activation="relu", padding="same")(h)
        h = Conv1D(kernel_sizeLayerer_b, 3, activation="relu", padding="same")(h)
        h = MaxPooling1D(pool_size=3, padding="same")(h)
        h = Dropout(dropoutCovLayer_a)(h)
        h = Flatten()(h)

    ## Fully Connected Layer
    h = Dense(264, activation="relu")(h)
    h = Dropout(dropoutFullLayer_a)(h)
    h = Dense(264, activation="relu")(h)
    h = Dropout(dropoutFullLayer_b)(h)
    h = Dense(128, activation="relu")(h)
    h = Dropout(dropoutFullLayer_c)(h)

    reg_output = Dense(1, activation="relu", name="reg_output")(h)
    class_output = Dense(n_class, activation="softmax", name="class_output")(h)

    model = Model(inputs=[model_in], outputs=[class_output, reg_output], name="Timesweeper")
    model.compile(
        loss={"class_output":"categorical_crossentropy", "reg_output":"mse"},
        optimizer="adam",
        metrics={"class_output": "accuracy", "reg_output": "mse"},
    )

    return model



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

        # Time-series model training and evaluation
        logger.info("Training time-series model.")


        # scale data to the range of [0, 1]
        trainData = trainData.astype("float32") / 255.0
        testData = testData.astype("float32") / 255.0

        # wrap our model into a scikit-learn compatible classifier
        print("[INFO] initializing model...")

        # define a grid of the hyperparameter search space
        kernel_sizeLayerer_a = [16, 32, 64, 128, 256]
        kernel_sizeLayerer_b = [16, 32, 64, 128, 256]
        dropoutCovLayer_a= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # learnRate = [1e-2, 1e-3, 1e-4]
        add_one_layer= [False, True]
        # add_two_layer= [False, True]
        dropoutFullLayer_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        dropoutFullLayer_b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        dropoutFullLayer_c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        # batchSize = [4, 8, 16, 32]
        # epochs = [10, 20, 30, 40]


        # create a dictionary from the hyperparameter grid
        grid = dict(
        	kernel_sizeLayerer_a=kernel_sizeLayerer_a,
        	kernel_sizeLayerer_b=kernel_sizeLayerer_b,
        	dropoutCovLayer_a=dropoutCovLayer_a,
        	add_one_layer=add_one_layer,
        	dropoutFullLayer_a=dropoutFullLayer_a,
            dropoutFullLayer_b=dropoutFullLayer_b,
        	dropoutFullLayer_c=dropoutFullLayer_c
        )

        param_grid = ParameterGrid(grid)

        for params in param_grid:
            cur_model = get_mlp_model(**params)

            # print(model.summary())
            plot_model(
                cur_model, to_file="model_plot.png", show_shapes=True, show_layer_names=True
            )
            cur_trained_model = fit_model(
                work_dir,
                cur_model,
                data_type,
                ts_train_data,
                train_labs,
                train_s,
                ts_val_data,
                val_labs,
                val_s,
                class_weights,
                ua.experiment_name,
            )
            evaluate_model( ### need to modify to store fitted results
                cur_trained_model,
                ts_test_data,
                test_labs,
                test_s,
                work_dir,
                ua.experiment_name,
                data_type,
                lab_dict,
            )


        logger.info(f"SP Data shape (samples, haps): {sp_train_data.shape}")
        # Use only the final timepoint
        sp_train_data = np.squeeze(ts_train_data[:, -1, :])
        sp_val_data = np.squeeze(ts_val_data[:, -1, :])
        sp_test_data = np.squeeze(ts_test_data[:, -1, :])
        logger.info(f"SP Data shape (samples, haps): {sp_train_data.shape}")

        sp_datadim = sp_train_data.shape[-1]

        for params in param_grid:
            cur_model = model.__init__(self, **params)

            sp_datadim = sp_train_data.shape[-1]
            model = create_1Samp_model(sp_datadim, len(lab_dict))

            trained_model = fit_model(
                work_dir,
                model,
                data_type,
                sp_train_data,
                train_labs,
                train_s,
                sp_val_data,
                val_labs,
                val_s,
                class_weights,
                ua.experiment_name,
            )
            evaluate_model(
                trained_model,
                sp_test_data,
                test_labs,
                test_s,
                work_dir,
                ua.experiment_name,
                data_type,
                lab_dict,
            )
            )



# # initialize a random search with a 3-fold cross-validation and then
# # start the hyperparameter search process
# print("[INFO] performing grid search...")
# searcher = GridSearchCV(estimator=model, n_jobs=-1, cv=1
# 	param_distributions=grid, scoring="accuracy")
# searchResults = searcher.fit(train_data, [train_labs, train_s])
#
#
# # summarize grid search information
# bestScore = searchResults.best_score_
# bestParams = searchResults.best_params_
# print("[INFO] best score is {:.2f} using {}".format(bestScore,
# 	bestParams))
#
# # extract the best model, make predictions on our data, and show a
# # classification report
# print("[INFO] evaluating the best model...")
# bestModel = searchResults.best_estimator_
# accuracy = bestModel.score(testData, testLabels)
# print("accuracy: {:.2f}%".format(accuracy * 100))

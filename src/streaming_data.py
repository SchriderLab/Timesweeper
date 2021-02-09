# Ripped straight from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# Very flexible

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, list_IDs, labels, batch_size, dim, n_classes, shuffle=True):
        "Initialization"
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labs_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_labs_temp):
        "Generates data containing batch_size samples"
        # Initialization
        X = np.empty((self.batch_size * 100, *self.dim))
        y = np.empty((self.batch_size * 100), dtype=int)

        # Generate data
        hundotracker = [0, 100]
        for ID, lab in zip(list_IDs_temp, list_labs_temp):
            # Store sample
            # fmt: off
            X[hundotracker[0]: hundotracker[1],] = np.load(ID)["X"]
            # fmt: on
            # Store class
            y[hundotracker[0] : hundotracker[1]] = [lab] * 100

            for j in [0, 1]:
                hundotracker[j] = hundotracker[j] + 100

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

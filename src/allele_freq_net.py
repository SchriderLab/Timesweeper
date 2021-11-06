import hap_networks as hn
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

ua = hn.parse_ua()
lab_dict = {"hard": 0, "neut": 1, "soft": 2}

base_dir = os.path.dirname(ua.input_npz)
ids, data = hn.get_data(ua.input_npz)
print("Starting training process.")
print("Base directory:", base_dir)
print("Input data file:", ua.input_npz)

num_ids = to_categorical(np.array([lab_dict[lab] for lab in ids]), len(set(ids)))
data = np.swapaxes(data, 1, 2)
data_dim = data.shape[1:]
print("Splitting Partition")
(
    train_data,
    val_data,
    test_data,
    train_labs,
    val_labs,
    test_labs,
) = hn.split_partitions(data, num_ids)

print("Data shape (samples, snps):", data.shape)

model = hn.create_hapsTS_model(data_dim)
print(model.summary())
trained_model = hn.fit_model(
    base_dir, model, train_data, train_labs, val_data, val_labs, ua.schema_name
)
hn.evaluate_model(trained_model, test_data, test_labs, base_dir, ua.schema_name)

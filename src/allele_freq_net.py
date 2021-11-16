import hap_networks as hn
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
import os

ua = hn.parse_ua()
lab_dict = {"neut": 0, "hard": 1, "soft": 2}

base_dir = os.path.dirname(ua.input_npz)
ids, data = hn.get_data(ua.input_npz)
print("Starting training process.")
print("Base directory:", base_dir)
print("Input data file:", ua.input_npz)
print("Number samples:", len(data))

num_ids = to_categorical(np.array([lab_dict[lab] for lab in ids]), len(set(ids)))
data = np.swapaxes(data, 1, 2)  # Needs to be in correct dims order for Conv1D layer

print(f"{len(data)} samples in dataset.")

datadim = data.shape[1:]
print("TS Data shape (samples, timepoints, haps):", data.shape)
print("\n")

print("Splitting Partition")
(
    ts_train_data,
    ts_val_data,
    ts_test_data,
    train_labs,
    val_labs,
    test_labs,
) = hn.split_partitions(data, num_ids)

# Time-series model training and evaluation
print("Training time-series model.")
model = hn.create_hapsTS_model(datadim)
print(model.summary())

trained_model = hn.fit_model(
    base_dir, model, ts_train_data, train_labs, ts_val_data, val_labs, ua.schema_name
)
hn.evaluate_model(trained_model, ts_test_data, test_labs, base_dir, ua.schema_name)

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
model = hn.create_haps1Samp_model(sp_datadim)
print(model.summary())

trained_model = hn.fit_model(
    base_dir, model, sp_train_data, train_labs, sp_val_data, val_labs, ua.schema_name
)
hn.evaluate_model(trained_model, sp_test_data, test_labs, base_dir, ua.schema_name)


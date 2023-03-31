from tensorflow.keras import layers
from tensorflow.keras.models import Model

# fmt: off
def create_TS_class_model(datadim, n_class):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)
    h = layers.Conv1D(64, 3, activation="relu", padding="same")(model_in)
    h = layers.Conv1D(64, 3, activation="relu", padding="same")(h)
    h = layers.MaxPooling1D(pool_size=3, padding="same")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(512, activation="relu")(h)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(264, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    class_output = layers.Dense(n_class, activation="softmax", name="class_output")(h)

    model = Model(inputs=[model_in], outputs=[class_output], name="Timesweeper_Class")
    model.compile(
        loss={"class_output":"categorical_crossentropy"},
        optimizer="adam",
        metrics={"class_output": "accuracy"},
    )

    return model

def create_TS_reg_model(datadim):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)
    h = layers.Conv1D(64, 3, activation="relu", padding="same")(model_in)
    h = layers.Conv1D(64, 3, activation="relu", padding="same")(h)
    h = layers.MaxPooling1D(pool_size=3, padding="same")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(264, activation="relu")(h)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(264, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    reg_output = layers.Dense(1, activation="linear", name="reg_output")(h)

    model = Model(inputs=[model_in], outputs=[reg_output], name="Timesweeper_Reg")
    model.compile(
        loss={"reg_output":"mse"},
        optimizer="adam",
        metrics={"reg_output": "mse"},
    )

    return model


# fmt: off
def create_2D_TS_class_model(datadim, n_class):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)
    h = layers.Conv2D(64, 3, activation="relu", padding="same")(model_in)
    h = layers.Conv2D(64, 3, activation="relu", padding="same")(h)
    h = layers.MaxPooling2D(pool_size=3, padding="same")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(512, activation="relu")(h)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(264, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    class_output = layers.Dense(n_class, activation="softmax", name="class_output")(h)

    model = Model(inputs=[model_in], outputs=[class_output], name="2DTimesweeper_Class")
    model.compile(
        loss={"class_output":"categorical_crossentropy"},
        optimizer="adam",
        metrics={"class_output": "accuracy"},
    )

    return model

def create_2D_TS_reg_model(datadim):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)
    
    h = layers.Conv2D(64, 3, activation="relu", padding="same")(model_in)
    h = layers.Conv2D(64, 3, activation="relu", padding="same")(h)
    h = layers.MaxPooling2D(pool_size=3, padding="same")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(264, activation="relu")(h)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(264, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    reg_output = layers.Dense(1, activation="relu", name="reg_output")(h)

    model = Model(inputs=[model_in], outputs=[reg_output], name="2DTimesweeper_Reg")
    model.compile(
        loss={"reg_output":"mse"},
        optimizer="adam",
        metrics={"reg_output": "mse"},
    )

    return model


# fmt: off
def create_big_TS_class_model(datadim, n_class):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)
    h = layers.Conv1D(1028, 3, activation="relu", padding="same")(model_in)
    h = layers.Conv1D(1028, 3, activation="relu", padding="same")(h)
    h = layers.MaxPooling1D(pool_size=3, padding="same")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(1028, activation="relu")(h)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(512, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    class_output = layers.Dense(n_class, activation="softmax", name="class_output")(h)

    model = Model(inputs=[model_in], outputs=[class_output], name="Big_Timesweeper_Class")
    model.compile(
        loss={"class_output":"categorical_crossentropy"},
        optimizer="adam",
        metrics={"class_output": "accuracy"},
    )

    return model

def create_big_TS_reg_model(datadim):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)
    h = layers.Conv1D(1028, 3, activation="relu", padding="same")(model_in)
    h = layers.Conv1D(1028, 3, activation="relu", padding="same")(h)
    h = layers.MaxPooling1D(pool_size=3, padding="same")(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Flatten()(h)

    h = layers.Dense(1028, activation="relu")(h)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(512, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    reg_output = layers.Dense(1, activation="relu", name="reg_output")(h)

    model = Model(inputs=[model_in], outputs=[reg_output], name="Big_Timesweeper_Reg")
    model.compile(
        loss={"reg_output":"mse"},
        optimizer="adam",
        metrics={"reg_output": "mse"},
    )

    return model


# fmt: off
def create_1tp_class_model(datadim, n_class):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)

    h = layers.Dense(264, activation="relu")(model_in)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(56, activation="relu")(h)
    h = layers.Dropout(0.1)(h)
    class_output = layers.Dense(n_class, activation="softmax", name="class_output")(h)

    model = Model(inputs=[model_in], outputs=[class_output], name="1TP_Timesweeper_Class")
    model.compile(
        loss={"class_output":"categorical_crossentropy"},
        optimizer="adam",
        metrics={"class_output": "accuracy"},
    )

    return model


def create_1tp_reg_model(datadim):
    """
    Returns:
        Model: Keras compiled model.
    """
    model_in = layers.Input(datadim)
    h = layers.Dense(264, activation="relu")(model_in)
    h = layers.Dropout(0.2)(h)        
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    h = layers.Dense(56, activation="relu")(h)
    h = layers.Dropout(0.1)(h)

    reg_output = layers.Dense(1, activation="relu", name="reg_output")(h)

    model = Model(inputs=[model_in], outputs=[reg_output], name="1TP_Timesweeper_Reg")
    model.compile(
        loss={"reg_output":"mse"},
        optimizer="adam",
        metrics={"reg_output": "mse"},
    )

    return model
# fmt: on


def create_rnn_class_model(datadim, n_class):
    model_in = layers.Input(datadim)
    h = layers.LSTM(100, input_shape=datadim, activation="tanh")(model_in)
    h = layers.Dropout(0.1)(h)

    class_output = layers.Dense(n_class, activation="softmax", name="class_output")(h)

    model = Model(inputs=[model_in], outputs=[class_output], name="Timesweeper_Class")
    model.compile(
        loss={"class_output": "categorical_crossentropy"},
        optimizer="adam",
        metrics={"class_output": "accuracy"},
    )

    return model


def create_rnn_reg_model(datadim):
    model_in = layers.Input(datadim)
    h = layers.LSTM(100, input_shape=datadim, activation="tanh")(model_in)
    h = layers.Dropout(0.1)(h)

    reg_output = layers.Dense(1, activation="relu", name="reg_output")(h)

    model = Model(inputs=[model_in], outputs=[reg_output], name="Timesweeper_Reg")
    model.compile(
        loss={"reg_output": "mse"}, optimizer="adam", metrics={"reg_output": "mse"},
    )

    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x + res


def create_transformer_reg_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="relu")(x)

    model = Model(inputs, outputs, name="Timesweeper_Transformer_Reg")

    model.compile(
        loss="mse", optimizer="adam", metrics="mse",
    )

    return model


def create_transformer_class_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_class=3,
):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_class, activation="softmax")(x)

    model = Model(inputs, outputs, name="Timesweeper_Transformer_Class")

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics="accuracy",
    )

    return model

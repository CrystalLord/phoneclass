import keras
from keras.layers import LSTM, Dense, Activation


def get_set1(feature_size, sample_count, batch_size, output_num):
    # Declare the sizes of the neural network layers.
    sizes = [70, 40, 30, output_num]
    # Construct the architecture
    arch = [LSTM(sizes[0],
                 batch_input_shape=(batch_size,
                                    sample_count,
                                    feature_size),
                 return_sequences=True),
            LSTM(sizes[1], return_sequences=True),
            Dense(sizes[2]),
            Dense(sizes[3], activation="softmax")]
    # Define the metrics to optimise
    metrics = [keras.metrics.categorical_accuracy]

    return {"arch": arch,
            "optimizer": keras.optimizers.RMSprop(),
            "loss": "mean_squared_error",
            "metrics": metrics}

def get_set2(feature_size, sample_count, batch_size, output_num):
    # Declare the sizes of the neural network layers.
    sizes = [70, 40, 30, output_num]
    # Construct the architecture
    arch = [LSTM(sizes[0],
                 input_dim=feature_size,
                 return_sequences=True),
            LSTM(sizes[1], return_sequences=True),
            Dense(sizes[2]),
            Dense(sizes[3], activation="softmax")]
    # Define the metrics to optimise
    metrics = [keras.metrics.categorical_accuracy]

    return {"arch": arch,
            "optimizer": keras.optimizers.RMSprop(),
            "loss": "mean_squared_error",
            "metrics": metrics}

def get_set3(feature_size, sample_count, batch_size, output_num):
    # Declare the sizes of the neural network layers.
    sizes = [70, 40, 30, output_num]
    # Construct the architecture
    arch = [LSTM(sizes[0],
                 input_dim=feature_size,
                 return_sequences=True),
            LSTM(sizes[1], return_sequences=True),
            Dense(sizes[2]),
            Dense(sizes[3], activation="softmax")]
    # Define the metrics to optimise
    metrics = [keras.metrics.categorical_accuracy]

    return {"arch": arch,
            "optimizer": keras.optimizers.RMSprop(),
            "loss": "mean_squared_error",
            "metrics": metrics}

# ============================================================================
#
# The actual settings found here.
active_set = get_set2
training_data =

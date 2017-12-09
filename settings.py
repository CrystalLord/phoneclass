import keras
from keras.layers import LSTM, Dense, Activation

X_FP = "/mnt/tower_1tb/neural_networks/training_data/training.npy"
Y_FP = "/mnt/tower_1tb/neural_networks/training_data/target.npy"
LABELS_FP = "/mnt/tower_1tb/neural_networks/training_data/labels.csv"

DB_X_FP = "/mnt/tower_1tb/neural_networks/training_data/decibel2/training.npy"
DB_Y_FP = "/mnt/tower_1tb/neural_networks/training_data/decibel2/target.npy"
DB_LABELS_FP = "/mnt/tower_1tb/neural_networks/training_data/decibel2/labels.csv"

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
    sizes = [100, 100, 100, output_num]
    # Construct the architecture
    arch = [LSTM(sizes[0],
                 input_dim=feature_size,
                 return_sequences=True),
            LSTM(sizes[1],
                 return_sequences=True),
            LSTM(sizes[2],
                 return_sequences=True),
            Dense(output_num, activation="softmax")]
    # Define the metrics to optimise
    metrics = [keras.metrics.categorical_accuracy]

    return {"arch": arch,
            "optimizer": keras.optimizers.RMSprop(lr=0.01),
            "loss": "mean_squared_error",
            "metrics": metrics}

def get_set4(feature_size, sample_count, batch_size, output_num):
    # Declare the sizes of the neural network layers.
    sizes = [60, 50, 40, 30, 30, 30, output_num]
    # Construct the architecture
    arch = [LSTM(sizes[0],
                 input_dim=feature_size,
                 return_sequences=True),
            LSTM(sizes[1],
                 return_sequences=True),
            LSTM(sizes[2],
                 return_sequences=True),
            LSTM(sizes[3],
                 return_sequences=True),
            LSTM(sizes[4],
                 return_sequences=True),
            LSTM(sizes[5],
                 return_sequences=True),
            Dense(output_num, activation="softmax")]
    # Define the metrics to optimise
    metrics = [keras.metrics.categorical_accuracy]

    return {"arch": arch,
            "optimizer": keras.optimizers.RMSprop(lr=0.04),
            "loss": "mean_squared_error",
            "metrics": metrics}

def get_set5(feature_size, sample_count, batch_size, output_num):
    # Declare the sizes of the neural network layers.
    sizes = [60, 50, 40, 30, 30, 30, output_num]
    # Construct the architecture
    arch = [LSTM(sizes[0],
                 input_dim=feature_size,
                 return_sequences=True),
            Dense(sizes[1]),
            Dense(sizes[2]),
            Dense(sizes[3]),
            Dense(sizes[4]),
            Dense(sizes[5]),
            Dense(output_num, activation="softmax")]
    # Define the metrics to optimise
    metrics = [keras.metrics.categorical_accuracy]

    return {"arch": arch,
            "optimizer": keras.optimizers.RMSprop(lr=0.04),
            "loss": "mean_squared_error",
            "metrics": metrics}

# ============================================================================
#
# The actual settings found here.
active_set = get_set3
training_indices = range(4)

input_data = X_FP
target_data = Y_FP
labels = LABELS_FP

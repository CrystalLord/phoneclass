import keras
from keras import optimizers, metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np
import os.path as path
import sys
import time

import utils
import dict_utils
import consts as ct
from audioclip import AudioClip
from ac_setup import ac_setup2
import bell_training_gen

DEBUG = True
DEBUG_X_FP = "/mnt/tower_1tb/neural_networks/training_data/training.npy"
DEBUG_Y_FP = "/mnt/tower_1tb/neural_networks/training_data/target.npy"
LABELS_FP = "/mnt/tower_1tb/neural_networks/training_data/labels.csv"


# ===========================================================================
# HYPERPARAMETERS

LAYER_SIZES = [70, # LSTM front layer
               40,
               30,
               37]
LOSS = "mean_squared_error"
OPT = optimizers.RMSprop()

# ============================================================================

def main(args):

    if len(args) != 3:
        print("Usage: nn (train|load) (epochs|model_to_load)")
        return

    #train_x, train_y = gather_audio()


    if args[1] == "train":
        epochs = int(args[2])
        #x_fp = input("input training data> ")
        #train_x = np.load(x_fp)
        #y_fp = input("input target data> ")
        #train_y = np.load(y_fp)
        if DEBUG:
            train_x = np.load(DEBUG_X_FP)
            train_y = np.load(DEBUG_Y_FP)

        # Get data sizes.
        arch_feat = train_x.shape[2]
        arch_samps = train_x.shape[1]
        batch_size = 1

        np.save(ct.DATA_DIR+"debug_y", train_y)
        model = buildmodel(arch_feat, arch_samps, batch_size)
        train_save(model, train_x, train_y, epochs, batch_size)
    elif args[1] == "load":
        model = load(args[2])
        #from keras.utils import plot_model
        #plot_model(model, to_file="/mnt/tower_1tb/model_plot")

    if DEBUG:
        labels = dict_utils.load_dict(LABELS_FP)
        test_on_audio_file(model, labels,
            "/mnt/tower_1tb/neural_networks/misc/test_data.wav")
    #test_x = np.copy(train_x)
    #pred_save(model, test_x, batch_size)


def train_save(model, train_x, train_y, epochs, batch_size):
    """Train and save the neural network"""
    model.fit(train_x, train_y,
            epochs=epochs,
            validation_split=0.1,
            batch_size=batch_size)
    print("Model Trained.")
    filename = ct.MODELDIR + "model" + utils.tstamp()
    model.save(filename)
    print("Model saved to " + filename)


def pred_save(model, test_x, batch_size):
    """Run a prediction test of the neural network and save result"""
    pred = model.predict(test_x, batch_size=batch_size)
    filename = ct.PREDDIR + "pred_new" + utils.tstamp()
    np.save(filename, pred)
    print("Prediction saved to " + filename + ".npy")


def load(fp):
    """Load a keras model file given the filename"""
    return keras.models.load_model(ct.MODELDIR + fp)


def buildmodel(feature_size, sample_count, batch_size):
    """Construct the neural network architecture and return it.
    """
    model = Sequential()
    model.add(LSTM(
        LAYER_SIZES[0],
        batch_input_shape=(batch_size, sample_count, feature_size),
        return_sequences=True))
    model.add(LSTM(LAYER_SIZES[1], return_sequences=True))
    model.add(Dense(LAYER_SIZES[2]))
    model.add(Dense(LAYER_SIZES[3], activation="softmax"))

    #sgd = optimizers.SGD(lr=0.02, momentum=0.01)
    model.compile(
            loss=LOSS,
            #optimizer=sgd,
            optimizer=OPT,
            metrics=[metrics.categorical_accuracy])
    return model


def gather_audio():
    """Gather audio data and convert to training data format"""
    gather_new = input("New data?> ").lower()
    if path.isfile(ct.DATA_DIR + "trainx.npy") \
            and path.isfile(ct.DATA_DIR + "trainy.npy") \
            and not (gather_new == "y"):
        # Don't gather new data.
        print("Found training data. Loading...")
        train_x = np.load(ct.DATA_DIR + "trainx.npy")
        train_y = np.load(ct.DATA_DIR + "trainy.npy")
    else:
        # Gather new data.
        print("Generating new training data...")
        acs = ac_setup2()
        train_x, train_y = utils.combtraining(acs)
        np.save(ct.DATA_DIR + "trainx", train_x)
        np.save(ct.DATA_DIR + "trainy", train_y)

    return train_x, train_y


def test_on_audio_file(model, labels, audio_file):
    ac = AudioClip(audio_file, mintime=4.74)
    batch_x = ac.raw_x_batch()
    batch_x = batch_x[np.newaxis, :, :]
    pred = model.predict(batch_x, 1)
    print(pred)
    print(utils.index_to_label(pred, labels))
    np.save("/mnt/tower_1tb/prediction", pred)

if __name__ == "__main__":
    main(sys.argv)

import keras
from keras import optimizers, metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np
import os.path as path
import sys
import time

import utils
import consts as ct
from audioclip import AudioClip
from ac_setup import ac_setup2

def main(args):
    train_x, train_y = gather_audio()

    # Get data sizes.
    arch_feat = train_x.shape[2]
    arch_samps = train_x.shape[1]
    batch_size = 1

    model = buildmodel(arch_feat, arch_samps, batch_size)
    if args[1] == "train":
        epochs = int(args[2])
        train_save(model, train_x, train_y, epochs, batch_size)
    elif args[1] == "load":
        model = load(args[2])

    test_x = np.copy(train_x)
    #np.random.shuffle(test_x)
    pred_save(model, test_x, batch_size)


def train_save(model, train_x, train_y, epochs, batch_size):
    """Train and save the neural network"""
    model.fit(train_x, train_y,
            epochs=epochs,
            validation_split=0.5,
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
        32,
        batch_input_shape=(batch_size, sample_count, feature_size),
        return_sequences=True))
    model.add(Dense(16))
    model.add(Dense(5, activation="softmax"))

    sgd = optimizers.SGD(lr=0.02, momentum=0.01)
    model.compile(
            loss="mean_squared_error",
            optimizer=sgd,
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

if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3

import keras
from keras import optimizers, metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np
import os.path as path
import sys
import time
import argparse

import utils
import dict_utils
import consts as ct
import settings
import bell_training_gen
from audioclip import AudioClip

DEBUG = True
DEBUG_X_FP = "/mnt/tower_1tb/neural_networks/training_data/training.npy"
DEBUG_Y_FP = "/mnt/tower_1tb/neural_networks/training_data/target.npy"
LABELS_FP = "/mnt/tower_1tb/neural_networks/training_data/labels.csv"


def main():
    args = parse_args()
    # Run the correct main program to operate on these args
    model = args.func(args)

    if DEBUG:
        labels = dict_utils.load_dict(LABELS_FP)
        test_on_audio_file(model, labels,
            "/mnt/tower_1tb/neural_networks/misc/test_data.wav")


def main_train(args):
    """Handle training program"""
    # If we want to train a new neural network, this will be set.
    epochs = int(args.epochs)
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
    return model

def main_load(args):
    """Handle loading program"""
    model = load(args.model_file)
    return model

# END MAIN ===================================================================

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
    return keras.models.load_model(fp)


def buildmodel(feature_size, sample_count, batch_size):
    """Construct the neural network architecture and return it.
    """
    model = Sequential()

    # Retrive the neural network architecture and settings from
    # settings.py
    set = settings.active_set(feature_size, sample_count, batch_size, 37)

    # Add the layers.
    for layer in set["arch"]:
        model.add(layer)

    # Compile the model with the settings we declared.
    model.compile(
            loss=set["loss"],
            optimizer=set["optimizer"],
            metrics=set["metrics"])
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
    #print(pred)
    utils.index_to_label(pred, labels)
    #print(utils.index_to_label(pred, labels))
    np.save("/mnt/tower_1tb/prediction", pred)


def parse_args():
    """Parse arguments passed to the program"""
    parser = argparse.ArgumentParser(prog="nn")

    subparsers = parser.add_subparsers(
        title="operations",
        help="should be either 'train' or 'load'. Determines"
        + " whether the NN should load a model or train a new"
        + " one.")
    # Parse the subparsers.
    train = subparsers.add_parser("train")
    load = subparsers.add_parser("load")
    train.add_argument("-e",
                       "--epochs",
                       default=10,
                       metavar="EPOCHS",
                       help="Number of epochs to train on.")
    load.add_argument("model_file",
                      metavar="MODEL",
                      help="Model to load")

    train.set_defaults(func=main_train)
    load.set_defaults(func=main_load)
    return parser.parse_args()

if __name__ == "__main__":
    main()

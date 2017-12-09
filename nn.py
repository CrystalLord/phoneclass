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

def main():
    args = parse_args()
    # Run the correct main program to operate on these args
    model = args.func(args)

def main_train(args):
    """Handle training program"""
    # If we want to train a new neural network, this will be set.
    epochs = int(args.epochs)

    train_x = np.load(settings.input_data)
    train_y = np.load(settings.target_data)

    # Get data sizes.
    arch_feat = train_x.shape[2]
    arch_samps = train_x.shape[1]
    batch_size = 1

    # Pick only a small sample of the data we want to train on.
    train_x, train_y = pick_training_data(train_x, train_y,
                                          settings.training_indices)

    np.save(ct.DATA_DIR+"debug_y", train_y)
    model = buildmodel(arch_feat, arch_samps, batch_size)
    # Train and save the model.
    train_save(model, train_x, train_y, epochs, batch_size,
               save=(not args.nosave))
    # Run a prediction test
    pred_save(model, train_x, 1, target_y=train_y, save=(not args.nosave))
    return model

def main_load(args):
    """Handle loading program"""
    model = load(args.model_file)
    return model

# END MAIN ===================================================================

def train_save(model,
               train_x,
               train_y,
               epochs,
               batch_size,
               save=True,
               validation_split=0.1):
    """Train and save the neural network

    Saves the model to the MODELDIR defined in consts.

    Arguments
    model -- The compiled Neural Network model to train.
    train_x -- The input training data. Must be a 3D Numpy array
    train_y -- The target training data. Must be a 3D Numpy array
    epochs -- Number of epochs to train with.
    batch_size -- The size of each batch. Set to 1 for online training.

    Keyword Arguments
    save -- Boolean to indicate whether we want to save the model.
    validation_split -- The fraction of training data we want to reserve for
                        validation.
    """
    model.fit(train_x, train_y,
              epochs=epochs,
              validation_split=validation_split,
              batch_size=batch_size)
    print("Model Trained.")
    filename = ct.MODELDIR + "model" + utils.tstamp()
    if save:
        model.save(filename)
        print("Model saved to " + filename)


def pred_save(model, test_x, batch_size, target_y=None, save=True):
    """Run a prediction test of the neural network and save result"""
    pred = model.predict(test_x, batch_size=batch_size)
    if save:
        filename = ct.PREDDIR + "pred" + utils.tstamp()
        np.save(filename, pred)
        print("Prediction saved to " + filename + ".npy")
        if target_y is not None:
            filename = ct.PREDDIR + "target" + utils.tstamp()
            np.save(filename, target_y)


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
    utils.index_to_label(pred, labels)
    #print(pred)
    #print(utils.index_to_label(pred, labels))
    #np.save("/mnt/tower_1tb/prediction", pred)


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
    train.add_argument("--nosave",
                       action='store_true',
                       help="If set, save nothing.")
    load.add_argument("model_file",
                      metavar="MODEL",
                      help="Model to load")

    train.set_defaults(func=main_train)
    load.set_defaults(func=main_load)
    return parser.parse_args()

def pick_training_data(input_data, target_data, indices):
    """Choose only certain batches from a large set of training data.

    Arguments
    input_data -- Input data we will pick from.
    target_data -- Target data we will pick from.
    indices -- Tuple of indices. If empty, will return the original data.

    Outputs a tuple of the form (picked_input_data, picked_target_data)
    """

    if len(indices) == 0:
        return input_data, target_data

    num_batches = len(indices)

    input_data_y = input_data.shape[1]
    input_data_z = input_data.shape[2]
    target_data_y = target_data.shape[1]
    target_data_z = target_data.shape[2]
    new_input_data = np.zeros((num_batches, input_data_y, input_data_z))
    new_target_data = np.zeros((num_batches, target_data_y, target_data_z))

    for i in indices:
        new_input_data[i, :, :] = input_data[i, :, :]
        new_target_data[i, :, :] = target_data[i, :, :]

    return (new_input_data, new_target_data)

if __name__ == "__main__":
    main()

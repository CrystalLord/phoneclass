#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import argparse

import consts as ct
import dict_utils

LABEL_DICT_FP = "/mnt/tower_1tb/neural_networks/training_data/labels.csv"
PLOT_Y_LIM = (-0.1, 1.1)
TARGET_ALPHA = 1
PRED_STYLE = "-"
TARGET_STYLE = ":"

def main(args):
    args = parse_args()

    if args.prediction is not None:
        pred_data = np.load(args.prediction)
    else:
        pred_data = None

    if args.target is not None:
        target_data = np.load(args.target)
    else:
        target_data = None

    label_dict = dict_utils.load_dict(LABEL_DICT_FP)
    if not args.is_diff:
        ipa_sample_plot(label_dict, pred_out=pred_data, target_out=target_data,
                        one_sym=args.symbol, title=args.title)
    else:
        difference_plot(label_dict, pred_data, target_data,
                        one_sym=args.symbol, title=args.title)


def ipa_sample_plot(label_dict, pred_out=None, target_out=None, sr=16000,
        title=None, one_sym=None):
    """Plotting function to showcase neural network outputs compared to
    the target vaules.

    label_dict -- Python dictionary which has the format {ipa_sym, index}
    pred_out -- Neural network prediction output.
    target_out -- Target output for the neural network.
    sr -- Sample rate of the outputs
    title -- Title of the plot.
    """

    if pred_out is not None:
        x = np.arange(pred_out.shape[1] * pred_out.shape[0])
    elif target_out is not None:
        x = np.arange(target_out.shape[1] * target_out.shape[0])
    else:
        raise ValueError("Require either pred_out or target_out args")

    sample_count = len(x)

    if one_sym is None:
        for sym in label_dict.keys():
            # Retrieve prediction y values for a specific IPA symbol.
            # We need the index of the IPA symbol in the training array.
            index = int(label_dict[sym])
            linecolor = gen_color()
            if pred_out is not None:
                prediction_y = one_index(pred_out, index)
                plt.plot(x, prediction_y[:sample_count], PRED_STYLE,
                         color=linecolor)
            if target_out is not None:
                reality_y = one_index(target_out, index)
                plt.plot(x, reality_y[:sample_count], TARGET_STYLE,
                         color=linecolor, alpha=TARGET_ALPHA)
    else:
        index = int(label_dict[one_sym])
        linecolor = gen_color()
        if pred_out is not None:
            prediction_y = one_index(pred_out, index)
            plt.plot(x, prediction_y[:sample_count], PRED_STYLE,
                     color=linecolor)
        if target_out is not None:
            reality_y = one_index(target_out, index)
            plt.plot(x, reality_y[:sample_count], TARGET_STYLE,
                     color=linecolor, alpha=TARGET_ALPHA)

    plt.ylim(PLOT_Y_LIM)

    if title is not None:
        plt.title(title)

    plt.xlabel('Sample Number')
    plt.ylabel('Phoneme Confidence')

    plt.show()

def difference_plot(label_dict, pred_out, target_out, sr=16000, title=None,
        one_sym=None):
    """"""

    if target_out is None or pred_out is None:
        raise ValueError("Require both pred_out and target_out args for diff")


    diff = target_out - pred_out

    x = np.arange(diff.shape[1] * diff.shape[0])

    sample_count = len(x)

    if one_sym is None:
        for sym in label_dict.keys():
            # Retrieve prediction y values for a specific IPA symbol.
            # We need the index of the IPA symbol in the training array.
            index = int(label_dict[sym])
            linecolor = gen_color()
            diff_y = one_index(diff, index)
            plt.plot(x, diff_y[:sample_count], PRED_STYLE,
                        color=linecolor)
    else:
            index = int(label_dict[one_sym])
            linecolor = gen_color()
            diff_y = one_index(diff, index)
            plt.plot(x, diff_y[:sample_count], PRED_STYLE,
                        color=linecolor)

    plt.ylim(PLOT_Y_LIM)

    if title is not None:
        plt.title(title)

    plt.xlabel('Sample Number')
    plt.ylabel('Confidence Error')
    plt.grid(b=True, color=(0.8,0.8,0.8), linestyle='--');
    plt.show()


def one_index(data, index):
    batch_columns = data[:, :, index]
    one_column = np.concatenate(batch_columns)
    return one_column

def one_symbol(data, symbol, label_dict):
    """Retrieve y-axis single dimension data for a given IPA symbol

    Arguments
    data -- Raw neural network output
    symbol -- String indicating which IPA symbol to use
    """
    column = label_dict[symbol]
    batch_columns = data[:, :, column]
    one_column = np.concatenate(batch_columns)
    return one_column

def gen_color():
    r = random.randrange(0, 255)/255
    g = random.randrange(0, 255)/255
    b = random.randrange(0, 255)/255
    return (r, g, b)

def parse_args():
    parser = argparse.ArgumentParser(prog="sampleplot")
    parser.add_argument("-d",
                       "--is_diff",
                       action="store_true",
                       help="Plot difference of prediction and target.")
    parser.add_argument("-p",
                        "--prediction",
                        metavar="PRED",
                        default=None,
                        help="Prediction Neural Network Output")
    parser.add_argument("-t",
                       "--target",
                       metavar="TARGET",
                       default=None,
                       help="Target Neural Network Output")
    parser.add_argument("-s",
                       "--symbol",
                       metavar="SYMBOL",
                       default=None,
                       help="If set, will only plot the given symbol")
    parser.add_argument("--title",
                       metavar="TITLE",
                       default=None,
                       help="Title of the plot")
    return parser.parse_args()

if __name__ == "__main__":
    main(sys.argv)

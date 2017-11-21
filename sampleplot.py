import numpy as np
import matplotlib.pyplot as plt
import sys
import random

import consts as ct

COLORS = ['r',
        'b',
        'g',
        'c',
        'm',
        'y',
        'k',
        'w']

def main(args):
    if len(args) != 3:
        print("sampleplot PRED REAL")
        return
    pred_data = np.load(ct.PREDDIR + args[1])
    real_data = np.load(ct.DATA_DIR + args[2])
    ipa_sample_plot(pred_data, real_data)

def ipa_sample_plot(pred_out, reality_out):
    used_colors = []
    for sym in ct.IPA_MAP.keys():
        pred_data = one_symbol(pred_out, sym)
        real_data = one_symbol(reality_out, sym)

        incr = 0
        linecolor = COLORS[incr]
        while linecolor in used_colors:
            incr += 1
            linecolor = COLORS[incr]
            if len(used_colors) == len(COLORS):
                break
        used_colors.append(linecolor)
        plt.plot(real_data, '-', color=linecolor)
        plt.plot(pred_data, '--', color=linecolor)
    plt.show()

def single_plot(reality_out, label_dict):
    """Plot only a single sample plot"""
    used_colors = []
    count = 0
    max_count = 5
    for k in label_dict.keys():
        if count > max_count:
            break
        print("ipa: " + k)
        print("index: " + str(label_dict[k]))
        data = one_index(reality_out, label_dict[k], label_dict)
        incr = 0
        linecolor = COLORS[incr]
        try:
            while linecolor in used_colors:
                incr += 1
                linecolor = COLORS[incr]
                if len(used_colors) >= len(COLORS):
                    break
        except IndexError:
            linecolor = gen_color()
        used_colors.append(linecolor)
        plt.plot(data, '-', color=linecolor)
        count += 1
    plt.show()

def one_index(data, index, label_dict):
    batch_columns = data[:, :, index]
    one_column = np.concatenate(batch_columns)
    return one_column

def one_symbol(data, symbol, label_dict=ct.IPA_NUM):
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
    r = random.randrange(0, 255)
    g = random.randrange(0, 255)
    b = random.randrange(0, 255)
    return (r, g, b)

if __name__ == "__main__":
    main(sys.argv)

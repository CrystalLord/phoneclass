import numpy as np
import matplotlib.pyplot as plt
import sys

import consts as ct

COLORS = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']

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

def one_symbol(data, symbol):
    """Retrieve y-axis single dimension data for a given IPA symbol

    data -- Raw neural network output
    symbol -- String indicating which IPA symbol to use
    """
    column = ct.IPA_NUM[symbol]
    batch_columns = data[:,:,column]
    one_column = np.concatenate(batch_columns)
    return one_column

if __name__ == "__main__":
    main(sys.argv)

import librosa
import librosa.display
import numpy as np
import time
import datetime
import dict_utils

def dbspec(spec):
    """Convert spectrogram data from power to decibels"""
    return librosa.power_to_db(spec)

def cutwave(waveform, sr, time):
    """Pads waveform data with silence at the end to reach minimum length.
    """
    new_waveform = waveform
    minlen = round(time * sr)
    diff = minlen - waveform.size
    if diff > 0:
        print("cutwave--Padding to " + str(time) + "s")
        diff_wave = np.zeros(diff)
        new_waveform = np.concatenate((waveform, diff_wave))
    if diff < 0:
        print("cutwave--Cutting to " + str(time) + "s")
        new_waveform = waveform[0:minlen]
    return new_waveform

def combtraining(audioclips, use_bell=False, db=False):
    """Take in a list of audioclips and merge their training data.

    Returns a tuple of the form
    (training_x, training_y)
    """
    batches_x = []
    batches_y = []

    for i, ac in enumerate(audioclips):
        if use_bell:
            train_x, train_y = ac.bell_batch(db=db)
        else:
            train_x, train_y = ac.batch(db=db)
        batches_x.append(train_x)
        batches_y.append(train_y)

    full_batch_x = np.stack(batches_x, axis=0)
    full_batch_y = np.stack(batches_y, axis=0)

    return full_batch_x, full_batch_y

def plot_spec(spec, sr):
    """Plot a spectrogram"""
    import librosa.display
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec, x_axis='time', sr=sr)

    plt.colorbar()
    plt.title('SPEC')
    plt.tight_layout()
    plt.show()

def sum_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors to be summed not same length")
    else:
        return [v1[i] + v2[i] for i in range(len(v1))]

def tstamp():
    """Generate a unique timestamp"""
    now = datetime.datetime.now()
    t = now.timetuple()
    return str(t[0]) + str(t[1]) + str(t[2]) \
        + "_" + str(t[3]) + "-" + str(t[4]) + "-" + str(t[5])

def flatten(l):
    """Flatten a nested list"""
    new_list = []
    for i in l:
        if isinstance(i, list):
            new_list += flatten(i)
        else:
            new_list.append(i)
    return new_list

def flatten_strlist(l):
    """Flatten a nested list, but also splits strings"""
    new_list = []
    for i in l:
        if isinstance(i, list) or isinstance(i, str):
            new_list += flatten(i)
        else:
            new_list.append(i)
    return new_list

def slashend(s):
    """End a string with a forward slash. Do nothing if exists already."""
    if s[-1] == "/":
        return s
    else:
        return s + "/"

def index_to_label(np_array, labels):
    """Given a 3-dimensional numpy array of outputs, align labels"""
    new_list = []
    flip_labels = dict_utils.invert_dict(labels)
    for batch_index in range(np_array.shape[0]):
        batch = np_array[batch_index, :, :]
        for arr_i in range(batch.shape[0]):
            arr = batch[arr_i, :]
            temp_list = [(flip_labels[str(i)], a) for i, a in
                         enumerate(arr)]
            new_list.append(temp_list)
    return new_list

if __name__ == "__main__":
    import consts as ct
    from audioclip import AudioClip
    print(flatten_strlist(["hi", "there"]))
    #ac = AudioClip(ct.DATA_DIR + "vowels/e2/e2_1.wav")
    #mfccs = ac.mfcc(bell=True)
    #plot_spec(mfccs, ac.sr)

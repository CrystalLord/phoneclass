import librosa
import librosa.display
import numpy as np
import time
import datetime

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

def combtraining(audioclips):
    """Take in a list of audioclips and merge their training data."""
    batches_x = []
    batches_y = []

    for i, ac in enumerate(audioclips):
        train_x, train_y = ac.batch()
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

def tstamp():
    """Generate a unique timestamp"""
    now = datetime.datetime.now()
    t = now.timetuple()
    return str(t[0]) + str(t[1]) + str(t[2]) \
        + str(t[3]) + str(t[4]) + str(t[5])

if __name__ == "__main__":
    import consts as ct
    from audioclip import AudioClip
    ac = AudioClip(ct.DATA_DIR + "vowels/e2/e2_1.wav")
    mfccs = ac.mfcc(bell=True)
    plot_spec(mfccs, ac.sr)

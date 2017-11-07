import librosa
import librosa.display
import numpy as np

def dbspec(spec):
    """Convert spectrogram data from power to decibels"""
    return librosa.power_to_db(spec)

def padwave(waveform, sr, mintime):
    """Pads waveform data with silence at the end to reach minimum length.
    """
    new_waveform = waveform
    minlen = round(mintime * sr)
    diff = minlen - waveform.size
    if diff > 0:
        print("PadWave: Padding to " + str(mintime) + "s")
        diff_wave = np.zeros(diff)
        new_waveform = np.concatenate((waveform, diff_wave))
    return new_waveform


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

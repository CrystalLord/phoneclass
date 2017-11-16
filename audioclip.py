import librosa
import librosa.feature as feature
import numpy as np

import utils
import consts as ct

class AudioClip:
    def __init__(self, fp, name=None, sr=16000, start=0, end=None, mintime=0):
        self.fp = fp
        if name is None:
            self.name = self.fp
        else:
            self.name = name
        # Retrieve audio data.
        if end is None:
            self.samples, self.sr = librosa.load(
                    fp,
                    sr=sr,
                    mono=True,
                    offset=start)
        else:
            self.samples, self.sr = librosa.load(
                    fp,
                    sr=sr,
                    mono=True,
                    offset=start,
                    duration=(end-start))
        print("AudioClip--Retrieved Audio.")
        print("AudioClip--length: " + str(self.samples.size/self.sr))
        print("AudioClip--samples: " + str(self.samples.size))

        if mintime > 0:
            self.samples = utils.cutwave(self.samples, self.sr, mintime)

        self.ipa_regions = None
        self.slices = None
        self.length = self.samples.size / self.sr
        self._ipa_full = None
        self.annotated_samples = None

    def batch(self, coeff_count=13):
        """Output a single batch of training data from this AudioClip.
        We make the assumption that the user has annotated the data prior.
        In the process, will compute the MFCC data, the deltas, and the


        Keyword Arguments
        coeff_count -- Number of Mel-Frequency Cepstrum Coefficients to use.
        """
        mfccs = self.mfcc(coeff_count)
        delta1, delta2 = self.delta_coeffs(mfccs)
        self._annotate(mfccs)

        mfccs_len = mfccs.shape[1]
        #full_batch = [None] * mfccs_len
        batch_x = np.concatenate((mfccs, delta1, delta2), axis=0).transpose()
        batch_y = np.array(self.annotated_samples)
        print("AudioClip--Generated Batch")
        return (batch_x, batch_y)

    def bell_batch(self, coeff_count=13):
        """Output a single batch of training data from this AudioClip using
        the bell trigger system

        Keyword Arguments
        coeff_count -- Number of Mel-Frequency Cepstrum Coefficients to use.
        """
        mfccs = self.mfcc(coeff_count, bell=True)
        delta1, delta2 = self.delta_coeffs(mfccs)
        self._annotate(mfccs)

        mfccs_len = mfccs.shape[1]
        #full_batch = [None] * mfccs_len
        batch_x = np.concatenate((mfccs, delta1, delta2), axis=0).transpose()
        batch_y = np.array(self.annotated_samples)
        print("AudioClip--Generated Batch")
        return (batch_x, batch_y)

    def region_setup(self, slices, ipa_regions):
        """Setup slices and ipa_regions. Deprecated."""
        self.ipa_regions = ipa_regions
        self.slices = slices

    def rsetup(self, pattern):
        """Setup slices and ipa_regions using a pattern of the format
        [time, ipa, time, ipa, time, ipa]
        e.g.
        [0, '0', 0.2 'i', 0.3, '0']
        """
        slices = []
        ipa_regions = []
        for i, p in enumerate(pattern):
            if i % 2 == 0:
                slices.append(p)
            else:
                ipa_regions.append(p)
        self.region_setup(slices, ipa_regions)

    def _annotate(self, mfccs):
        """ Internal annotation call.
        Must be called after region_setup.

        Arguments
        mfccs -- MFCC spectrogram 2D numpy array.
        """
        if self.slices is None or self.ipa_regions is None:
            raise ValueError("No IPA regions. Call setup_regions() prior")

        # Define some short hands
        slices = self.slices
        ipa_regions = self.ipa_regions

        annotation_regions = []
        mfcc_len = mfccs.shape[1]

        #ipavect_len = len(ct.IPA_MAP["sil"])

        sample_ann = [None] * mfcc_len
        #sample_ann = np.zeros((mfcc_len, ipavect_len))

        # Convert slices into sample points.
        for s in slices:
            mfcc_rate = mfcc_len / self.length
            annotation_regions.append(round(s * mfcc_rate))
        annotation_regions.append(mfcc_len)

        # Loop through the annotation regions, and set them.
        for i in range(len(annotation_regions) - 1):
            low = annotation_regions[i]
            high = annotation_regions[i+1]
            for sample_ind in range(low, high):
                if sample_ind >= len(sample_ann):
                    print(sample_ind)
                    print(len(sample_ann))
                sample_ann[sample_ind] = ct.IPA_MAP[ipa_regions[i]]
        self.annotated_samples = sample_ann

    def end_with_ipa(self, ipa):
        """Annotate this whole AudioClip file with a list of found IPAs"""
        self._ipa_full = ipa

    def append_bell(self, wave, pre_silence_length=0.5, post_silence_length=1):
        """Append a 'bell' to the given sample data.

        Arguments
        wave -- 1D numpy array representing a waveform.

        Keyword Arguments
        pre_silence_length -- The length of silence to prepend in seconds.
        post_silence_length -- The length of the silence to append in seconds.

        Returns a new set of samples with the appended bell and silences.
        """
        bell_samps, bell_sr = librosa.load(
                    ct.BELL_FILE,
                    sr=self.sr,
                    mono=True)
        bell_length = bell_samps / bell_sr

        pre_silence = np.zeros(round(pre_silence_length * self.sr))
        post_silence = np.zeros(round(post_silence_length * self.sr))
        new_wave = np.concatenate([wave,
                                   pre_silence,
                                   bell_samps,
                                   post_silence])
        return new_wave

    def mfcc(self, coeff_count=13, bell=False):
        """Retrieve a 2D numpy array of decibel MFCC coefficients given an
        audio file.

        fp -- File path of audio to read
        sr -- Sample rate to read in the audio with.
        """
        samps = self.samples
        if bell:
            samps = self.append_bell(samps)

        mfcc_seq = feature.mfcc(samps, sr=self.sr, n_mfcc=coeff_count)
        print("AudioClip--MFCC Sequence Generated")
        return mfcc_seq

    def delta_coeffs(self, mfcc_seq):
        """Retrieve the delta and delta-delta of the MFCC cofficients"""

        length = mfcc_seq.shape[1]
        deltas = np.empty(mfcc_seq.shape)
        deltas2 = np.empty(mfcc_seq.shape)

        for (coeff, samp), value in np.ndenumerate(deltas):
            if samp == 0 or samp == length-1:
                deltas[coeff, samp] = 0
                deltas2[coeff, samp] = 0
            else:
                deltas[coeff, samp] = mfcc_seq[coeff, samp+1] \
                        - mfcc_seq[coeff, samp-1]
                deltas2[coeff, samp] = 0.5*mfcc_seq[coeff, samp+1] \
                        - 2*mfcc_seq[coeff, samp] \
                        + 0.5*mfcc_seq[coeff, samp-1]
        print("AudioClip--Deltas Retrieved")
        return (deltas, deltas2)


if __name__ == "__main__":
    # Some debug code. Should not be run normally.
    ac = AudioClip("/mnt/tower_1tb/music/test.wav", mintime=3.0)
    ac.region_setup([4,6], ["sil","a1", "sil"])
    print(ac.batch()[1])


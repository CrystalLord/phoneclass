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
        self._ipa_full_vector = None
        self.annotated_samples = None

    def batch(self, coeff_count=13, db=False):
        """Output a single batch of training data from this AudioClip.
        We make the assumption that the user has annotated the data prior.
        In the process, will compute the MFCC data, the deltas, and the


        Keyword Arguments
        coeff_count -- Number of Mel-Frequency Cepstrum Coefficients to use.
        db -- Use decibel magnitude instead of absolute.
        """
        mfccs, _ = self.mfcc(coeff_count)
        if db:
            mfccs = utils.dbspec(mfccs)
        delta1, delta2 = self.delta_coeffs(mfccs)
        self._annotate(mfccs)

        mfccs_len = mfccs.shape[1]
        batch_x = np.concatenate((mfccs, delta1, delta2), axis=0).transpose()
        batch_y = np.array(self.annotated_samples)
        print("AudioClip--Generated Batch")
        return (batch_x, batch_y)

    def bell_batch(self, coeff_count=13, db=False):
        """Output a single batch of training data from this AudioClip using
        the bell trigger system

        Keyword Arguments
        coeff_count -- Number of Mel-Frequency Cepstrum Coefficients to use.
        db -- Use decibel magnitude instead of absolute.
        """
        # Retrieve MFCC sequence, the time at which the bell occurs, and the
        # new length of the appended sequence in seconds.
        mfccs, mfcc_sr, bell_time = self.mfcc(coeff_count, bell=True)
        if db:
            mfccs = utils.dbspec(mfccs)
        delta1, delta2 = self.delta_coeffs(mfccs)
        self._bell_annotate(mfccs, round(mfcc_sr * bell_time))


        mfccs_len = mfccs.shape[1]
        batch_x = np.concatenate((mfccs, delta1, delta2), axis=0).transpose()
        batch_y = np.array(self.annotated_samples)
        print("AudioClip--Generated Batch")
        return (batch_x, batch_y)

    def raw_x_batch(self, coeff_count=13):
        """Retrieve only the Neural Network input batch data. No Annotations.
        """
        mfccs, mfcc_sr = self.mfcc(coeff_count)
        delta1, delta2 = self.delta_coeffs(mfccs)
        batch_x = np.concatenate((mfccs, delta1, delta2), axis=0).transpose()
        return batch_x

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

    def _bell_annotate(self, mfccs, bell_start_sample):
        """Modify this AudioClip to set its annotated samples for bell.
        Each time step of the MFCC sequence is annotated with a vector.
        If the time step exists before the
        """
        if self._ipa_full_vector is None:
            raise ValueError("Full ipa not set. Call end_with_ipa() prior.")
        mfcc_len = mfccs.shape[1]
        sample_ann = [[0]*len(self._ipa_full_vector)] * mfcc_len

        for i in range(bell_start_sample, mfcc_len):
            sample_ann[i] = self._ipa_full_vector
        self.annotated_samples = sample_ann

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
        sample_ann = [None] * mfcc_len

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

    def end_with_ipa_vector(self, ipa):
        """Annotate this whole AudioClip file with a list of found IPAs"""
        self._ipa_full_vector = ipa

    def append_bell(self, wave, pre_silence_length=0.5, post_silence_length=1):
        """Append a 'bell' to the given sample data.

        Arguments
        wave -- 1D numpy array representing a waveform.

        Keyword Arguments
        pre_silence_length -- The length of silence to prepend in seconds.
        post_silence_length -- The length of the silence to append in seconds.

        Returns a new set of samples with the appended bell and silences.
        """
        # Retrieve bell audio data.
        bell_samps, bell_sr = librosa.load(
                    ct.BELL_FILE,
                    sr=self.sr,
                    mono=True)
        # Length of the bell sound, in seconds.
        bell_length = bell_samps.shape[0] / bell_sr

        # Silence before the bell triggers.
        pre_silence = np.zeros(round(pre_silence_length * self.sr))
        # Silence after the bell triggers.
        post_silence = np.zeros(round(post_silence_length * self.sr))
        new_wave = np.concatenate([wave,
                                   pre_silence,
                                   bell_samps,
                                   post_silence])
        bell_time = wave.shape[0] / self.sr + pre_silence_length + bell_length
        new_time = wave.shape[0] / self.sr + pre_silence_length + bell_length \
            + post_silence_length
        return new_wave, bell_time, new_time

    def mfcc(self, coeff_count=13, bell=False):
        """Retrieve a 2D numpy array of decibel MFCC coefficients given an
        audio file.

        fp -- File path of audio to read
        sr -- Sample rate to read in the audio with.

        If bell is not set, returns only the mfcc sequence.
        If bell is set, returns a tuple of (mffcc seq, bell_time)
        """
        samps = self.samples
        sound_length = self.length
        if bell:
            # Update the samples, the bell_time, and the length of
            # sound.
            samps, bell_time, sound_length = self.append_bell(samps)

        mfcc_seq = feature.mfcc(samps, sr=self.sr, n_mfcc=coeff_count)
        mfcc_sr = mfcc_seq.shape[1] / sound_length
        print("AudioClip--MFCC Sequence Generated")
        if bell:
            return mfcc_seq, mfcc_sr, bell_time
        return mfcc_seq, mfcc_sr

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


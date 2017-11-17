from one_hotify import one_hotify

DATA_DIR = "/mnt/tower_1tb/neural_networks/audio_train/"
MODELDIR = "/mnt/tower_1tb/neural_networks/models/"
PREDDIR = "/mnt/tower_1tb/neural_networks/prediction/"
TRAINDIR = "/mnt/tower_1tb/neural_networks/audio_train/"
BELL_FILE = "/mnt/tower_1tb/neural_networks/misc/bell.wav"
IPAWORDFILE = "/mnt/tower_1tb/neural_networks/misc/ipa_word_map.csv"

TEST_TRANSCRIPT_FP = \
"/mnt/tower_1tb/neural_networks/ucsb_transcripts/SBC001.trn"

IPA_NUM = {
        "sl": 0,
        "a1": 1,
        "i1": 2,
        "ai": 3,
        "e2": 4}

IPA_MAP = one_hotify(IPA_NUM)

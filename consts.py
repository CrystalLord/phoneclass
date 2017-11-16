DATA_DIR = "/mnt/tower_1tb/neural_networks/audio_train/"
MODELDIR = "/mnt/tower_1tb/neural_networks/models/"
PREDDIR = "/mnt/tower_1tb/neural_networks/prediction/"
TRAINDIR = "/mnt/tower_1tb/neural_networks/audio_train/"
BELL_FILE = "/mnt/tower_1tb/neural_networks/misc/bell.wav"
IPAWORDFILE = "/mnt/tower_1tb/neural_networks/misc/ipa_word_map.csv"

IPA_NUM = {
        "sl": 0,
        "a1": 1,
        "i1": 2,
        "ai": 3,
        "e2": 4}

def one_hotify(dic):
    """Convert a dictionary's keys to a one hot vector"""
    new_dic = {}
    sym_count = max(tuple(dic.values()))+1
    for k in dic.keys():
        one_hot = [0]*sym_count
        one_hot[dic[k]] = 1
        new_dic[k] = one_hot
    return new_dic

IPA_MAP = one_hotify(IPA_NUM)

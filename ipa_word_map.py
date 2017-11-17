import os, sys
from operator import itemgetter

import dict_utils
import transparser
from wikiipa import PageGrabber
import consts as ct

transcript = "/mnt/tower_1tb/neural_networks/ucsb_transcripts/SBC001.trn"

def main():
    # Initialise PageGrabber
    pg = PageGrabber(delay=1)
    # Load the ipa dictionary from a file.
    ipa_dict = dict_utils.load_dict(ct.IPAWORDFILE)
    # Retrieve a parsed list of lines from a transcript file.
    line_list = transparser.parse_transcript(transcript)
    # Retrieve a list of words for that line.
    word_list = line_list_extract(line_list)
    try:
        for word in word_list:
            if not word in ipa_dict:
                print("\"" + word + "\"" + " not found...")
                try:
                    ipa_try = pg.pull_word(word)
                except Exception as e:
                    ipa_try = None
                if ipa_try is None:
                    ipa_try = manual_ipa()
                else:
                    print(word + "  -->  " + ipa_try)
                    i = str(input("Good? (y)> "))
                    flag = (i == "" or i == "y")
                    if not flag:
                        ipa_try = manual_ipa()
                ipa_dict[word] = ipa_try
            else:
                print(word + " found.")
    except KeyboardInterrupt:
        pass
    finally:
        dict_utils.write_dict(ct.IPAWORDFILE, ipa_dict)

def line_list_extract(line_list):
    """Extract words from a line list"""
    nested_words = [line[2].split(" ") for line in line_list]
    word_list = []
    for nest in nested_words:
        for word in nest:
            word_list.append(word)
    return word_list


def manual_ipa():
    return input("Manual> ")




if __name__ == "__main__":
    main()

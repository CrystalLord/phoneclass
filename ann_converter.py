import numpy as np

import utils
import dict_utils
import consts as ct
from one_hotify import one_hotify

TO_REMOVE = ['ː', ".", "ˈ", "(", ")", "ˌ", "\u0329", "\u0361", "\u032F"]
TO_MERGE = {"\u028D":"w", "ɑ":"a", "ə":"ʌ"}

def convert(line_list):
    """Convert a list of annotations that use English words to phonetic,
    set-based annotations.

    Input: [(start, end, text_in_english), ... ]
    Output: [(start, end, text_in_IPA), ... ]
    """
    converted_list = []
    for line in line_list:
        english_text = line[2]
        english_list = english_text.split()

        english_to_ipa = dict_utils.load_dict(ct.IPAWORDFILE)

        # Convert to phonemes.
        ipa_list = [english_to_ipa[word] for word in english_list]
        # Remove word seperations with phonemes.
        ipa_list = utils.flatten_strlist(ipa_list)
        # Clean the list
        cleaned_ipa_list = clean_ipa_list(ipa_list)

        new_tuple = (line[0], line[1], cleaned_ipa_list)
        converted_list.append(new_tuple)
    return converted_list

def category_convert(line_list):
    """Convert a list of tuples of the format (start, end, english_text) to IPA

    The conversion is as follows:
    [(start, end, english), ... ] --> [(start, end, categorical vector), ... ]

    Arguments
    line_list -- List of tuples representing a stripped transcription line.

    Returns
    position 1 -- A dictionary of each phoneme mapped to its index.
    position 2 -- A list of tuples of the form above.
    """
    # Get the IPA of the lines.
    ipa_lines = convert(line_list)

    # Find the one hot vector notation for each IPA symbol.
    found_num = {}
    count = 0
    for line in ipa_lines:
        for sym in line[2]:
            if sym not in found_num:
                found_num[sym] = count
                count += 1
    categories = one_hotify(found_num)

    # Calculate the IPA vector sum for each line, and append
    # to the new list.
    ipa_vector_lines = []
    for line in ipa_lines:
        summed_vector = []
        for i, sym in enumerate(line[2]):
            sym_v = categories[sym]
            if i == 0:
                summed_vector = sym_v
            else:
                summed_vector = utils.sum_vectors(summed_vector, sym_v)
        new_line = (line[0], line[1], summed_vector)
        ipa_vector_lines.append(new_line)
    return found_num, ipa_vector_lines


def clean_ipa_list(l):
    """Clean a list of IPA symbols."""
    new_l = merge_syms(l)
    new_l = removedups(new_l)
    new_l = remove_unwanted(new_l)
    return new_l

def remove_unwanted(l):
    """Remove unwanted IPA symbols"""
    new_l = []
    for i in l:
        if i not in TO_REMOVE:
            new_l.append(i)
    return new_l

def merge_syms(l):
    """Take in a list of IPA symbols (strings), and convert some symbols
    into simplified versions
    """
    new_l = []
    for i in l:
        if i in TO_MERGE:
            new_l.append(TO_MERGE[i])
        else:
            new_l.append(i)
    return new_l

def removedups(l):
    """Remove duplicates from a list"""
    found = {}
    for i in l:
        if not i in found:
            found[i] = True
    return list(found.keys())

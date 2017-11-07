#!/usr/bin/env python3

import re

WEIRD_CHARS_REG1 = "[(..)X\@\+\&\^\%<>\[\]\$\*\:\-]+"
WEIRD_CHARS_REG2 = "[0-9]+"

FILES_TO_READ = ["/home/crystal/Desktop/TRN/SBC001.trn"]

def main():
    line_list = []
    for fp in FILES_TO_READ:
        with open(fp, 'r') as f:
            for line in f:
                items = line.split("\t")
                t_low, t_high = items[0].split(" ")
                if len(items) == 3:
                    text = items[2]
                elif len(items):
                    text = items[1]
                else:
                    raise ValueError("")
                if not contains_weird(text):
                    text = text_clean(text)
                    text = text.lower()
                    line_list.append((float(t_low), float(t_high), text))
    print(line_list)

def contains_weird(text):
    """Does the given text contain any bizarre, abnormal characters?"""
    pattern1 = re.compile(WEIRD_CHARS_REG1)
    pattern2 = re.compile(WEIRD_CHARS_REG2)
    return (pattern1.search(text) is not None
        or pattern2.search(text) is not None)

def text_clean(text):
    """Clean the text."""
    t = ""
    for c in text:
        if re.match("[a-zA-Z]| ", c):
          t += c
    t = t.strip()
    return t


#def bracket_indices(text, b_open="<", b_close=">"):
#    if text == "":
#        return []
#
#    open_count = 0
#    starting_index = -1
#    ending_index = -1
#
#    region_list = []
#
#    for i, c in enumerate(text):
#        if c == b_open:
#            if open_count == 0:
#                starting_index = i
#            open_count += 1
#        elif c == b_close:
#            open_count -= 1
#            if open_count == 0:
#                ending_index = i+1
#                region_list.append((starting_index, ending_index))
#
#    return region_list
#
#def char_index(text, chars):
#    """Return a list of indices where a character appears in text"""
#    l = []
#    for i in range(len(text)):
#        if text[i] in chars:
#            l.append(i)
#    return l
#
#def cut_regions(text, regions):
#    new_text = ""
#    for i, c in enumerate(text):
#        can_use = True
#        for r in regions:
#            if isinstance(r, tuple):
#                # r is a range of indices
#                if i >= r[0] and i < r[1]:
#                    can_use = False
#            else:
#                # r must be a single number.
#                if i == r:
#                    can_use = False
#        if can_use:
#            new_text += c
#    return new_text
#
#
#def weird_count(regions):
#    count = 0
#    for r in regions:
#        if isinstance(r, tuple):
#            count += r[1] - r[0]
#        else:
#            count += 1
#    return count
#
#def weird_percent(text, regions):
#    return len(text)/weird_count(regions)

if __name__ == "__main__":
    main()
    #text = "<Hello> yes <hi!>"
    #print(stripper(text))

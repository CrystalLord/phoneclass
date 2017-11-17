"""Utilities for reading and writing IPA dictionaries."""

import os

def load_dict(fp):
    """Load up the IPA dictionary from a file.
    Arguments
    fp -- Filepath of dictionary to look up.

    Returns a dictionary read from the given file.
    """
    dict = {}

    if not os.path.isfile(fp):
        print("No dict file found")
        return dict

    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line != "" and line != "\n":
                elems = line.split(",")
                dict[elems[0]] = elems[1]
    return dict


def write_dict(fp, dict, overwrite=False):
    """Write the given dictionary to a file

    Arguments
    fp -- Filepath to write dictionary to.
    dict -- Dictionary to write.

    Keyword Arguments
    overwrite -- Boolean Force overwrites without asking?
    """
    if os.path.isfile(fp) and not overwrite:
        print("File exists: " + fp)
        flag = (str(input("Overwrite? (y|n)> ")).lower() == "y")
        if not flag:
            print("Exiting without writing")
            return
    tups_list = []
    for k in dict:
        tups_list.append((k, dict[k]))
    print(tups_list)
    tups_list = sorted(tups_list, key=(lambda x: x[0]))
    with open(fp, 'w') as f:
        for t in tups_list:
            f.write(str(t[0]) + "," + str(t[1]) + "\n")
    print("Dictionary written to " + fp)

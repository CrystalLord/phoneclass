#!/usr/bin/env python3

import argparse
import numpy as np

import ann_converter
import transparser
import consts as ct
from audioclip import AudioClip
import utils
import sampleplot
import dict_utils

np.set_printoptions(threshold=np.nan)

DEBUG = False
TRAIN_NAME ="training.npy"
TARGET_NAME = "target.npy"
LABEL_NAME = "labels.csv"

def main():
    output_fp_x = None
    output_fp_y = None
    if DEBUG:
        audio_file = "/mnt/tower_1tb/neural_networks/sbc_audio/SBC001.wav"
        trans = "/mnt/tower_1tb/neural_networks/sbc_transcripts/SBC001.trn"
        num = 0
        outdir = utils.slashend("/mnt/tower_1tb/neural_networks/training_data")
    else:
        args = parse_args()
        # Retrieve argument data.
        audio_file = args.audio_file
        trans = args.transcript
        outdir = args.outdir
        num = 0
        #output_fp_x = args.outx
        #output_fp_y = args.outy
        #num = args.num

    # Get training data
    labels, train_x, train_y = get_training(audio_file, trans, num,
                                            args.decibel)

    # Save the data
    if outdir is None:
        outdir = utils.slashend(input("Training directory save filepath> "))

    np.save(outdir + TRAIN_NAME, train_x)
    np.save(outdir + TARGET_NAME, train_y)
    dict_utils.write_dict(utils.slashend(outdir) + LABEL_NAME, labels)


def get_training(audio_file, transcript, num=0, db=False):
    """Retrieve training data from an audio file."""
    # Parse the transcript
    if num == 0:
        parsed = transparser.parse_transcript(transcript)
    else:
        parsed = transparser.parse_transcript(transcript)[:num]
    labels, line_annotations = ann_converter.category_convert(parsed)
    print(labels)
    # Apply annotations, and return
    training_x, training_y = apply_annotations(audio_file,
                                               line_annotations,
                                               db=db)
    return labels, training_x, training_y

def apply_annotations(audio_file, line_annotations, db=False):
    """Apply line annotations to AudioClips, and batch them"""
    audioclips = []
    for line in line_annotations:
        # Get the start and end of each clip.
        start = line[0]
        end = line[1]
        vector = line[2]

        ac = AudioClip(audio_file, start=start, end=end, mintime=3)
        ac.end_with_ipa_vector(vector)
        audioclips.append(ac)
    return utils.combtraining(audioclips, use_bell=True, db=db)

def parse_args():
    """Parse arguments passed to the program"""
    parser = argparse.ArgumentParser(prog="bell_training_gen")
    parser.add_argument("audio_file",
                        metavar="AUDIO",
                        help="Audio file to annotate and train with")
    parser.add_argument("transcript",
                        metavar="TRANS",
                        help="Transcript to generate training data on")
    parser.add_argument("-o",
                        "--outdir",
                        metavar="OUTPUT_DIR",
                        default=None,
                        help="Training data directory output filepath")
    parser.add_argument("-d",
                        "--decibel",
                        action='store_true',
                        help="Compute training data with decibel power mag.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

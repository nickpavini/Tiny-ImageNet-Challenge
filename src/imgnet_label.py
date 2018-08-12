"""
    Programs related to labels encoding/decoding and overall label processing.
"""

import imgnet_constants
import numpy as np
import os

# Parameters: Dir of Challenge data and photo label (n12345678) as specified in README.md..
# Converts a label to a onehot encoded array based on position in wnids.txt file
def onehot_encode(dir, label):
    onehot_label = np.zeros((imgnet_constants.CLASSIFICATIONS), dtype=np.float32)
    ids = [f.split()[0].lower() for f in open(os.path.join(dir, 'wnids.txt'), 'r').readlines()] # get possible id classifications in order
    onehot_label[ids.index(label.lower())] = 1.0 # set the index where we found the label as a 1.0 for 100% match
    return onehot_label

# Parameters: Dir of Challenge data, and onehot encoded array of label
# Returns label (n12345678) based on positioning in wnids.txt
def onehot_decode(dir, onehot_array):
    ids = [f.split()[0] for f in open(os.path.join(dir, 'wnids.txt'), 'r').readlines()]
    return ids[np.argmax(onehot_array)].lower()

#convert label (n12345678) to a list of describing words
def label_description(dir, label):
    for line in open(os.path.join(dir, 'words.txt')): # go thru each line of the file
        if line.split()[0].lower() == label.lower(): # if the label matches the provided label
            line = line.split()[1:] # split on white space leaving commas
            return ' '.join(line).split(', ') # little hack to get list of descriptions, as some descriptions are 2 words
    return None

#-------------------------------------------------------------------------------
"""
    Unit Tests.
"""

if __name__ == '__main__':
    None

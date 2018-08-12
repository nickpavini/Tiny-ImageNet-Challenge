"""
    File to easily access constant information regarding the Stanford Tiny
    ImageNet Challenge data accross multiple programs. May later remove this
    file if I can think of a better place for this information. We can simply
    have it in h5Merge.py and then recover this info from the data in the hdf5
    file to be used in the dataset and other areas. We will see how the imports
    all look.
"""


TRAIN_PICS = 100000
VAL_PICS = 10000
TEST_PICS = 10000
SHAPE = (64,64,3) # 64 x 64 colored images
CLASSIFICATIONS = 200

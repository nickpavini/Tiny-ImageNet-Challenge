"""
    Programs for testing aspects of this software. Mostly will be filled with
    trash code and serious unit tests will be in the original files where the
    code can be found.
"""

# imports
import h5py, os, sys
import matplotlib.pyplot as plt # for plotting images
import numpy as np
import constants, label, image


DIR = str(sys.argv[1])
FILENAME = str(sys.argv[2])
h5 = h5py.File(os.path.join(DIR, FILENAME), 'r')

"""descs = [label.label_description(DIR, label.onehot_decode(DIR, f))[0] for f in h5['train_labels'][:25]]
print(descs)
image.plot_images(5,5, h5['train_photos'][:25], descs)"""

"""print(h5['train_filenames'][0].decode("utf-8"))
print(h5['val_filenames'][0].decode("utf-8"))
print(h5['test_filenames'][0].decode("utf-8"))"""

"""print(h5['train_filenames'][10].decode("utf-8"), h5['train_filenames'][10].decode("utf-8").split('_')[0])
print(label.onehot_decode('/Users/nickpavini/datasets/tiny-imagenet-200', h5['train_labels'][10]))
print(np.argmax(h5['train_labels'][10]))
print(label.label_description('/Users/nickpavini/datasets/tiny-imagenet-200', label.onehot_decode('/Users/nickpavini/datasets/tiny-imagenet-200', h5['train_labels'][10])))
plt.imshow(h5['train_photos'][10])
plt.show()"""

"""print('Starting...')
time = datetime.datetime.now()
photos = np.zeros((constants.VAL_PICS, constants.SHAPE[0], constants.SHAPE[1], constants.SHAPE[2]), dtype=np.uint8)
for i in range(int(constants.TRAIN_PICS/20)):
    photos[i:i+19] = h5['train_photos'][i:i+19] #get 20 photos
print('Done', datetime.datetime.now() - time)"""

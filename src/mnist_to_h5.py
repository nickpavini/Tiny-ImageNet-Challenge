import tensorflow as tf
import numpy as np
import sys, h5py

file = h5py.File(str(sys.argv[1]), 'w')

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_data = np.reshape(train_data, (train_data.shape[0], 28, 28, 1)) # reshape to 28 x 28 x 1 images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
train_labels_onehot = np.zeros((train_labels.shape[0], 10), dtype=np.float32) # for onehot encoding array
filenames = np.empty((train_labels.shape[0]), dtype='S25')
for i in range(len(train_labels)):
    train_labels_onehot[i][train_labels[i]] = 1

eval_data = mnist.test.images # Returns np.array
eval_data = np.reshape(eval_data, (eval_data.shape[0], 28, 28, 1))
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
eval_labels_onehot = np.zeros((eval_labels.shape[0], 10), dtype=np.float32)
for i in range(len(eval_labels)):
    eval_labels_onehot[i][eval_labels[i]] = 1

print(train_data.shape, train_labels_onehot.shape, eval_data.shape, eval_labels_onehot.shape)

file.create_dataset('train_photos', data=train_data, compression='lzf')
file.create_dataset('train_labels', data=train_labels_onehot, compression='lzf')
file.create_dataset('train_filenames', data=filenames, compression='lzf')

file.create_dataset('val_photos', data=eval_data, compression='lzf')
file.create_dataset('val_labels', data=eval_labels_onehot, compression='lzf')
file.create_dataset('val_filenames', data=filenames, compression='lzf')

file.create_dataset('test_photos', data=eval_data, compression='lzf')
file.create_dataset('test_labels', data=eval_labels_onehot, compression='lzf')
file.create_dataset('test_filenames', data=filenames, compression='lzf')

file.close()

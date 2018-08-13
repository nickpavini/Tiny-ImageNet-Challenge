"""
    Class to that manages the Tiny ImageNet Challenge dataset from an hdf5
    formatted as specified in README.md.
"""
import h5py, os # hdf5 handling, path management
import sys # for unit test command arguments
import imgnet_constants as consts
import numpy as np # array management
import math # for rounding steps up
from tqdm import tqdm # for unit tests
from imgnet_utils import unsisonShuffle

class Dataset:
    def __init__(self, hdf5_file, batch_size, chunk_size):
        # assertions
        assert (os.path.exists(hdf5_file)), ('Path to hdf5 file does not exist.') # make sure the provided hdf5_file exists
        assert (isinstance(batch_size, int) and batch_size > 0), ('Batch size must be a positve integer') # make sure batch_size is a pos integer
        assert (isinstance(batch_size, int) and chunk_size > 0 and chunk_size <= consts.TRAIN_PICS and consts.TRAIN_PICS%chunk_size == 0), ('Chunk size out of range of total training images or does not divide evenly.')

        # assignments
        self.hdf5_file = h5py.File(hdf5_file, 'r') # handle to hdf5 file
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        # get steps per epoch for train/val/test sets
        self.train_steps = int(math.ceil(consts.TRAIN_PICS / batch_size))
        self.val_steps = int(math.ceil(consts.VAL_PICS / batch_size))
        self.test_steps = int(math.ceil(consts.TEST_PICS / batch_size))

        # initialize variables to keep track of number of photos processed for each set
        self.train_photos_processed = 0
        self.val_photos_processed = 0
        self.test_photos_processed = 0

        # setup chunking setttings
        self.current_chunk = 0 # start on chunk 0
        self.total_chunks = int(consts.TRAIN_PICS/chunk_size)
        self.train_photos_loaded = False # start without loading training chunk

    # load chunk_size photos/labels/filenames into ram
    def load_train_photos(self, start, finish):
        assert (finish-start == self.chunk_size), 'Difference does not equal chunk_size.'
        self.permutation = np.random.permutation(finish-start) #create permutation
        self.train_photos = self.hdf5_file['train_photos'][start:finish] # take chunk_size photos
        self.train_labels = self.hdf5_file['train_labels'][start:finish] # take chunk_size photos
        self.train_filenames= self.hdf5_file['train_filenames'][start:finish] # take chunk_size photos
        self.train_photos, self.train_labels, self.train_filenames = unsisonShuffle(self.train_photos, self.train_labels, self.train_filenames, self.permutation)


    # get next batch_size train images, labels and filenames... training data is always shuffled between epochs so we must take from shuffled indices
    # need to speed up while still randomizing data between epochs
    def next_train_batch(self):
        flag = False # flag for if we are taking the last of the photos to be processed
        batch_size = self.batch_size

        if (self.train_photos_loaded == False): # if we have not loaded the a chunk, then load next chunk
            self.load_train_photos(int(self.chunk_size*self.current_chunk), int(self.chunk_size*self.current_chunk + self.chunk_size)) # load next chunk_size images/labels/filenames
            self.current_chunk += 1 # increment total chunks
            if (self.current_chunk == self.total_chunks): # if we have done the total amount of chunks then reset
                self.current_chunk = 0
            self.train_photos_loaded = True # set training photos chunk to loaded

        # set batch_size to correct value based on number of photos left to process
        if (self.chunk_size - self.train_photos_processed) < batch_size:
            flag = True
            batch_size = self.chunk_size % batch_size # find how many photos are left to process

        # get photo labels and filenames
        photos = self.train_photos[self.train_photos_processed:int(self.train_photos_processed + batch_size)] # get next batch_size photos
        labels = self.train_labels[self.train_photos_processed:int(self.train_photos_processed + batch_size)] # get next batch_size labels
        filenames = self.train_filenames[self.train_photos_processed:int(self.train_photos_processed + batch_size)] # get next batch_size filenames

        if flag: # start from begginning and shuffle
            self.train_photos_processed = 0
            self.train_photos_loaded = False
        else:
            self.train_photos_processed += batch_size

        return photos, labels, filenames

    # get next batch_size val images, labels and filenames
    def next_val_batch(self):
        flag = False # flag for if we are taking the last of the photos to be processed
        batch_size = self.batch_size

        # set batch_size to correct value based on number of photos left to process
        if (consts.VAL_PICS - self.val_photos_processed) < batch_size:
            flag = True
            batch_size = consts.VAL_PICS % batch_size # find how many photos are left to process

        # get photo labels and filenames
        photos = self.hdf5_file['val_photos'][self.val_photos_processed:self.val_photos_processed + batch_size] # get next batch_size photos
        labels = self.hdf5_file['val_labels'][self.val_photos_processed:self.val_photos_processed + batch_size] # get next batch_size labels
        filenames = self.hdf5_file['val_filenames'][self.val_photos_processed:self.val_photos_processed + batch_size] # get next batch_size filenames

        if flag:
            self.val_photos_processed = 0
        else:
            self.val_photos_processed += batch_size

        return photos, labels, filenames

    # get next batch_size train images and filenames
    def next_test_batch(self):
        flag = False # flag for if we are taking the last of the photos to be processed
        batch_size = self.batch_size

        # set batch_size to correct value based on number of photos left to process
        if (consts.TEST_PICS - self.test_photos_processed) < batch_size:
            flag = True
            batch_size = consts.TEST_PICS % batch_size # find how many photos are left to process

        # get photo labels and filenames
        photos = self.hdf5_file['test_photos'][self.test_photos_processed:self.test_photos_processed + batch_size] # get next batch_size photos
        filenames = self.hdf5_file['test_filenames'][self.test_photos_processed:self.test_photos_processed + batch_size] # get next batch_size filenames

        if flag:
            self.test_photos_processed = 0
        else:
            self.test_photos_processed += batch_size

        return photos, filenames

#-------------------------------------------------------------------------------
"""
    Unit Tests.
"""

if __name__ == '__main__':
    dataset = Dataset(str(sys.argv[1]), int(sys.argv[2]), consts.TRAIN_PICS/10)

    for i in tqdm(range(dataset.train_steps), desc='Train batches'):
        photos, labels, filenames = dataset.next_train_batch()

    for i in tqdm(range(dataset.val_steps), desc='Validation batches'):
        photos, labels, filenames = dataset.next_val_batch()

    for i in tqdm(range(dataset.test_steps), desc='Test batches'):
        photos, labels = dataset.next_test_batch()

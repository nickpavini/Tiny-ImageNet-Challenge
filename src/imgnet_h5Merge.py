"""
    Merge Stanford Tiny ImageNet Challenge photos from jpegs to an hdf5 file
    formatted as specified in README.md.

    - dirToH5:  Provide the directory (formatted as specified in README.md) to
                the Stanford Tiny ImageNet Challenge data, as well as a
                filename for the new hdf5 file to be created within the
                provided directory.

    - dloadToH5:    Have the photos downloaded from
                    http://cs231n.stanford.edu/tiny-imagenet-200.zip and saved
                    in the provided directory with the provided filename.
"""

# imports
import requests, zipfile, io # for downloading zip if need be
import os # directory and file path management
import h5py # for handling hdf5 files
import matplotlib.image as mpimg # for reading images as numpy arrays
import numpy as np # array management
import imgnet_label, imgnet_image, imgnet_constants # Encode labels, image formatting and normalization, constant information about the provided dataset
import random # shuffle data
import sys # command arguments for unit tests
from tqdm import tqdm

# shuffle 2 or 3 numpy arrays in parallel
def unsisonShuffle(a, b, c = None):
    p = np.random.permutation(len(a)) # create permutation

    # swap ith variable with p[i]th permutation
    if c is None:
        assert (len(a) == len(b)), (len(a), len(b))
        for i in tqdm(range(len(a)), desc='Shuffling in unison'):
            temp = [a[i], b[i]]
            a[i], b[i] = a[p[i]], b[p[i]]
            a[p[i]], b[p[i]] = temp[0], temp[1]
        return a, b
    else:
        assert (len(a) == len(b) and len(a) == len(c)), (len(a), len(b), len(c))
        for i in tqdm(range(len(a)), desc='Shuffling in unison'):
            temp = [a[i], b[i], c[i]] #
            a[i], b[i], c[i] = a[p[i]], b[p[i]], c[p[i]]
            a[p[i]], b[p[i]], c[p[i]] = temp[0], temp[1], temp[2]
        return a, b, c

# Parse directory returns photos, onehot arrays for labels and filenames related to training data
def getTrainData(dir):
    train_dir = os.path.join(dir,'train') # now we have the training directory
    assert (os.path.isdir(train_dir)), 'Not an existing directory.' # assert the directory exists
    train_dirs = next(os.walk(train_dir))[1] #list of directories in the train/ directory... should be only folders representing classification codes

    # Allocate arrays to be returned
    photos = np.zeros((imgnet_constants.TRAIN_PICS, imgnet_constants.SHAPE[0], imgnet_constants.SHAPE[1], imgnet_constants.SHAPE[2]), dtype=np.float32)
    labels = np.zeros((imgnet_constants.TRAIN_PICS, imgnet_constants.CLASSIFICATIONS), dtype=np.uint8)
    filenames = np.empty((imgnet_constants.TRAIN_PICS), dtype='S25') # string of 25 characters for the filename

    i = 0 # count for which photo we are on [0,CLASSIFICATIONS)
    for id in tqdm(train_dirs, desc='Parsing training data'): #go thru each training directory... dir/train/
        id_dir = os.path.join(os.path.join(train_dir, id), 'images') # /dir/train/id/images
        for img_name in os.listdir(id_dir): # get all photo names in this id directory
            img = mpimg.imread(os.path.join(id_dir, img_name)) # /dir/train/id/images/image_name.JPEG
            photos[i] = img if img.shape == imgnet_constants.SHAPE else imgnet_image.grayToRgb(img) # get photo as array
            photos[i] = imgnet_image.normalizeImage(photos[i])
            labels[i] = imgnet_label.onehot_encode(dir, id) # onehot encoded array
            filenames[i] = img_name
            i += 1

    return photos, labels, filenames

# Parse directory returns photos, onehot arrays for labels and filenames related to validation data
def getValData(dir):
    val_dir = os.path.join(dir,'val') # now we have the validation directory, dir/val
    assert (os.path.isdir(val_dir)), 'Not an existing directory.' # assert the directory exists

    # Allocate arrays to be returned
    photos = np.zeros((imgnet_constants.VAL_PICS, imgnet_constants.SHAPE[0], imgnet_constants.SHAPE[1], imgnet_constants.SHAPE[2]), dtype=np.float32)
    labels = np.zeros((imgnet_constants.VAL_PICS, imgnet_constants.CLASSIFICATIONS), dtype=np.uint8)
    filenames = np.empty((imgnet_constants.VAL_PICS), dtype='S25') # bytes string array of 25 characters for the filename

    filenames[:] = os.listdir(os.path.join(val_dir,'images'))[:] # get filenames of photos, dir/val/images

    for i in tqdm(range(len(filenames)), desc='Parsing validation data'): # go thru all images and retrieve their matrices and labels
        #get validation photos
        img = mpimg.imread(os.path.join(os.path.join(val_dir, 'images'), filenames[i].decode("utf-8"))) # /dir/val/imgages/filenames
        photos[i] = img if img.shape == imgnet_constants.SHAPE else imgnet_image.grayToRgb(img) # get photo as array
        photos[i] = imgnet_image.normalizeImage(photos[i])

        # get validation photo labels
        val_annotations = open(os.path.join(val_dir, 'val_annotations.txt'))
        j=0 # only a counter used as a flag for if no matching label was found
        for line in val_annotations: # find photo name in file
            if line.split()[0].lower() == filenames[i].decode("utf-8").lower(): # return onehot encoded label if we find matching name
                labels[i] = imgnet_label.onehot_encode(dir, line.split()[1])
                break
            j+=1 # didnt find so increment
        assert (j != imgnet_constants.VAL_PICS), 'Error: Could not find a matching label for this photo.' # problem if photo was not labeled
    return photos, labels, filenames

# Parse directory returns photos, onehot arrays for labels and filenames related to testing data
def getTestData(dir):
    test_dir = os.path.join(dir, 'test')
    assert (os.path.isdir(test_dir)), 'Not an existing directory.' # assert the directory exists

    # Allocate arrays to be returned
    photos = np.zeros((imgnet_constants.VAL_PICS, imgnet_constants.SHAPE[0], imgnet_constants.SHAPE[1], imgnet_constants.SHAPE[2]), dtype=np.float32)
    filenames = np.empty((imgnet_constants.VAL_PICS), dtype='S25') # bytes string array of 25 characters for the filename

    filenames[:] = os.listdir(os.path.join(test_dir,'images'))[:] # get filenames of photos, dir/test/images... wont work if amount of files doesnt match # of test pics

    for i in tqdm(range(len(filenames)), desc='Parsing test data'): # [0, len(filenames))
        img = mpimg.imread(os.path.join(os.path.join(test_dir, 'images'), filenames[i].decode("utf-8")))
        photos[i] = img if img.shape == imgnet_constants.SHAPE else imgnet_image.grayToRgb(img) # get photo as array
        photos[i] = imgnet_image.normalizeImage(photos[i])

    return photos, filenames

# Parameters: /path/to/photos/ && filename to be saved
def dirToH5(dir, filename):
    assert (os.path.isdir(dir)), 'Not an existing directory.' # assert the directory exists
    h5 = h5py.File(os.path.join(dir, filename+'.h5'), 'w')

    # Populate hdf5 file as specified in README.md with training data
    photos, labels, filenames = getTrainData(dir)
    photos, labels, filenames = unsisonShuffle(photos, labels, filenames) # shuffle data... may be RAM heavy
    print('Saving training data to ' + os.path.join(dir, filename+'.h5') + '...')
    h5.create_dataset('train_photos', data=photos, compression='lzf')
    h5.create_dataset('train_labels', data=labels, compression='lzf')
    h5.create_dataset('train_filenames', data=filenames, compression='lzf')
    print('Done')

    # Populate hdf5 file as specified in README.md with validation data
    print()
    photos, labels, filenames = getValData(dir)
    photos, labels, filenames = unsisonShuffle(photos, labels, filenames) # shuffle data... may be RAM heavy
    print('Saving validation data to ' + os.path.join(dir, filename+'.h5') + '...')
    h5.create_dataset('val_photos', data=photos, compression='lzf')
    h5.create_dataset('val_labels', data=labels, compression='lzf')
    h5.create_dataset('val_filenames', data=filenames, compression='lzf')
    print('Done')

    # Populate hdf5 file as specified in README.md with testing data)
    print()
    photos, filenames = getTestData(dir)
    photos, filenames = unsisonShuffle(photos, filenames) # shuffle data... may be RAM heavy
    print('Saving testing data to ' + os.path.join(dir, filename+'.h5') + '...')
    h5.create_dataset('test_photos', data=photos, compression='lzf')
    h5.create_dataset('test_filenames', data=filenames, compression='lzf')
    print('Done')

    h5.close() # Close the file and we are done

# Parameters: /path/to/dir/ for saving file and photos && filename for saving
def dloadToH5(dir, filename):
    assert (os.path.isdir(dir)), 'Not an existing directory.' # assert the directory exists

    #extract photos to the provided dir
    print('Downloading Tiny ImageNet Challenge data...')
    tinynet_req = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    tinynet_zip = zipfile.ZipFile(io.BytesIO(tinynet_req.content))
    print('Done','\nExtracting to ' + dir + '...')
    tinynet_zip.extractall(dir)
    print('Done\n')

    dirToH5(os.path.join(dir,'tiny-imagenet-200'), filename) # Merge data to an hdf5 file to be stored in the downloaded directory

#-------------------------------------------------------------------------------
"""
    Unit tests related to merging the ImageNet challenge data to a single hdf5
    file. Functions to test are dloadToH5() and dirToH5() and unsisonShuffle as
    other outside functions will be tested in their respective files. We also
    want to assert that for every training photo that their decoded onehot
    labels should match the filename split at the '_'. Validation photo labels
    will be slightly harder to validate so for now I my thought is to randomly
    select some validation photo names and use the code in getValData() to get
    those random validation images ids and visually check that the corresponding
    files did match with their ids. Test photos do not have labels so no need to
    check.

    Command Arguments:  1) Main directory of the challenge data.
                                ex. /dir/to/challenge_data/
                        2) Name of the h5 file to be created (do not provide
                           file extension).
                                ex. filename
"""

if __name__ == '__main__':
    dir = str(sys.argv[1])
    filename = str(sys.argv[2])

    print('Test downloading and parsing directory to h5...')
    dloadToH5(dir, filename) # runs dloadToH5 and dirToH5, if no errors are thrown then we ran successfully
    print('Passed')

    # get handle on newly created file
    dir = os.path.join(dir, 'tiny-imagenet-200') # dir/tiny-imagenet-200/
    h5 = h5py.File(os.path.join(dir, filename) + '.h5', 'r') # dir/tiny-imagenet-200/filename.h5

    # assert all rgb values are in range [0, 1.0]
    print('\nAssert all photos have been normalized.')
    for photo in tqdm(h5['train_photos'], desc='Training Photos'):
        for row in photo:
            for col in row:
                assert (col[0] <= 1.0 and col[1] <= 1.0 and col[2] <= 1.0), ('Image not normalized:', col)

    for photo in tqdm(h5['val_photos'], desc='Validation Photos'):
        for row in photo:
            for col in row:
                assert (col[0] <= 1.0 and col[1] <= 1.0 and col[2] <= 1.0), ('Image not normalized:', col)

    for photo in tqdm(h5['test_photos'], desc='Test Photos'):
        for row in photo:
            for col in row:
                assert (col[0] <= 1.0 and col[1] <= 1.0 and col[2] <= 1.0), ('Image not normalized:', col)
    print('Passed')

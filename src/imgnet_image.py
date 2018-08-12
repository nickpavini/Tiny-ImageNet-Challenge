"""
    Programs related to editing images or image data representation in some
    fashion.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for reading images

# takes an image with rgb range [0.0, 1.0] and returns with range [0,255]
def unNormalizeImage(photo):
    for row in photo:
        for col in row:
            col[0] *= 255.0
            col[1] *= 255.0
            col[2] *= 255.0
    return photo

# Takes an image and converts 0-255 range to 0-1... image must be of float type
def normalizeImage(photo):
    for row in photo:
        for col in row:
            col[0] /= 255.0
            col[1] /= 255.0
            col[2] /= 255.0
    return photo

# Takes a grey scale image (n,n,1) and converts to (n,n,3)
def grayToRgb(photo):
    newPhoto = np.dstack((photo,photo))
    return np.dstack((newPhoto, photo)) # 64 x 64 x 3, where r=g=b... gray

# display grid of dimensions rows x cols with rows*cols photos
def plot_images(rows, cols, images, descs, mode):
    assert (len(images) == rows*cols and len(images) == len(descs)), ('Not equal slots to provided images.', 'len of images:', len(images), 'rows*cols:', rows*cols)
    fig=plt.figure(figsize=(14, 14)) # set overall grid window size
    for i in range(1, cols*rows +1):
        fig.add_subplot(rows, cols, i) # add a subplot to a rows x cols grid at position i
        plt.imshow(images[i-1]) # put image in that plot
        plt.title(descs[i-1]) # give that plot a title
    plt.suptitle(mode) # set main title to train, validate, or test
    # plt.show() # display entire plot until the window is closed
    plt.waitforbuttonpress() # display entire plot until a button press
    plt.close(fig) # destroy plot

#-------------------------------------------------------------------------------
"""
    Unit Tests.
"""

if __name__ == '__main__':
    None

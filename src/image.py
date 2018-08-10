"""
    Programs related to editing images or image data representation in some
    fashion.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for reading images


# Takes an image and converts 0-255 range to 0-1
def normalizeImage(photo):
    None

# Takes a grey scale image (n,n,1) and converts to (n,n,3)
def grayToRgb(photo):
    newPhoto = np.dstack((photo,photo))
    return np.dstack((newPhoto, photo)) # 64 x 64 x 3, where r=g=b... gray

# display grid of dimensions rows x cols with rows*cols photos
def plot_images(rows, cols, images, descs):
    assert (len(images) == rows*cols and len(images) == len(descs)), ('Not equal slots to provided images.', 'len of images:', len(images), 'rows*cols:', rows*cols)
    fig=plt.figure(figsize=(14, 14))
    for i in range(1, cols*rows +1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(images[i-1])
        plt.title(descs[i-1])
    plt.show()
    plt.close(fig)

#-------------------------------------------------------------------------------
"""
    Unit Tests.
"""

if __name__ == '__main__':
    None

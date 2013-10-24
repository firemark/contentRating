from scipy.ndimage.measurements import histogram
from scipy.misc import imresize
from scipy.ndimage import sobel
from matplotlib import colors
from time import clock
import numpy as np
import pylab as pl

# conts
R, G, B = ((Ellipsis, i) for i in range(3))
norm = colors.Normalize(-1, 1)


def get_lum(img):
    return img.dot([0.2126, 0.7152, 0.0722])


def get_hue(img):
    return np.arctan2(np.sqrt(3) * (img[G] - img[B]),
                      2 * img[R] - img[G] - img[B])


def get_edges(img):
    return np.hypot(sobel(img, axis=0, mode='constant'),
                    sobel(img, axis=1, mode='constant'))


def prepare_image(img, size):
    """Prepare image and return edges from thumbnail """

    img = imresize(img, size)

    if img.shape[2] == 4:  # remove alpha channel
        img = np.delete(img, 3, axis=2)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    return get_lum(img).flatten()

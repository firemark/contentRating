# -*- coding: utf-8 -*-
import numpy as np
import os
from scipy import ndimage
from skimage.filter import canny
from skimage.morphology import remove_small_objects
from config import Config
from pybrain.tools.xml.networkreader import NetworkReader
from utils import get_lum, get_hue, prepare_image
import pylab as pl

config = Config('config.ini')
dir_name = config.seek.dir_name
names = (os.path.join(dir_name, name) for name in os.listdir(dir_name))


image_names = config.load_image_names()


class Result:

    """
    Result object has shape (image) and list of probably symbols 
    """

    def __init__(self, shape, symbols):
        self.shape = shape
        self.symbols = symbols


def seek(net, filename):
    """
    Loading image and return list of Result objects
    """

    full_img = pl.imread(filename)
    #img = full_img
    img = full_img[:full_img.shape[0] // 4, :full_img.shape[1] // 4]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    img_size = float(img.shape[0] * img.shape[1])
    lum = get_lum(img)
    edges = canny(lum, sigma=config.seek.lum_sigma)

    fill = ndimage.binary_fill_holes(edges)
    fill = remove_small_objects(fill, 100)
    labels, n = ndimage.label(fill)

    for y, x in (t for t in ndimage.find_objects(labels) if t is not None):
        # where = np.argwhere(labeled == i)
        # s - start, e- end
        width = float(x.start - x.stop)
        height = float(y.start - y.stop)

        size = width * height / img_size
        ratio = width / height

        # print '%5.2f%% %5.2f' % (size * 100, ratio)

        if not (config.seek.size.in_range_close(size)
                and config.seek.ratio.in_range_close(ratio)):
            continue

        shape = img[y, x]

        prepared_img = prepare_image(shape, size=config.image.size)

        result = net.activate(prepared_img)

        yield Result(shape,
                     sorted(((image_names[i], x)
                             for i, x in enumerate(result)),
                            key=lambda a: -a[1]))


if __name__ == "__main__":

    print('Loading network...')
    net = NetworkReader.readFrom(config.network.path)

    for filename in names:

        print filename

        for result in seek(net, filename):

            symbols = result.symbols

            fig = pl.figure().add_subplot(1, 1, 1)

            fig.imshow(result.shape)
            fig.table(
                cellText=[["%.2f%%" % (n * 100) for _, n in symbols]],
                colLabels=[name for name, _ in symbols],
                loc='bottom'
            ).scale(1.5, 1.5)

            fig.set_title(filename)
            fig.yaxis.set_visible(False)
            fig.xaxis.set_visible(False)

            pl.show()

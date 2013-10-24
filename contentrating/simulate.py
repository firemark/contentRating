# encoding: utf-8

from pylab import imread, imshow, show
from utils import prepare_image
from skimage import color
from config import Config
from pybrain.tools.xml.networkreader import NetworkReader
from matplotlib.backends.backend_pdf import PdfPages
import pylab as pl
import numpy as np
import os
import sys


config = Config('config.ini')
test_dir_name = 'test/'

pdf = PdfPages('results.pdf')

cords = {
    "polsat2": (85, 65, 70, 60),
    "tvp": (87, 67, 30, 30),
    "polsat-hd": (90, 50, 60, 60),
    "puls": (115, 50, 70, 50),
    "canal+": (60, 30, 35, 35)
}

files = {
    "polsat-hd": ('1.png', '3.png', '4.png'),
    "polsat2": ('6.png', '8.png', '9.png'),
    "tvp": ('11.png', '12.png', '13.png', '14.png'),
    "puls": ( '18.png',),
    "canal+": ('010736162.png', '010741162.png',
               '010746162.png', '010751162.png')
}

image_names = config.load_image_names()

print('Loading network...')
net = NetworkReader.readFrom(config.network.path)

print('Compute...')

for tvname, names in files.items():
    # get cords
    cord = cords[tvname]
    img_cords = (slice(cord[1], cord[1] + cord[3]),
                 slice(cord[0], cord[0] + cord[2]))

    for fname in names:
        # load file
        name = test_dir_name + fname
        img = imread(name)

        # get logo
        box = img[img_cords]

        fig = pl.figure('logo_%s' % fname.split('.')[0].replace(' ', '_'))

        # create image
        prepared_img = prepare_image(box, size=config.image.size)

        # simulate
        result = net.activate(prepared_img)

        # result
        min_result = min(result)
        delta = max(result) - min_result
        result = sorted(((image_names[i], x)
                         for i, x in enumerate(result)),
                        key=lambda a: -a[1])

        fig = pl.figure()

        fig.add_subplot(1, 2, 1).imshow(img)

        fig_img = fig.add_subplot(1, 2, 2)
        fig_img.imshow(box, interpolation='nearest')
        fig_img.set_title(fname)
        fig_img.yaxis.set_visible(False)
        fig_img.xaxis.set_visible(False)
        fig_img.set_frame_on(False)

        fig_img.table(cellText=[["%.2f%%" % (n * 100) for _, n in result]],
                      colLabels=[name for name, _ in result],
                      loc='bottom')

        pdf.savefig()
        pl.close()

        # print '|'.join(str(name).center(15) for name, _ in result)
        # print '|'.join("%10.2f%%    " % (num * 100) for _, num in result)

pdf.close()

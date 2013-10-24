# -*- coding: utf-8 -*-
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from datetime import datetime
from collections import OrderedDict
from config import Config
from utils import prepare_image
from numpy.random import random_sample
from random import shuffle, random
import numpy as np
import pylab as pl


config = Config('config.ini')

# Generate backgrounds
len_color = config.data.background_colors
ran_color = range(len_color + 1)
len_color = float(len_color)


# generate backgrounds
backgrounds = [
    np.array((r, g, b), dtype=float) / len_color
    for r in ran_color
    for g in ran_color
    for b in ran_color
]

# generate dataSet
ds = ClassificationDataSet(config.network.in_size,
                           nb_classes=config.network.out_size)

# Mix images with backgrounds
print 'Generate images with backgrounds'

t = datetime.now()

numered_images = config.load_images()

len_mul = lambda a, b: a * len(b)

total_images = reduce(len_mul, (backgrounds,
                                numered_images,
                                config.data.alpha_levels,
                                config.data.noise_levels), 1)
n_image = 0

for i, img in enumerate(numered_images):

    for back in backgrounds:

        for alpha_level in config.data.alpha_levels:
            alpha = img[..., 3] * alpha_level
            # resize to every channel in the pixel
            alpha = alpha.reshape(img.shape[:2] + (1,)).repeat(3, axis=2)

            for noise_level in config.data.noise_levels:
                new_img = img[..., :3]  # remove alpha channel

                # mix with alpha
                new_img = (new_img * alpha) + (back * (1 - alpha))

                # add margins
                w, h = new_img.shape[0:2]
                margins = {}

                for mar in config.hor_margin_types:
                    margins[mar] = w * config.data.margins[mar] * random()

                for mar in config.ver_margin_types:
                    margins[mar] = h * config.data.margins[mar] * random()

                margin = lambda w, h: np.zeros((int(w), int(h), 3)) + back

                # hor
                new_img = np.vstack((
                    margin(margins['left'], h),
                    new_img,
                    margin(margins['right'], h)
                ))

                # ver
                w = new_img.shape[0]
                new_img = np.hstack((
                    margin(w, margins['top']),
                    new_img,
                    margin(w, margins['left'])
                ))

                # add noise
                noise = random_sample(new_img.shape) * noise_level

                new_img = np.fmod(new_img + noise, 1.0)

                prepared_img = prepare_image(new_img, config.image.size)

                ds.addSample(prepared_img, [i])
                n_image += 1

            print '%6.2f%%... time: %s\r' % (100.0 * n_image / total_images,
                                             datetime.now() - t),


    # ds.newSequence()
print '\r',
print "Done. Total time: %s" % (datetime.now() - t)

# Clean memory
del backgrounds

# create network
net = buildNetwork(
    config.network.in_size,
    int(np.sqrt(config.network.in_size * config.network.out_size)),
    config.network.out_size)

# net.sortModules()

print('Train...')
t = datetime.now()

ds._convertToOneOfMany()
trainer = BackpropTrainer(net, ds, verbose=True)
trainer.trainUntilConvergence(maxEpochs=config.network.epochs)

print "total time: %s" % (datetime.now() - t)

print('Saving to file %s...' % config.network.path)

NetworkWriter.writeToFile(net, config.network.path)

exit()

from ConfigParser import ConfigParser
from pylab import imread
import os


class Section:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<Section %s>" % self.name


def float_or_percent(x):
    x = x.strip()
    return float(x[:-1]) / 100.0 if x.endswith('%') else float(x)


class ValueRange:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def in_range(self, x):
        return self.end > x > self.start

    def in_range_close(self, x):
        return self.end >= x >= self.start


class Config(ConfigParser):

    hor_margin_types = ('left', 'right')
    ver_margin_types = ('top', 'bottom')

    margin_types = ver_margin_types + hor_margin_types

    def __init__(self, path):
        ConfigParser.__init__(self)
        self.read(path)
        self.path = path

        for section in ('image', 'data', 'network', 'seek'):
            setattr(self, section, Section(section))

        self.image.size = self.get_nums('image', 'size', 'x', int)[0:2]
        self.image.dir_name = self.get('image', 'dir')
        self.image.files = sorted(os.listdir(self.image.dir_name))

        self.data.margins = {name: self.get_percent('data',
                                                    '%s-margin' % name)
                             for name in self.margin_types}

        self.data.alpha_levels = self.get_nums('data', 'alpha-levels')
        self.data.noise_levels = self.get_nums('data', 'noise-levels')
        self.data.background_colors = self.getint('data', 'background-colors')

        self.network.epochs = self.getint('network', 'epochs')
        self.network.in_size = self.image.size[0] * self.image.size[1]
        self.network.out_size = len(os.listdir(self.image.dir_name))
        self.network.path = self.get('network', 'path')

        self.seek.dir_name = self.get('seek', 'dir')
        self.seek.ratio = self.get_range('seek', 'ratio-range')
        self.seek.size = self.get_range('seek', 'size-range')
        self.seek.lum_sigma = self.get_percent('seek', 'lum-sigma')

    def get_nums(self, section, option, delimeter=',', t=float_or_percent):
        return [t(i) for i in self.get(section, option).split(delimeter) if i]

    def get_range(self, section, option, delimeter='-', t=float_or_percent):
        nums = self.get_nums(section, option, delimeter, t)
        return ValueRange(*nums[0:2])

    def get_percent(self, option, section):
        return float_or_percent(self.get(option, section))

    def load_images(self):
        dir_name = self.image.dir_name
        image_files = self.image.files
        return [imread(os.path.join(dir_name, name)) for name in image_files]

    def load_image_names(self):
        return [name.split('.')[0] for name in self.image.files]

    def __repr__(self):
        return "<Config %s>" % self.path

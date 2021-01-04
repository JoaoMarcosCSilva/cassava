import numpy as np
import json
import tensorflow as tf

class Genome:
    def __init__(self):
        self.score = None
        self.mutate(1)

    def mutate(self, probability):
        def p():
            return np.random.uniform(0,1) < probability
    
        if p():
            self.crop_size = np.random.randint(300, 512)
        if p():
            self.brightness_max_delta = np.random.uniform(0, 0.5)
        if p():
            self.contrast_lower = np.random.uniform(0,1)
            self.contrast_upper = np.random.uniform(self.contrast_lower, 1)
        if p():
            self.hue_max_delta = np.random.uniform(0, 0.5)
        if p():
            self.saturation_lower = np.random.uniform(0,1)
            self.saturation_upper = np.random.uniform(self.saturation_lower, 1)
        if p():
            self.log_lr_1 = np.random.uniform(-2, 0)
            self.lr1 = 10**self.log_lr_1
            self.log_lr_2 = np.random.uniform(-4, self.log_lr_1)
            self.lr2 = 10**self.log_lr_2
        if p():
            self.neg_log_momentum = np.random.uniform(-2, 0)
            self.momentum = 1 - 10**self.neg_log_momentum
        if p():
            self.dropout = np.random.uniform(0, 0.5)
        if p():
            self.smoothing = np.random.uniform(0,0.2)
        if p():
            self.cutout_size = np.random.randint(0, 256)
        if p():
            self.radians = np.random.uniform(0, np.pi/2)

    def get_augment(self):
        def augment(im, lbl):
            im = tf.image.random_flip_left_right(im)
            im = tf.image.random_flip_up_down(im)
            
            im = tf.image.resize(tf.image.random_crop(im, (self.crop_size,self.crop_size,3)), (512,512))
            
            im = tf.image.random_brightness(im, self.brightness_max_delta)
            im = tf.image.random_contrast(im, self.contrast_lower, self.contrast_upper)
            im = tf.image.random_hue(im, self.hue_max_delta)
            im = tf.image.random_saturation(im, self.saturation_lower, self.saturation_upper)
            
            im = tfa.image.random_cutout(tf.expand_dims(im,0), (self.cutout_size,self.cutout_size))[0]
            im = tfa.image.rotate(im, tf.random.uniform([], -self.radians, self.radians))
            
            return im, lbl
        return augment

    def evaluate(self, train_func):
        if self.score == None:
            self.score = train_func(self)
        return self.score

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent = 2)

    def read(self, filename):
        with open(filename, 'r') as f:
            d = json.load(f)
            self.__dict__ = d
                        
def crossover(a, b):
    c = Genome()
    
    halves = ['brightness_max_delta', 'contrast_lower', 'contrast_upper', 'hue_max_delta', 'saturation_lower', 'saturation_upper', 'log_lr_1', 'log_lr_2', 'neg_log_momentum', 'dropout', 'radians', 'smoothing']
    c.crop_size = (a.crop_size + b.crop_size) // 2
    c.cutout_size = (a.cutout_size + b.cutout_size) // 2
    for key in halves:
        c.__dict__[key] = (a.__dict__[key] + b.__dict__[key]) / 2

    c.lr1 = 10**c.log_lr_1
    c.lr2 = 10**c.log_lr_2
    
    c.momentum = 1 - 10**c.neg_log_momentum
    return c

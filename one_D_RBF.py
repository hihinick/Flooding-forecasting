import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf
from itertools import product
import matplotlib.pyplot as plt
# from tensorflow.keras.utils import multi_gpu_model#.keras

class FuzzyLayer(Layer):
    # 可變動參數fuzzy_size
    def __init__(self,
                 fuzzy_size=1,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.fuzzy_size = fuzzy_size
        super(FuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2  # 防呆機制
        self.input_dim = input_shape[-1]

        self.output_dim = self.input_dim*self.fuzzy_size
        self.mean = self.add_weight(name='mean',
                                    shape=(input_shape[1], self.fuzzy_size),
                                    initializer='uniform',
                                    trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(input_shape[1], self.fuzzy_size),
                                     initializer='uniform',
                                     trainable=True)
        # print(self.mean, self.sigma)
        super(FuzzyLayer, self).build(input_shape)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fuzzy_size': self.fuzzy_size 
        })
        return config

    def call(self, x):
        aligned_x = K.repeat_elements(
            K.expand_dims(x, axis=-1), self.fuzzy_size, -1)
        aligned_mean = self.mean
        aligned_sigma = self.sigma
#         print(aligned_x, aligned_mean, aligned_sigma)

        exp = K.exp(-0.5 * (K.square((aligned_x - aligned_mean) / aligned_sigma)))
        xc = 1./(aligned_sigma*K.sqrt(K.variable(2.*np.pi))*exp+0.0000001)
        xc = exp
        #print(K.flatten(xc))
        return xc  # xc / K.maximum(sums, less)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
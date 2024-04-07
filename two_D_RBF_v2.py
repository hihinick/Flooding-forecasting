import tensorflow as tf
import numpy as np


class FuzzyLayer(tf.keras.layers.Layer):

    # 可變動參數fuzzy_size
    def __init__(self, output_dim,count_W = 15,count_L = 15 ,window_count=6 ,look_back=12, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.count_W = count_W
        self.count_L = count_L
        self.window_count = window_count
        self.look_back = look_back
        self.output_dim = output_dim
        self.create_nine_grid()
#         self.x_y_all = x_y_all

        super(FuzzyLayer, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim
        })
        return config

    def build(self, input_shape):

        init = tf.random_uniform_initializer(minval=0.1, maxval=0.9)
        self.sigma = self.add_weight(name='sigma', shape=(
            self.window_count * self.count_W * self.count_L,), initializer=init, trainable=True)

        super(FuzzyLayer, self).build(input_shape)

    def create_nine_grid(self):
        #################################################
        count = max(self.count_W ,self.count_L)
        if self.count_W > self.count_L:
            del_axis = 2
        else:
            del_axis = 1
        x_al = np.zeros((count * count,int(count),int(count)))
        y_al = np.zeros((count * count,int(count),int(count)))
        k = 0
        time = 0
        for i in range(count):
            m=0
            for j in range(count):
                for t in range(count):
                    x_al[time][t] = np.arange(m,m+count)
                    y_al[time].T[t] = np.arange(k,k+count)
                time = time + 1
                m = m-1
            k = k - 1
        minus = (np.arange(abs(self.count_W - self.count_L))+1)*-1 #刪掉多餘的位置
        x_al = tf.Variable(x_al)
        y_al = tf.Variable(y_al)
        self.x_y_ = x_al ** 2 + y_al ** 2
        self.x_y_ = np.delete(self.x_y_, minus, axis=del_axis)
        self.x_y_ = self.x_y_[:self.count_W*self.count_L]
#         self.x_y_ = tf.Variable(x_y_)
        self.x_y_ = tf.reshape(self.x_y_, [1, self.count_W*self.count_L, self.count_W, self.count_L])
        self.x_y_all = tf.keras.backend.repeat_elements(self.x_y_, self.window_count, 0)
        self.x_y_all = tf.dtypes.cast(self.x_y_all, tf.float32)
        
#         return 
        #################################################

    def call(self, input):

        # make Gaussian Filter
        all_x_y = tf.reshape(self.x_y_all, [self.window_count * self.count_W * self.count_L, self.count_W * self.count_L])
        out = all_x_y / self.sigma[:, None] ** 2
        out_exp = tf.math.exp(-0.5 * out)
        out_sum = tf.reduce_sum(out_exp, 1)
        self.nor_out = out_exp / out_sum[:, None]
        self.nor_out = tf.reshape(self.nor_out, [self.window_count * self.count_W * self.count_L, self.count_W, self.count_L])

        x_input = tf.keras.backend.repeat_elements(input, self.count_W * self.count_L, 1)
        output = x_input*self.nor_out
        return output

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class MaxPoolingWithArgmax2D(Layer):
    
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self)
        self.padding = padding
        self.strides = strides
        self.pool_size = pool_size
    
    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        
        ksize = [1, *pool_size, 1]
        strides = [1, *strides, 1]
        
        output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize = ksize, 
                                                   strides=strides, padding=padding)
        
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax ]
    


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self)
        self.size = size
    
    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        
        

    
      
    
    
from keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K

from capsnet.layers.activation import squash


class Capsule(layers.Layer):
    def __init__(self, num_capsule, capsule_dim, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.capsule_dim = capsule_dim
        self.routings = routings
        self.kernel_initializer = kernel_initializer

        self.in_num_capsule = None
        self.in_capsule_dim = None
        self.W = None

    def build(self, input_shape):
        self.in_num_capsule = input_shape[1]
        self.in_capsule_dim = input_shape[2]

        self.W = self.add_weight(name='W', shape=[self.num_capsule, self.in_num_capsule, self.capsule_dim, self.in_capsule_dim],
                                 initializer=self.kernel_initializer)
        super(Capsule, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = K.expand_dim(inputs, 1)
        inputs = K.tile(inputs, [1, self.num_capsule, 1, 1])
        inputs = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs)

        b = tf.zeros(shape=[K.shape(inputs)[0], self.num_capsule, self.in_num_capsule])

        for i in range(self.routings):
            c = tf.nn.softmax(b, dim=1)
            outputs = squash(K.batch_dot(c, inputs, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.capsule_dim])

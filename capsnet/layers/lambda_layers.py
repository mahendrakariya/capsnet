from keras import layers
import tensorflow.keras.backend as K

from capsnet.layers.activation import squash


def primary_capsule_layer(inputs, capsule_dim, n_channels, kernel_size=9, strides=2, padding="valid"):
    out = layers.Conv2D(filters=capsule_dim * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                        name="PrimaryCapsuleLayer")(inputs)
    out = layers.Reshape(target_shape=[-1, capsule_dim])(out)
    return layers.Lambda(squash)(out)


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        return K.batch_flatten(inputs * K.expand_dims(mask, -1))

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

from keras import layers

from capsnet.layers.activation import squash


def primary_capsule_layer(inputs, capsule_dim, n_channels, kernel_size=9, strides=2, padding="valid"):
    out = layers.Conv2D(filters=capsule_dim * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                        name="PrimaryCapsuleLayer")(inputs)
    out = layers.Reshape(target_shape=[-1, capsule_dim])(out)
    return layers.Lambda(squash)(out)


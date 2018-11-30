import numpy as np
from keras import layers, models
from keras import backend as K

from capsnet.layers.capsule import Capsule
from capsnet.layers.layers import primary_capsule_layer, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings=3):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primary_caps = primary_capsule_layer(conv1, capsule_dim=8, n_channels=32)
    digit_caps = Capsule(num_capsule=n_class, capsule_dim=16, routings=routings)(primary_caps)
    out_caps = Length(name="capsnet")(digit_caps)

    # Decoder Network
    y = layers.Input(shape=(n_class,))
    masked_for_training = Mask()([digit_caps, y])
    masked_for_inference = Mask()(digit_caps)

    decoder = models.Sequential(name='decoder_net')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape))

    train_model = models.Model([x, y], [out_caps, decoder(masked_for_training)])
    eval_model = models.Model(x, [out_caps, decoder(masked_for_inference)])

    return train_model, eval_model

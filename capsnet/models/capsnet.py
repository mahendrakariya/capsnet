import numpy as np
from keras import layers, models, callbacks, optimizers
from keras import backend as K

from capsnet.layers.capsule import Capsule
from capsnet.layers.layers import primary_capsule_layer, Length, Mask
from capsnet.layers.loss import margin_loss
from capsnet.viz.plot_log import plot_log

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


def train(model, data, args):
    (x_train, y_train), (x_test, y_test) = data

    log = callbacks.CSVLogger(args.save_dir + "/log.csv")
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc', save_best_only=True,
                                           save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[margin_loss, 'mse'], loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)
    return model

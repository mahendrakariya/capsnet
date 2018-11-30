import tensorflow.keras.backend as K


def squash(vectors, axis=-1):
    squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    return (squared_norm / (1 + squared_norm)) * (vectors / K.sqrt(squared_norm + K.epsilon()))


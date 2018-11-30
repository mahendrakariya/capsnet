import tensorflow.keras.backend as K


def margin_loss(labels, logits, m_plus=0.9, m_minus=0.1, downweight=0.5):
    positive_cost = labels * K.square(K.maximum(0., m_plus - logits))
    negative_cost = (1 - labels) * K.square(K.maximum(0., logits - m_minus))

    loss = positive_cost + downweight * negative_cost

    return K.mean(K.sum(loss, 1))

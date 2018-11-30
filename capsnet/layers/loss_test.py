import unittest

import numpy as np
import tensorflow as tf

from capsnet.layers.loss import margin_loss


class LossTest(unittest.TestCase):
    def test_margin_loss(self):
        labels = tf.constant([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        logits = tf.constant([[-0.3, 0.3, 0.9], [1.2, 0.5, -0.5]])

        expected_loss = 1.06999993

        with tf.Session() as sess:
            actual_loss = sess.run(margin_loss(labels, logits))

        np.testing.assert_almost_equal(actual_loss, expected_loss)

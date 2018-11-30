import unittest

import numpy as np
import tensorflow as tf

from capsnet.layers.activation import squash


class ActivationTest(unittest.TestCase):
    def test_squash(self):
        inp_vector = tf.ones((2, 3, 2))
        with tf.Session() as sess:
            squashed = sess.run(squash(inp_vector))

        expected_output = [
            [[0.471404493, 0.471404493], [0.471404493, 0.471404493], [0.471404493, 0.471404493]],
            [[0.471404493, 0.471404493], [0.471404493, 0.471404493], [0.471404493, 0.471404493]]
        ]

        np.testing.assert_equal(len(squashed.shape), len(inp_vector.shape))
        np.testing.assert_almost_equal(squashed, expected_output)

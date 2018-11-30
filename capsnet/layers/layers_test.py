import unittest

import numpy as np
import tensorflow as tf

from capsnet.layers.layers import primary_capsule_layer


class LambdaLayersTest(unittest.TestCase):
    def test_primary_capsule_layer(self):
        input_tensor = tf.random_uniform((2, 11, 11, 2))
        capsule_dim = 2
        n_channels = 3

        out = primary_capsule_layer(input_tensor, capsule_dim, n_channels)

        np.testing.assert_equal(out.get_shape().as_list(), [2, 12, 2])

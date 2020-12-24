import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from model.xception import Xception


def test_num_parameters():
    model = Xception()
    inputs = tf.keras.Input(shape=[299, 299, 3])
    model(inputs)
    assert(np.sum([K.count_params(p) for p in model.trainable_weights]) == 22855952)

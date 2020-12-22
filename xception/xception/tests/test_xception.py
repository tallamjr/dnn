import numpy as np
import tensorflow.keras.backend as K

from model.xception import Xception


def test_num_parameters():
    model = Xception()
    assert(np.sum([K.count_params(p) for p in model.trainable_weights]) == 22855952)

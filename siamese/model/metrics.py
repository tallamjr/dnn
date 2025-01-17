import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() > 0.5
    return np.mean(pred == y_true)


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


class ContrastiveLossWithMargin(tf.keras.losses.Loss):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # initialize instance attributes
    def __init__(self, margin, name="contrastive_loss"):
        super().__init__(name=name)
        self.margin = margin

    # compute loss
    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        loss = K.mean(y_true * square_pred + (1 - y_true) * margin_square)
        return loss

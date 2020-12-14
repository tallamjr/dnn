import numpy as np
import random
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import RMSprop

from model.siamese import Siamese
from model.metrics import (
    compute_accuracy,
    euclidean_distance,
    eucl_dist_output_shape,
    ContrastiveLossWithMargin,
)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


def create_pairs_on_set(images, labels):

    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype('float32')

    return pairs, y


# load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# prepare train and test sets
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# normalize values
train_images = train_images / 255.0
test_images = test_images / 255.0

# create pairs on train and test sets
tr_pairs, tr_y = create_pairs_on_set(train_images, train_labels)
ts_pairs, ts_y = create_pairs_on_set(test_images, test_labels)

base_network = Siamese()

# create the left input and point to the base network
input_a = tf.keras.Input(
    shape=(
        28,
        28,
    ),
    name="left_input",
)
vect_output_a = base_network(input_a)

# create the right input and point to the base network
input_b = tf.keras.Input(
    shape=(
        28,
        28,
    ),
    name="right_input",
)
vect_output_b = base_network(input_b)

# measure the similarity of the two vector outputs
# output = tf.keras.layers.Lambda(
#     lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1])
# )([vect_output_a, vect_output_b])

output = tf.keras.layers.Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

# specify the inputs and output of the model
model = tf.keras.Model([input_a, input_b], output)

rms = RMSprop()
model.compile(loss=ContrastiveLossWithMargin(margin=1), optimizer=rms)

history = model.fit(
    [tr_pairs[:, 0], tr_pairs[:, 1]],
    tr_y,
    epochs=20,
    batch_size=128,
    validation_data=([ts_pairs[:, 0], ts_pairs[:, 1]], ts_y),
)

loss = model.evaluate(x=[ts_pairs[:, 0], ts_pairs[:, 1]], y=ts_y)

y_pred_train = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
train_accuracy = compute_accuracy(tr_y, y_pred_train)

y_pred_test = model.predict([ts_pairs[:, 0], ts_pairs[:, 1]])
test_accuracy = compute_accuracy(ts_y, y_pred_test)

print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))

import tensorflow as tf


class Siamese(tf.keras.Model):
    def __init__(self):
        super(Siamese, self).__init__()

        self.flatten = tf.keras.layers.Flatten(name="flatten_input")
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name="first_base_dense")
        self.drop1 = tf.keras.layers.Dropout(0.1, name="first_dropout")

        self.dense2 = tf.keras.layers.Dense(128, activation='relu', name="second_base_dense")
        self.drop2 = tf.keras.layers.Dropout(0.1, name="second_dropout")

        self.dense3 = tf.keras.layers.Dense(128, activation='relu', name="third_base_dense")

    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        x = self.dense1(x)

        if training:
            x = self.drop1(x, training=training)
        x = self.dense2(x)

        if training:
            x = self.drop2(x, training=training)

        x = self.dense3(x)
        return x

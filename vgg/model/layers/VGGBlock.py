import tensorflow as tf


class Block(tf.keras.Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions

        # Define a conv2D_0, conv2D_1, etc based on the number of repetitions
        for i in range(repetitions):

            # Define a Conv2D layer, specifying filters, kernel_size, activation and padding.
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same')

        # Define the max pool layer that will be added after the Conv2D blocks
        self.max_pool = tf.keras.layers.MaxPool2D(
            pool_size=(pool_size, pool_size), strides=(strides, strides)
        )

    def call(self, inputs):
        # access the class's conv2D_0 layer
        conv2D_0 = vars(self)['conv2D_0']

        # Connect the conv2D_0 layer to inputs
        x = conv2D_0(inputs)

        # for the remaining conv2D_i layers from 1 to `repetitions` they will be connected to the previous layer
        for i in range(1, self.repetitions):
            # access conv2D_i by formatting the integer `i`
            conv2D_i = vars(self)[f"conv2D_{i}"]

            # Use the conv2D_i and connect it to the previous layer
            x = conv2D_i(x)

        # Finally, add the max_pool layer
        max_pool = self.max_pool(x)

        return max_pool

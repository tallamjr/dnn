import tensorflow as tf

from model.layers import ResidualUnit


class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            64, 7, strides=2, input_shape=[224, 224, 3], padding="same", use_bias=False
        )

        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        prev_filters = 64

        self.residuals = []
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.residuals.append(ResidualUnit(filters, strides=strides))
            prev_filters = filters

        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.max_pool(x)
        for layer in self.residuals:
            x = layer(x)
        x = self.gap(x)
        x = self.flatten(x)
        return self.classifier(x)

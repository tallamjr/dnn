import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    SeparableConv2D,
    Add,
    Dense,
    BatchNormalization,
    ReLU,
    MaxPool2D,
    GlobalAvgPool2D,
)


class ConvBatchNormBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(ConvBatchNormBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def call(self, inputs):
        x = inputs
        x = Conv2D(filters=self.filters,
                   kernel_size=self.kernel_size,
                   strides=self.strides,
                   padding='same',
                   use_bias=False,
                   input_shape=(299, 299, 3))(x)
        x = BatchNormalization()(x)
        return x


class SeparableConvBatchNormBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(SeparableConvBatchNormBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def call(self, inputs):
        x = inputs
        x = SeparableConv2D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            strides=self.strides,
                            padding='same',
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        return x


class EntryFlow(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EntryFlow, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        x = ConvBatchNormBlock(x, filters=32, kernel_size=3, strides=2)
        x = ReLU()(x)
        x = ConvBatchNormBlock(x, filters=64, kernel_size=3)
        tensor = ReLU()(x)

        x = SeparableConvBatchNormBlock(tensor, filters=128, kernel_size=3)
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=128, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        tensor = ConvBatchNormBlock(tensor, filters=128, kernel_size=1, strides=2)

        x = Add()([tensor, x])
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=256, kernel_size=3)
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=256, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        tensor = ConvBatchNormBlock(tensor, filters=256, kernel_size=1, strides=2)

        x = Add()([tensor, x])
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=728, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        tensor = ConvBatchNormBlock(tensor, filters=728, kernel_size=1, strides=2)
        x = Add()([tensor, x])

        return x


class MiddleFlow(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MiddleFlow, self).__init__(**kwargs)

    def call(self, inputs):
        tensor = inputs
        x = ReLU()(tensor)
        x = SeparableConvBatchNormBlock(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=728, kernel_size=3)

        tensor = Add()([tensor, x])

        return tensor


class ExitFlow(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExitFlow, self).__init__(**kwargs)

    def call(self, inputs):
        tensor = inputs
        x = ReLU()(tensor)
        x = SeparableConvBatchNormBlock(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=1024, kernel_size=3)
        x = MaxPool2D(3, strides=2, padding='same')(x)

        tensor = ConvBatchNormBlock(tensor, filters=1024, kernel_size=1, strides=2)

        x = Add()([tensor, x])
        x = SeparableConvBatchNormBlock(x, filters=1536, kernel_size=3)
        x = ReLU()(x)
        x = SeparableConvBatchNormBlock(x, filters=2048, kernel_size=3)
        x = ReLU()(x)
        x = GlobalAvgPool2D()(x)
        x = Dense(units=1000, activation='softmax')(x)

        return x

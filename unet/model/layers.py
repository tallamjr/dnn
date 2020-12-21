import tensorflow as tf

OUTPUT_CHANNELS=3


class Conv2DBlock(tf.keras.Model):
    def __init__(self, n_filters, kernel_size=3):
        super(Conv2DBlock, self).__init__(name='')
        '''
          Adds 2 convolutional layers with the parameters passed to it

          Args:
            input_tensor (tensor) -- the input tensor
            n_filters (int) -- number of filters
            kernel_size (int) -- kernel size for the convolution

          Returns:
            tensor of output features
          '''
        self.conv2d = tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
            input_shape=(128, 128, 3),
            data_format="channels_first",
        )
        self.act = tf.keras.layers.Activation('relu')

    def call(self, input_tensor):
        x = input_tensor
        for i in range(2):
            x = self.conv2d(x)
        x = self.act(x)
        return x


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters=64, pool_size=(2, 2), dropout=0.3, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        '''
        Adds two convolutional blocks and then perform down sampling on output of convolutions.

        Args:
        input_tensor (tensor) -- the input tensor
        n_filters (int) -- number of filters
        kernel_size (int) -- kernel size for the convolution

        Returns:
        f - the output features of the convolution block
        p - the maxpooled features with dropout
        '''
        self.feats = Conv2DBlock(n_filters=n_filters)
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs):

        f = self.feats(inputs)
        p = self.maxpool(f)
        p = self.dropout(p)

        return f, p


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_filters=64, pool_size=(2, 2), dropout=0.3, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        '''
        This function defines the encoder or downsampling path.

        Args:
        inputs (tensor) -- batch of input images

        Returns:
        p4 - the output maxpooled features of the last encoder block
        (f1, f2, f3, f4) - the output features of all the encoder blocks
        '''
        self.encoderblock_1 = EncoderBlock(n_filters=64, pool_size=(2, 2), dropout=0.3)
        self.encoderblock_2 = EncoderBlock(n_filters=128, pool_size=(2, 2), dropout=0.3)
        self.encoderblock_3 = EncoderBlock(n_filters=256, pool_size=(2, 2), dropout=0.3)
        self.encoderblock_4 = EncoderBlock(n_filters=512, pool_size=(2, 2), dropout=0.3)

    def call(self, inputs):

        f1, p1 = self.encoderblock_1(inputs)
        f2, p2 = self.encoderblock_2(p1)
        f3, p3 = self.encoderblock_3(p2)
        f4, p4 = self.encoderblock_4(p3)

        return p4, (f1, f2, f3, f4)


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, n_filters=64, pool_size=(2, 2), dropout=0.3, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        '''
        This function defines the bottleneck convolutions to extract more features before the upsampling layers.
        '''
        self.bottleneck = Conv2DBlock(n_filters=1024)

    def call(self, inputs):
        x = self.bottleneck(inputs)
        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        '''
        defines the one decoder block of the UNet

        Args:
        inputs (tensor) -- batch of input features
        conv_output (tensor) -- features from an encoder block
        n_filters (int) -- number of filters
        kernel_size (int) -- kernel size
        strides (int) -- strides for the deconvolution/upsampling
        padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

        Returns:
        c (tensor) -- output features of the decoder block
        '''
        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(
            n_filters, kernel_size, strides=strides, padding="same"
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv2d_block = Conv2DBlock(n_filters, kernel_size=3)

        self.conv_output = conv_output

    def call(self, inputs):
        u = self.conv2d_transpose(inputs)
        c = tf.keras.layers.Concatenate()([u, self.conv_output])
        c = self.dropout(c)
        c = self.conv2d_block(c)

        return c


class Decoder(tf.keras.layers.Layer):
    def __init__(self, convs, output_channels, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        '''
        Defines the decoder of the UNet chaining together 4 decoder blocks.

        Args:
        inputs (tensor) -- batch of input features
        convs (tuple) -- features from the encoder blocks
        output_channels (int) -- number of classes in the label map

        Returns:
        outputs (tensor) -- the pixel wise label map of the image
        '''
        self.convs = convs
        f1, f2, f3, f4 = self.convs

        self.decoder_block1 = DecoderBlock(
            f4, n_filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.3
        )
        self.decoder_block2 = DecoderBlock(
            f3, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3
        )
        self.decoder_block3 = DecoderBlock(
            f2, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3
        )
        self.decoder_block4 = DecoderBlock(
            f1, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3
        )

        self.conv2d = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='softmax')

    def call(self, inputs):

        c6 = self.decoder_block1(inputs)
        c7 = self.decoder_block2(c6)
        c8 = self.decoder_block3(c7)
        c9 = self.decoder_block4(c8)

        outputs = self.conv2d(c9)
        return outputs

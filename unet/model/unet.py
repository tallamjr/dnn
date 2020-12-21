import tensorflow as tf

from model.layers import Encoder, Bottleneck, Decoder

OUTPUT_CHANNELS = 3


class UNet(tf.keras.Model):
    def __init__(self, output_channels=OUTPUT_CHANNELS):
        super(UNet, self).__init__()
        '''
        Defines the UNet by connecting the encoder, bottleneck and decoder.
        '''
        self.output_channels = output_channels

    def call(self, inputs):
        # specify the input shape
        # inputs = tf.keras.layers.Input(shape=(128, 128,3,))

        # feed the inputs to the encoder
        encoder_output, convs = Encoder()(inputs)

        # feed the encoder output to the bottleneck
        bottle_neck = Bottleneck()(encoder_output)

        # feed the bottleneck and encoder block outputs to the decoder
        # specify the number of classes via the `output_channels` argument
        outputs = Decoder(convs, self.output_channels)(bottle_neck)

        # create the model
        # model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return outputs

    # def build(self, **kwargs):
    #     """
    #     Replicates Model(inputs=[inputs], outputs=[outputs]) of functional model.
    #     """
    #     # Replace with shape=[None, None, None, 1] if input_shape is unknown.
    #     # specify the input shape
    #     inputs = tf.keras.layers.Input(
    #         shape=(128, 128, 3,)
    #     )
    #     outputs = self.__call__(inputs)
    #     super(UNet, self).__init__(name="UNet", inputs=inputs, outputs=outputs, **kwargs)

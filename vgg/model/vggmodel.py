import tensorflow as tf

from model.layers import VGGBlock


class MyVGG(tf.keras.Model):

    def __init__(self, num_classes):
        super(MyVGG, self).__init__()

        # Creating blocks of VGG with the following
        # (filters, kernel_size, repetitions) configurations
        self.block_a = VGGBlock.Block(filters=64, kernel_size=3, repetitions=2)
        self.block_b = VGGBlock.Block(filters=128, kernel_size=3, repetitions=2)
        self.block_c = VGGBlock.Block(filters=256, kernel_size=3, repetitions=3)
        self.block_d = VGGBlock.Block(filters=512, kernel_size=3, repetitions=3)
        self.block_e = VGGBlock.Block(filters=512, kernel_size=3, repetitions=3)

        # Classification head
        # Define a Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # Create a Dense layer with 256 units and ReLU as the activation function
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        # Finally add the softmax classifier using a Dense layer
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Chain all the layers one after the other
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x

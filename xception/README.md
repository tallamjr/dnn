# UNet

Implementation of the **UNet** architecture shown below.

<p>
    <!<img src="data/VGG.png" width="220" height="240" />
    <img src='https://drive.google.com/uc?export=view&id=1BeQSKL2Eq6Fw9iRXsN1hgunY-CS2nH7V' alt='unet'>
</p>

A UNet consists of an encoder (downsampler) and decoder (upsampler) with a bottleneck in between.
The gray arrows correspond to the skip connections that concatenate encoder block outputs to each
stage of the decoder. Let's see how to implement these starting with the encoder.

### Encoder

Like the FCN model you built in the previous lesson, the encoder here will have repeating blocks
(red boxes in the figure below) so it's best to create functions for it to make the code modular.
These encoder blocks will contain two Conv2D layers activated by ReLU, followed by a MaxPooling and
Dropout layer. As discussed in class, each stage will have increasing number of filters and the
dimensionality of the features will reduce because of the pooling layer.

<p>
    <img src='https://drive.google.com/uc?export=view&id=1Gs9K3_8ZBn2_ntOtJL_-_ww4ZOgfyhrS' alt='unet'>
</p>

### Bottleneck

A bottleneck follows the encoder block and is used to extract more features. This does not have a
pooling layer so the dimensionality remains the same.

### Decoder

Finally, we have the decoder which upsamples the features back to the original image size. At each
upsampling level, you will take the output of the corresponding encoder block and concatenate it
before feeding to the next decoder block. This is summarized in the figure below.

<p>
    <img src='https://drive.google.com/uc?export=view&id=1Ql5vdw6l88vxaHgk7VjcMc4vfyoWYx2w' alt='unet_decoder'>
</p>

##### Refs: This information about UNet has been taken from Coursera course: Advanced Computer Vision in TensorFlow

## Training

Simply running `python train.py` should start the download of the dataset and being training. _This
will take a while depending on the hardware_

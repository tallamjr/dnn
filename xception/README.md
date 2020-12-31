# Xception

Implementation of the **Xception** architecture shown below.

<p>
    <img src=https://raw.githubusercontent.com/Machine-Learning-Tokyo/DL-workshop-series/master/Part%20I%20-%20Convolution%20Operations/images/Xception/Xception.png width="600">
</p>

# Architecture

The model is separated in 3 flows as:
- Entry flow
- Middle flow with 8 repetitions of the same block
- Exit flow

All Convolution and Separable Convolution layers are followed by batch normalization.

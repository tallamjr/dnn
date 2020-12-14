# ResNet

Implementation of the **ResNet** architecture. For this implementation, there are two variants, a
__mini__ version, shown here:

<p>
    <img src="data/mini-resnet.png" height="100"/>
</p>

as well as a deep version, __resnet34__, shown here:

<p>
    <img src="data/resent-34-overview.png"/>
</p>

Where the blue and orange blocks are defined as follows:

<p>
    <img src="data/identity-block.png"/>
</p>

<p>
    <img src="data/idenity-block-with-residual.png"/>
</p>


## Training

Simply running `python train-mini.py` or `python train-resnet34.py` should start the download of the
dataset and being training. _This will take a while depending on the hardware_

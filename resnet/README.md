# ResNet

Implementation of the **ResNet** architecture. For this implementation, there are two variants, a
__mini__ version, shown here:
<p>
    <img src="data/mini-resnet.png" height="400"/>
</p>
as well as a deeper version, __ResNet34__ shown here:
<p>
    <img src="data/resent-34-overview.png" height="500"/>
</p>
Where the blue and orange blocks are defined as follows:
<p>
    <img src="data/blocks-combined.png" height="500"/>
</p>
## Training

Simply running `python train-mini.py` or `python train-resnet34.py` should start the download of the
dataset and being training. _This will take a while depending on the hardware_

# Getting started

<img src="img/Jetson_nano.png" alt="Jtson Nano" width="400" height="auto">

1. attached a 32 GB microSD card.
2. connected it to a power supply that can deliver 5V⎓2A.
3. Download the Jetson Nano Developer Kit SD Card Image. can be found [here](https://developer.nvidia.com/embedded/downloads#?search=image).
4. insert the SD card in your computer.

The instructions to follow are for Linux:

> **_NOTE:_** The user guide instructions using etcher didn't work so I used this methos instead.

1. unzip the image
2. run this command: sudo dd if=Downloads/sd-blob-b01.img of=/dev/mmcblk0 bs=1M
3. Eject the SD card from computer via command line or file application.

links
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write

https://blog.roboflow.com/how-to-deploy-to-jetson-orin-nano-computer-vision/

https://forums.developer.nvidia.com/t/deploy-a-pretrained-custom-model-in-jetson-nano/208930

https://github.com/dusty-nv/jetson-inferences
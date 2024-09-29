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

On the Jetson NANO:
1. nsert the microSD card (with system image already written to it) into the slot on the underside of the Jetson Nano module.
2. connect the jetson nano to the power supply, computer display, USB keyboard and mouse.

<img src="img/set_up.jpg" alt="Load Cells" width="400" height="auto">

3. you'll see an ubuntu screen, follow the instruction, choose a language and so on (easy ubuntu setup).
4. Set username: ansamz, and Password: ****************** (shall be given upon request :D)
5. We get to initial setup mode.

[JETSON developer guide](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/index.html)

[Hello AI World](https://developer.nvidia.com/embedded/twodaystoademo#hello_ai_world), this link contains fun demos to try to deploy.

# Deploy a model:

Trying [this](https://github.com/dusty-nv/jetson-inferences) first.

There are multiple options, there a few ready projects but I am going for [](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md)



links:

https://blog.roboflow.com/how-to-deploy-to-jetson-orin-nano-computer-vision/

https://forums.developer.nvidia.com/t/deploy-a-pretrained-custom-model-in-jetson-nano/208930

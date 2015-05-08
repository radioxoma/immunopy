# Immunopy #

*Immunopy — realtime immunostain analysis application.*

The program acquires microscope camera video stream and performs realtime image analysis of current field of view. With color labels and statistics (labeling index, cell count), showing as video overlay, pathologist can observe assay in *augmented* way. This short [video](https://www.youtube.com/watch?v=d-7YBjyk-rw) with algorithm demonstration helps understand the concept.

Immunopy targeted to breast cancer immunohistochemical assays with nuclear markers (Ki-67, estrogen and progesterone receptors stained with DAB & hematoxylin).

## Installation ##
Immunopy is written in python 2. Dependences that not supported by [pip](https://pip.pypa.io) is listed in `setup.py` file near with "install_requires" section.

    python2 setup.py install


## Configuration ##

Image acquisition lies on [Micro-manager](https://www.micro-manager.org). You need to create [configuration file](https://micro-manager.org/wiki/Micro-Manager_Configuration_Guide) (e.g. `camera_demo.cfg`) with group *"System"* and preset *"Startup"*. This device configuration will be loaded as default on startup, also some camera settings can be changed during work.

It's necessary to calibrate microscope and define pixel size for used magnifications.


## Running

    cd immunopy
    python2 -m immunopy.main


## Backgroud ##
This project is my thesis work in [Vitebsk state medical university](http://www.vsmu.by), originally called "Автоматическая оценка иммуногистохимических препаратов в режиме реального времени".

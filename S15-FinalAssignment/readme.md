# Capstone Project - EVA5 Final Assigment
# Submitted by: Avnish Midha (avnishbm@gmail.com)


## Goal of the assignment:
The goal of this project was to create a Model using Transfer Learning such that given an Image,
the model can generate:

1. Bounding Boxes on the Image, highlighting Mask, Vest, Boots & Hardhat
2. Depth Image
3. Planer Surface Image, or Segmentation Image

We already have 3 models, with each of them achieving one purpose as follows:
Customized YoloV3 is trained to generate images with bounding boxes for Masks, Vests, Boots & Hardhat.
MidasNet that can generate Depth Images
PlaneRCNN that can generated segmentation images (plane surfaces)

We need to use Transfer Learning to come up with a single Model that can generate all 3 image types in single run.

## Solution:

### First Step: Generate more data, as 3000 images will not be sufficient.
For this purpose, it was required to generate data for training the model for depth and planer surfaces. Hence, relevant videos were downloaded from youtube that just had interiors of the house/offices shown without any person in it. Using the videos, images were generared using ffmpeg to extract an image every second. Sample command used to generate images from video:

ffmpeg -i DESIGNERVILLA.mp4 -r 1 ../interiors/int%04d.png

Additional images generated are stored here: 

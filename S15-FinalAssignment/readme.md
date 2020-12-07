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

### Code Structure:
https://github.com/midhaworks/EVA5-Avnish/tree/main/S15-FinalAssignment

ModelSummary.ipynb: Notebook showing the details of the 3 Base Models and the newly created combined model Tricycle. Also, Tricycle Model also saves the weights for itself after having loaded weights of each of it's layers from the base model.

MiDas - Folder containing MiDasNet code.

planercnn - Fodler containing PlaneRCNN code.

YoloV3 - Folder containing YoloV3 code.

Tricycle - Folder containing the code of the combined model. It has been named as Tricycle, as it has 3 wheels - relating to 3 outcomes & given that i have just started to learn Transfer Learning & it is first such model (like we start with tricycle to further graduate to riding a cycle, bike and so on) :)

attempts - The folder contains source code of the previous attempts made to defined a combined model

data - folder contains some base data, but for overall data, please refer the gdrive link (as not all data can be added to github due to data limit) - https://drive.google.com/drive/folders/1KMyWDwS76VVK5A9yiIAlwPNO2I3Dn-yA?usp=sharing


### First Step: Generate more data, as 3000 images will not be sufficient.
For this purpose, it was required to generate data for training the model for depth and planer surfaces. Hence, relevant videos were downloaded from youtube that just had interiors of the house/offices shown without any person in it. Using the videos, images were generared using ffmpeg to extract an image every second. Sample command used to generate images from video:

ffmpeg -i DESIGNERVILLA.mp4 -r 1 ../interiors/int%04d.png

5000+ images created from videos are stored here: 
https://drive.google.com/drive/folders/1EC8OrgCg0FZRWLWTvv2fzd75zzMFh-h_?usp=sharing

Based on these images, MidasNet was run on the images to generare corresponding depth images, that can serve as additional ground data for training the model for depth. 

Additional 5000+ depth images are stored here:
https://drive.google.com/drive/folders/127Ax6rktMwsdAfBAN6zwil0lJd9KLd4V?usp=sharing


Also PlaneRCNN model was run on these 5000+ images to generate corresponding planer images.

Additional 5000+ Planer images generated are stored here:
https://drive.google.com/drive/folders/1bifHkBqyP_x4_mXHY1d8AzEt8FqcWZMe?usp=sharing

Overall Data is stored here:
https://drive.google.com/drive/folders/1KMyWDwS76VVK5A9yiIAlwPNO2I3Dn-yA?usp=sharing

### Study & Analysis of the 3 models:
As a next step before planning transfer learning, it was important to understand the 3 models & their respectiev code base.

Here, i tried to look at the Model Summary & print the Model Children, as well as tried to visualize the interconnections between various modules within the model using external libraries like torchviz.

The notebook to visualize the models can be seen here: 
https://colab.research.google.com/drive/1x0YbgcNUk3OQ7P5Mv08CbSmVD0Lh_5gU?usp=sharing

Note: While the above ModelSummary.ipynb was used to study and analyse the model, it was also used to experiment and visualise the combined model, named as tricycle_model in the code above. More about the model is explained in next section.

Please ntoe that the entire work area was kept in gdrive:
https://drive.google.com/drive/folders/1oxMv0hdsHfwIJbx0eONKuDCwY4czTViJ?usp=sharing

#### Findings from Model Study:

1. The first thing that came to my mind was to view the models structure and try and define a custom model that would just take specific layers from the 3 preloaded models (MidasNet, YoloV3, PlaneRCNN). My first attempt was to try out visualising the models and to try and write that custom model. However, there were challenges faced when trying to load the 3 models and take bits and pieces from each model, the main problem here was my lack of skills and experience with PyTorch and be able to figure out how a custom model of this order (with 3 outcomes) is defined  & how various pieces could be frozen. While googling, i could find a number of articles showing how transfer learning is applied to a model to just change the last couple of layers, there was no such extensive/complex example available on the internet. 

Hence, i even tried formally leanring pytorch by going through all the documentation available o pytorch site, however, these documentations / training material also remained quite high level and did not delve into complex examples! However, after lot of reading & failed trials that resulted in custom model picking up the entire base models (Yolo, MidasNet, PlaneRCNN) as their children (based on how i was defining it), did click an idea of defining the model dynamically using a method, that takes instances of these base models and starts adding specific layers from these base model to a new Module instance, and then returns the Module (or model) thus created. While this was a very good idea, what was not clear was how i could define forward method for such a dynamically created model such that it generated 3 outcomes! Somehow, the complexity of the model kept keeping me at bay to move forward as thinking about the complex task ahead & whether i was going to hit a roadblock was a concern!

2. **bold The Yolo V3 model which was based on Darknet coding framework is a very well written & modular code. The concept of using config file (cfg) to define a model is very good, one that can be used to define a model quickly & the code is written such that it will continue to work without any change as far as the underlying components are adhered to while defining the model i.e. as far as one uses the predefined types, like convolutional, route, shortcut, maxpool, etc. ** Hence, it really interested me a lot to reuse this code structure and config mechanism, as it gives lot of flexibility to try out just about any model type in future, as we keep adding more supported module types to it. The best part here was that we could clearly define any route or shortcut, something that could be used to carry forward layers to later layers, like in attention mechanism. And more importantly, it suited our purpose to define 3 routes from the same encoder!!!!

Found this very good write up that explained the YoloV3 Model and source structure really well, though it shows YoloMini model, it could be easily related to main model: programmersought.com/article/97114912009/

2. However, looking at the MidasNet model & source code it was found that it had some sub-modules that may not be available in the Yolov3 config mechanism. Hence, it was required to enhance the YoloV3 model to support these additional building blocks if we were to use the YoloV3 framework to code the Combined Model.

3. PlaneRCNN source code though good, was outdated as it was using pieces that no longer worked in latest cuda version and also required compiling some libraries using C/C++ compilers. It was quite a challenge to get this code compiled and running. Given the amount of time i spent to get this working and the fact that i was still not able to get all the 3 Models running in a single setup, as MidasNet and YoloV3 had dependency on latest features which were not available in PlaneRCNN setup, i was seriously considering finding alternatives to PlaneRCNN, one that could be equally effective in generating Plane segmentation images and not have the dependency challenges associated with it. Based on the reading, i did come across interchangeable approaches like: 

a) MaskRCNN: 
https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/
https://github.com/matterport/Mask_RCNN

b) Considered generating Planar Segmentation from Depth Images: Did come across an interesting implementation (that did not require a separate network, but could reuse depth images to generate planer images), shall find the link again and update, somehow i had to restart my Macbook due to memory issues and lost the open link.


### Work Completed:

#### First tried writing dynamically created model, made 2 attempts for the same:

Attempt 1: Tried writing dynamically created model by having the 3 pretrained models as member variables:
https://github.com/midhaworks/EVA5-Avnish/blob/main/S15-FinalAssignment/attempts/tricycle_net_attempt1.py

Could not figure out if this will work, got blocked by the YoloV3 forward pass where in some Yolo model layers required passing the outputs array. This also complicated matters as looking at YoloV3 code it appeared quite complex.

Attemp 2:
https://github.com/midhaworks/EVA5-Avnish/blob/main/S15-FinalAssignment/attempts/tricycle_net.py

This approach was about not having the base models as member variables and instead have them as function paramters so they don't show up fully under model summary or model children and only relevant layers get used /shown. However, this approach also got stuck due to complexity of YoloV3 forward pass which had multiple arrays being tracked, yolo_layers and out array and then some layers required outputs to be passed. Please note that this analysis was before fully understanding the YoloV3 code structure. Next step changed my perception about the YoloV3 code completely.

#### Final Attempt:
Having done some more reading out YoloV3 and how it can be customised, came across this very descriptive article: programmersought.com/article/97114912009/

Along with this article, having studied the cfg file & it's source code as well, it started to develop my liking and interest towards this code structure and the possibility of using the same for the capstone project as it offered good flexibility to define routes and shortcuts and provided some required building blocks as well. However, having reviewed MidasNet and PlaneRCNN it was quite evident that i will need to add some more building blocks to YoloV3 code such that the config file will be able to take those building blocks as valid blocks for any given model being defined.

My first goal was to name the Combined Model as Tricycle net and start by integrating MidasNet layers with YoloV3 (as they had no dependency issues). Having worked on the code, i ended up making following improvements to the Yolo V3 code:

1. Updated the cfg code to support definition of high level layers viz. 4 main layers of resnext101 (resnext101_layer1, resnext101_layer2, resnext101_layer3, resnext101_layer4) and MidasDecoder (defined as a new Module). Associated parameters like pretrained, features, non_negative also added in supported list of features in parse_config.py

2. Updated the implementation of convolutional layer, to also support specifying in_channels and bias values. It was particularly needed to add some encoder layers of midasnet, where in in_channels was defined.

3. Created cfg file for the combined Tricycle model by the name tricycle.cfg, ensuring that MidasNet Encoder was used and separate decoders for Depth & BBox detection, taking layer definitions from respective models & ensuring overall model integrity: https://github.com/midhaworks/EVA5-Avnish/blob/main/S15-FinalAssignment/Tricycle/cfg/tricycle.cfg

4. Wrote a function to load model weights from pretrained MidasNet & Customized YoloV3 models, Darknet->load_weights() function in models.py: https://github.com/midhaworks/EVA5-Avnish/blob/main/S15-FinalAssignment/Tricycle/models.py

Please note that while writing and testing the tricycle.cfg, most layers compatibility issues were resolved and also the model designed such that enough capacity was ensured (similar to base model) to ensure equivalent learning of features. Also, remaining 1 or 2 compatibility issues were resolved when testing the load_weights() function as weights data won't match if layers had different size.

The updated code for the combined layer is available here: https://github.com/midhaworks/EVA5-Avnish/tree/main/S15-FinalAssignment/Tricycle

Notebook making use of this model and displaying the Model Summary and Children, apart from saving the weights of the New Model: https://github.com/midhaworks/EVA5-Avnish/blob/main/S15-FinalAssignment/ModelsSummary.ipynb


## Achievements:
1. Learned how to generate additional training data for combined models to learn.

2. Learned the Darknet Coding & Configuration Framework, which i believe can be used further as a framework to define and create new Models, and it may be of major use if one wants to configure and test a number of configurations overnight or over a duration automatically, to be run based on a number of configuration files provided to the program to load, train and generate report.

3. Transfer Learning learned in detail. Completed Integration of MidasNet & YoloV3 successfully also ensuring that the Darknet Framework was re-used and enhanced, so it can be easily be used to further add PlaneRCNN Layers and trained further to create the final Model. Shall be training the model further to generate bbox and depth images effectively. And further to add MaskRCNN layers (to avoid dependency issues with PlaneRCNN) for Plane Segmentation, if that's ok.

## Learnings:
1. Lack of experience and skills with Pytorch, Transfer Learning initially caused delays for me to start, despite keen interest. However, having gone through the source code and understanding each of the models in detail was key here, which i could better imagine after having searched for many more articles on YoloV3, and once the code was understood better, it has given me good confidence, where in pytorch skills really did not matter much, could figure out those aspects quickly once the framework was in mind that worked! Learning here was to delve into model code and understanding structure of the model in detail is key, and next time i should focus on that aspect early on to complete the work at hand faster.

2. Been stretched on time due to my startup's work that is also demanding in these weeks. It was a bit stressful not being able to progress as i had no idea how transfer learning was done, that too for such complex models like YoloV3 & PlaneRCNN. May be some basic training / coding assignments upfront on how weights are transferred or how models can be reused from other models, etc could have been useful to show the way forward. Somehow during these weeks, the entire group became uncommunicative, may be because it was an individual assignment, but had group members been communicative or helpful to others to share how these basic things are done (like custom model creation or transferring weights), it could have helped other members to get started earlier on & done much more.   











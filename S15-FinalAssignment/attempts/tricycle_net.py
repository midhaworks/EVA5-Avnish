"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import torch

from torchvision.transforms import Compose
from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.base_model import BaseModel
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchsummary import summary


from YoloV3.models import *  # set ONNX_EXPORT in models.py
from YoloV3.utils.datasets import *
from YoloV3.utils.utils import *


MIDAS_MODEL_PATH = "/content/gdrive/My Drive/ML/model/midas-f46da743.pt"
PLANERCNN_MODEL_PATH = "./models"
YOLOV3_MODEL_PATH = "/content/gdrive/My Drive/ML/model/yolov3.pt"
YOLOV3_CONFIG_PATH =  "/content/gdrive/My Drive/ML/YoloV3/cfg/yolov3-custom.cfg"



class TricycleNet(BaseModel):
    """Network for monocular depth estimation, segmentation & object bounding box detection, .
    """

    def __init__(self, midas_model, yolov3_model, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(TricycleNet, self).__init__()

        use_pretrained = False if path is None else True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: %s" % device)

        self.encoder = midas_model.pretrained
        self.depth_decoder = midas_model.scratch
        self.depth_output = midas_model.output_conv
        self.bbox_decoder = torch.nn.Sequential(
            yolov3_model.module_list[99],
            yolov3_model.module_list[100],
            yolov3_model.module_list[101],
            yolov3_model.module_list[102],
            yolov3_model.module_list[103],
            yolov3_model.module_list[104],
            yolov3_model.module_list[105],
            yolov3_model.module_list[106],
            yolov3_model.module_list[107],
            yolov3_model.module_list[108],
            yolov3_model.module_list[109],
            yolov3_model.module_list[110],
            yolov3_model.module_list[111],
            yolov3_model.module_list[112],
            yolov3_model.module_list[113],
        )




    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        out = []
        x = self.encoder(x)
        x = self.depth_decoder(x)

        decoder1_out = self.depth_output(x)

        depth_out = torch.squeeze(decoder1_out, dim=1)

        # decoder2_layer_1 = self.yolov3_model.module_list[104](encoder_layer_4)
        # decoder2_layer_2 = self.yolov3_model.module_list[105](decoder2_layer_1, out)
        # decoder2_layer_3 = self.yolov3_model.module_list[106](decoder2_layer_2)
        # decoder2_layer_4 = self.yolov3_model.module_list[107](decoder2_layer_3)
        # decoder2_layer_5 = self.yolov3_model.module_list[108](decoder2_layer_4)
        # decoder2_layer_6 = self.yolov3_model.module_list[109](decoder2_layer_5)
        # decoder2_layer_7 = self.yolov3_model.module_list[110](decoder2_layer_6)
        # decoder2_layer_8 = self.yolov3_model.module_list[111](decoder2_layer_7)
        # decoder2_layer_9 = self.yolov3_model.module_list[112](decoder2_layer_8)
        # decoder2_output = self.yolov3_model.module_list[113](decoder2_layer_9, out)
        
        return depth_out

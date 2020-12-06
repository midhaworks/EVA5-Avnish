import torch
import torch.nn as nn


def _make_encoder(features, use_pretrained):
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features)

    return pretrained, scratch

def _make_midas_decoder(features = 256, non_negative = True):
    scratch = _make_scratch([256, 512, 1024, 2048], features)
    scratch.refinenet4 = FeatureFusionBlock(features)
    scratch.refinenet3 = FeatureFusionBlock(features)
    scratch.refinenet2 = FeatureFusionBlock(features)
    scratch.refinenet1 = FeatureFusionBlock(features)

    scratch.output_conv = nn.Sequential(
        nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
        Interpolate(scale_factor=2, mode="bilinear"),
        nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        nn.ReLU(True) if non_negative else nn.Identity(),
    )
    return scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    return scratch



class MidasDecoder(nn.Module):
    """MidasDecoder module.
    """
    def __init__(self, features = 256, non_negative = True):
        super(MidasDecoder, self).__init__()
        # in_shape = [256, 512, 1024, 2048]
        # out_shape = features
        # self.layer1_rn = nn.Conv2d(
        #     in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
        # )
        # self.layer2_rn = nn.Conv2d(
        #     in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
        # )
        # self.layer3_rn = nn.Conv2d(
        #     in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
        # )
        # self.layer4_rn = nn.Conv2d(
        #     in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
        # )

        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

    def forward(self, x, outputs):
        # layer_1_rn = self.layer1_rn(outputs[len(outputs)-4])
        # layer_2_rn = self.layer2_rn(outputs[len(outputs)-3])
        # layer_3_rn = self.layer3_rn(outputs[len(outputs)-2])
        # layer_4_rn = self.layer4_rn(outputs[len(outputs)-1])

        path_4 = self.refinenet4(outputs[len(outputs)-1])
        path_3 = self.refinenet3(path_4, outputs[len(outputs)-3])
        path_2 = self.refinenet2(path_3, outputs[len(outputs)-5])
        path_1 = self.refinenet1(path_2, outputs[len(outputs)-7])

        out = self.output_conv(path_1)

        return torch.squeeze(out, dim=1)



class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output

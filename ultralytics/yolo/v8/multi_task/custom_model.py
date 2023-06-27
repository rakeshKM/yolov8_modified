import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.layers = self._create_layers()

    def _create_layers(self):
        layers = nn.ModuleList()

        # Initial convolutional layer
        layers.append(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        )

        # Downsample blocks
        channels = [32, 64, 128, 256, 512]
        for i in range(5):
            layers.append(
                ConvBlock(channels[i], channels[i] * 2, kernel_size=3, stride=2, padding=1)
            )

        # Extra convolutional layers
        for _ in range(4):
            layers.append(
                ConvBlock(512, 512, kernel_size=1, stride=1, padding=0)
            )
            layers.append(
                ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1)
            )

        return layers

    def forward(self, x):
        outputs = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == 3 or i == 5:
                outputs.append(x)

        return outputs[::-1]


class YOLOBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = ConvBlock(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


# class YOLOv8(nn.Module):
#     def __init__(self, num_classes):
#         super(YOLOv8, self).__init__()
#         self.num_classes = num_classes
#         self.backbone = Darknet53()
#         self.neck = self._create_neck()
#         self.heads = self._create_heads()

#     def _create_neck(self):
#         neck_modules = nn.ModuleList()

#         for _ in range(3):
#             neck_modules.append(
#                 ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
#             )
#             neck_modules.append(
#                 ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
#             )
#             neck_modules.append(
#                 ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
#             )
#             neck_modules.append(
#                 ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
#             )
#             neck_modules.append(
#                 ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
#             )

#         return neck_modules

#     def _create_heads(self):
#         num_anchors = 3
#         heads = nn.ModuleList()

#         for _ in range(3):
#             head = nn.Sequential(
#                 ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
#                 nn.Conv2d(512, num_anchors * (5 + self.num_classes), kernel_size=1, stride=1, padding=0)
#             )
#             heads.append(head)

#         return heads

#     def forward(self, x):
#         outputs = []
#         backbone_out = self.backbone(x)

#         for i in range(4, -1, -1):
#             if i == 4:
#                 x = self.neck[i * 5](backbone_out[i])
#             else:
#                 x = torch.cat((x, backbone_out[i]), dim=1)
#                 x = self.neck[i * 5](x)

#             x = self.neck[i * 5 + 1](x)
#             x = self.neck[i * 5 + 2](x)
#             x = self.neck[i * 5 + 3](x)
#             x = self.neck[i * 5 + 4](x)

#             if i > 0:
#                 outputs.append(self.heads[i - 1](x))

#                 x = self.neck[i * 5 + 5](x)
#                 x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')

#         return outputs[::-1]


# Create an instance of the YOLOv8 model
num_classes = 80  # Number of COCO dataset classes
#model = YOLOv8(num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import MultiScaleRoIAlign

class YOLOv8(nn.Module):
    def __init__(self, num_classes, num_keypoints):
        super(YOLOv8, self).__init__()

        # Backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc

        # Intermediate layers
        self.intermediate_layers = IntermediateLayerGetter(self.backbone, {'layer4': 'feat'})

        # Detection head
        self.num_classes = num_classes
        self.detection_head = nn.Conv2d(2048, (5 + num_classes) * 3, kernel_size=1)

        # Segmentation head
        self.segmentation_head = nn.Conv2d(2048, 1, kernel_size=1)

        # Keypoint detection head
        self.num_keypoints = num_keypoints
        self.keypoint_head = nn.Conv2d(2048, 2 * num_keypoints, kernel_size=1)

        # ROI align
        self.roi_align = MultiScaleRoIAlign(featmap_names=['feat'], output_size=7, sampling_ratio=2)

    def forward(self, images, targets=None):
        # Backbone feature extraction
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.intermediate_layers(x)

        # Detection branch
        detection_output = self.detection_head(features['feat'])

        # Segmentation branch
        segmentation_output = self.segmentation_head(features['feat'])
        segmentation_output = torch.sigmoid(segmentation_output)

        # Keypoint detection branch
        keypoints_output = self.keypoint_head(features['feat'])

        if self.training:
            # During training, compute losses
            detection_loss = compute_detection_loss(detection_output, targets)
            segmentation_loss = compute_segmentation_loss(segmentation_output, targets)
            keypoints_loss = compute_keypoints_loss(keypoints_output, targets)

            return detection_loss, segmentation_loss, keypoints_loss

        # During inference, return outputs
        return detection_output, segmentation_output, keypoints_output


import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from backbone import create_backbone
from ultralytics.nn.modules import  AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv, RTDETRDecoder, Segment


class YOLOv8(nn.Module):
    def __init__(self, num_classes_detection, num_classes_segmentation, num_keypoints):
        super(YOLOv8, self).__init__()

        # Backbone
        self.backbone = create_backbone()

        # Object Detection Head
        self.detect = Detect(in_channels=self.backbone.out_channels[-1],
                             num_classes=num_classes_detection)

        # Image Segmentation Head
        self.segment = Segment(in_channels=self.backbone.out_channels[-1],
                               num_classes=num_classes_segmentation)

        # Keypoint Detection Head
        self.keypoints = HGStem(in_channels=self.backbone.out_channels[-1],
                                num_keypoints=num_keypoints)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # Object Detection Head
        detection_output = self.detect(x)

        # Image Segmentation Head
        segmentation_output = self.segment(x)

        # Keypoint Detection Head
        keypoints_output = self.keypoints(x)

        return detection_output, segmentation_output, keypoints_output


# Example usage:
num_classes_detection = 80
num_classes_segmentation = 21
num_keypoints = 17

model = YOLOv8(num_classes_detection, num_classes_segmentation, num_keypoints)

# Training loop with MSCOCO dataset
dataset = COCODataset(...)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion_detection = nn.CrossEntropyLoss()
criterion_segmentation = nn.CrossEntropyLoss()
criterion_keypoints = nn.MSELoss()

for epoch in range(num_epochs):
    for images, targets in dataloader:
        optimizer.zero_grad()

        # Forward pass
        detection_output, segmentation_output, keypoints_output = model(images)

        # Compute losses
        detection_loss = criterion_detection(detection_output, targets['detection_labels'])
        segmentation_loss = criterion_segmentation(segmentation_output, targets['segmentation_labels'])
        keypoints_loss = criterion_keypoints(keypoints_output, targets['keypoints'])

        # Backward pass
        total_loss = detection_loss + segmentation_loss + keypoints_loss
        total_loss.backward()
        optimizer.step()

        # Print progress
        print(f"Epoch: {epoch+1}, Detection Loss: {detection_loss.item()}, Segmentation Loss: {segmentation_loss.item()}, Keypoints Loss: {keypoints_loss.item()}")

if __name__ =="__main__":

    # Print the model architecture
    model = YOLOv8(1,17)
    print(model)
    out=model(torch.zeros(10, 3, 640, 640))
    print(out[0].shape,out[1].shape,out[2].shape)

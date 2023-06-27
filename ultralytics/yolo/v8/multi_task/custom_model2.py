import torch
import torch.nn as nn

from ultralytics.nn.modules import AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv, RTDETRDecoder, Segment


class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super(YOLOv8Backbone, self).__init__()

        self.layers = nn.ModuleList([
            Conv(3, 64, 3, 2),
            Conv(64, 128, 3, 2),
            C2f(128, True),
            Conv(128, 256, 3, 2),
            C2f(256, True),
            Conv(256, 512, 3, 2),
            C2f(512, True),
            Conv(512, 1024, 3, 2),
            C2f(1024, True),
            SPPF(1024, 5)
        ])

    def forward(self, x):
        outputs = []

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs


class YOLOv8Head(nn.Module):
    def __init__(self, num_classes_detection, num_classes_segmentation, num_keypoints):
        super(YOLOv8Head, self).__init__()

        self.num_classes_detection = num_classes_detection
        self.num_classes_segmentation = num_classes_segmentation
        self.num_keypoints = num_keypoints

        self.layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(dim=1),
            C2f(512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(dim=1),
            C2f(256),
            Conv(256, 256, 3, 2),
            Concat(dim=1),
            C2f(512),
            Conv(512, 512, 3, 2),
            Concat(dim=1),
            C2f(1024),
            Detect([15, 18, 21], self.num_classes_detection),
            Segment([15, 18, 21], self.num_classes_segmentation, 32, 256),
            Pose([15, 18, 21], self.num_classes_detection, [17, 3])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        detection_output = x[0]
        segmentation_output = x[1]
        keypoints_output = x[2]

        return detection_output, segmentation_output, keypoints_output


class YOLOv8(nn.Module):
    def __init__(self, num_classes_detection, num_classes_segmentation, num_keypoints):
        super(YOLOv8, self).__init__()

        self.backbone = YOLOv8Backbone()
        self.head = YOLOv8Head(num_classes_detection, num_classes_segmentation, num_keypoints)

    def forward(self, x):
        backbone_outputs = self.backbone(x)
        detection_output, segmentation_output, keypoints_output = self.head(backbone_outputs)

        return detection_output, segmentation_output, keypoints_output

if __name__ =="__main__":

    # Print the model architecture
    model = YOLOv8(1,1,17)
    print(model)
    out=model(torch.zeros(10, 3, 640, 640))
    print(out[0].shape,out[1].shape,out[2].shape)

'''
# Example usage:
num_classes_detection = 1
num_classes_segmentation = 1
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
'''
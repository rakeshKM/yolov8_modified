from  dataloader import  COCODataset

from torch.utils.data import DataLoader
from default import _C as cfg
import torchvision.transforms as transforms
import cv2
import random
import torch

import asyncio
import contextlib

import sys
#sys.path.append("/data/rakesh/YOLOv8/ultralytics/ultralytics/yolo/utils")

from ultralytics.yolo.utils.plotting import plot_images 
#from plotting import plot_images

# Data loading
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

train_dataset = COCODataset(
        cfg=cfg,
        is_train=True,
        image_size=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
rank=-1

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
    shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
    num_workers=cfg.WORKERS,
    sampler=train_sampler,
    pin_memory=cfg.PIN_MEMORY,
    collate_fn=COCODataset.collate_fn
)
i=0
for item in train_dataset:
    img,target, masks,img_path= item
    print(img.shape)
    #print(classes)
    #print(target)
    bbox_labels, class_label, keypoints_labels = target
    # i=i+1
    # if  labels_out.any():
    #     break

    images =img.unsqueeze(0)
    print(images.shape)
    kpts = keypoints_labels
    cls = class_label.squeeze(-1)
    bboxes = bbox_labels
    paths = img_path
    batch_idx = 0
    ni=1
    plot_images(images,
                batch_idx,
                cls,
                bboxes,
                kpts=kpts,
                paths=paths,
                fname="/data/rakesh/YOLOv8/ultralytics/ultralytics/yolo/v8/multi_task/train_batch_1.jpg" ,
                on_plot=False)
    break
def plot_one_box(bbox_out, image, color=None, label=None, line_thickness=None):
    
    height, width, channels = image.shape

    staff = bbox_out
    x_center, y_center, w, h = float(bbox_out[0])*width, float(bbox_out[1])*height, float(bbox_out[2])*width, float(bbox_out[3])*height
    x1 = round(x_center-w/2)
    y1 = round(y_center-h/2)
    x2 = round(x_center+w/2)
    y2 = round(y_center+h/2)     

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x1), int(y1)) ,(int(x2), int(y2))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    return image
i=0
# for item in train_dataset:
#     img, target, image_name, shapes= item
#     labels_out, seg_label, lane_label =target
#     im_o=cv2.imread(image_name)
#     width,height,c=im_o.shape
#     print(img.shape)
#     print(labels_out)
#     for line in labels_out:
#         bbox = line[2:]
#         print(bbox)
#         imo_o=plot_one_box(bbox, im_o)
#     cv2.imwrite("dataset_test_1_{}.jpg".format(i),im_o)
#     i=i+1
#     if i==10:
#         break

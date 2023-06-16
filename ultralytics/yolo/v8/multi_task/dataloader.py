import torch
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


from default import _C as cfg
from default import update_config
import cv2

class COCODataset(data.Dataset):
    #def __init__(self, root, annotation_file, image_size=(416, 416), is_training=True):
    def __init__(self, cfg, is_train, image_size=(640,640), transform=None):

        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.image_size = image_size
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        #keypoint_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        #self.keypoint_root = keypoint_root / indicator

        self.label_root = self.label_root.iterdir()
        print('database build ')
        self.gt_db=[]
        for label in tqdm(list(self.label_root)[:100]):
            label_path = str(label)
            mask_path = label_path.replace(str('/labels/'), str('/masks/')).replace(".txt", ".jpg") #TODO
            image_path = label_path.replace(str('/labels/'), str('/images/')).replace(".txt", ".jpg") #TODO
            
            with open(label_path, 'r') as f:
                labels = f.readlines()

            bboxes = np.zeros((len(labels), 4))
            classes= np.zeros(len(labels))
            keypoints = np.zeros((len(labels), 17,3))
            for idx, label in enumerate(labels):
                    label = label.split(' ')
                    label= [float(x) for x in label]
                    class_id=label[0]
                    bbox=label[1:5]
                    keypoint=label[5:]
                    bboxes[idx] = bbox                    
                    classes[idx] = class_id
                    keypoints[idx] = np.reshape(keypoint ,(17,3))
            rec = [{
                'image_path': image_path,
                'classes':classes,
                'bboxes': bboxes,
                'keypoints':keypoints,
                'mask_path': mask_path,
            }]
            
            self.gt_db += rec
        print('database build finish')

    def __getitem__(self, index):
        # Load image
        img_path = self.gt_db[index]['image_path']
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #load segmenation
        mask = cv2.imread(self.gt_db[index]['mask_path'], 0)
        _,bin_seg1 = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
        _,inv_bin_seg1 = cv2.threshold(mask,1,255,cv2.THRESH_BINARY_INV)
        seg_label = np.stack((bin_seg1, inv_bin_seg1),2)

        # Load annotations
        annotations_class,annotations_bbox,anotation_keypoints = self.gt_db[index]['classes'],self.gt_db[index]['bboxes'], self.gt_db[index]['keypoints']
        
        target = []
        bbox_labels = torch.zeros((len(annotations_bbox), 5))
        class_label = torch.zeros(len(annotations_bbox))
        keypoints_labels = torch.zeros(len(annotations_bbox),17,3)                            
        for i,bbox in enumerate(annotations_bbox):
                x, y, w, h = bbox
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                bbox_labels[i][0]=i
                bbox_labels[i][1:]=torch.tensor([x1, y1, x2, y2])
                # Extract class label
                class_label[i] = annotations_class[i]

                # Extract keypoints
                keypoints_labels[i] = torch.tensor(anotation_keypoints[i])

        target = [bbox_labels, class_label, keypoints_labels]
        return self.Tensor(img), target,seg_label, img_path

    def __len__(self):
        return len(list(self.label_root))

    @staticmethod
    def collate_fn(batch):
        images, targets ,masks= zip(*batch)
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        targets = [torch.from_numpy(target) for target in targets]
        img_path = torch.stack(img_path, dim=0)
        return images, targets,masks, img_path


def get_dataloader(cfg, is_train, image_size=(640,640), transform=None, batch_size=8, num_workers=0):
    
    dataset = COCODataset(cfg, is_train, image_size=(640,640), transform=None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        collate_fn=COCODataset.collate_fn,
        num_workers=num_workers
    )

    return dataloader

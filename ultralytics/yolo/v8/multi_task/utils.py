# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
from tqdm import tqdm

from ultralytics.nn.autobackend import check_class_names
from ultralytics.yolo.utils import (DATASETS_DIR, LOGGER, NUM_THREADS, ROOT, SETTINGS_YAML, clean_url, colorstr, emojis,
                                    yaml_load)
from ultralytics.yolo.utils.checks import check_file, check_font, is_ascii
from ultralytics.yolo.utils.downloads import download, safe_download, unzip_file
from ultralytics.yolo.utils.ops import segments2boxes

HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
                    assert (lb[:, 5::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert (lb[:, 6::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                else:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb[:, 1:] <= 1).all(), \
                        f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # All labels
                max_cls = int(lb[:, 0].max())  # max label count
                assert max_cls <= num_cls, \
                    f'Label class {max_cls} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros(
                    (0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros((0, 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
                kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
                kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(imgsz, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def check_det_dataset(dataset, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    data = check_file(dataset)

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (zipfile.is_zipfile(data) or is_tarfile(data)):
        new_dir = safe_download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False)
        data = next((DATASETS_DIR / new_dir).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)  # dictionary

    # Checks
    for k in 'train', 'val':
        if k not in data:
            raise SyntaxError(
                emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])

    data['names'] = check_class_names(data['names'])

    # Resolve paths
    path = Path(extract_dir or data.get('path') or Path(data.get('yaml_file', '')).parent)  # dataset root

    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
        data['path'] = path  # download scripts
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()]
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
                r = None  # success
            elif s.startswith('bash '):  # bash script
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}\n')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf')  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset: str, split=''):
    """
    Check a classification dataset such as Imagenet.

    This function takes a `dataset` name as input and returns a dictionary containing information about the dataset.
    If the dataset is not found, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str): Name of the dataset.
        split (str, optional): Dataset split, either 'val', 'test', or ''. Defaults to ''.

    Returns:
        data (dict): A dictionary containing the following keys and values:
            'train': Path object for the directory containing the training set of the dataset
            'val': Path object for the directory containing the validation set of the dataset
            'test': Path object for the directory containing the test set of the dataset
            'nc': Number of classes in the dataset
            'names': List of class names in the dataset
    """
    data_dir = (DATASETS_DIR / dataset).resolve()
    if not data_dir.is_dir():
        LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
        t = time.time()
        if dataset == 'imagenet':
            subprocess.run(f"bash {ROOT / 'yolo/data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip'
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / 'train'
    val_set = data_dir / 'val' if (data_dir / 'val').exists() else None  # data/test or data/val
    test_set = data_dir / 'test' if (data_dir / 'test').exists() else None  # data/val or data/test
    if split == 'val' and not val_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == 'test' and not test_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / 'train').iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))
    return {'train': train_set, 'val': val_set or test_set, 'test': test_set or val_set, 'nc': nc, 'names': names}


class HUBDatasetStats():
    """
    Class for generating HUB dataset JSON and `-hub` dataset directory

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        task:           Dataset task. Options are 'detect', 'segment', 'pose', 'classify'.
        autodownload:   Attempt to download dataset if not found locally

    Usage
        from ultralytics.yolo.data.utils import HUBDatasetStats
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8.zip', task='detect')  # detect dataset
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8-seg.zip', task='segment')  # segment dataset
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8-pose.zip', task='pose')  # pose dataset
        stats.get_json(save=False)
        stats.process_images()
    """

    def __init__(self, path='coco128.yaml', task='detect', autodownload=False):
        """Initialize class."""
        LOGGER.info(f'Starting HUB dataset checks for {path}....')
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            # data = yaml_load(check_yaml(yaml_path))  # data dict
            data = check_det_dataset(yaml_path, autodownload)  # data dict
            if zipped:
                data['path'] = data_dir
        except Exception as e:
            raise Exception('error/HUB/dataset_stats/yaml_load') from e

        self.hub_dir = Path(str(data['path']) + '-hub')
        self.im_dir = self.hub_dir / 'images'
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {'nc': len(data['names']), 'names': list(data['names'].values())}  # statistics dictionary
        self.data = data
        self.task = task  # detect, segment, pose, classify

    @staticmethod
    def _find_yaml(dir):
        """Return data.yaml file."""
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(self, path):
        """Unzip data.zip."""
        if not str(path).endswith('.zip'):  # path is data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), f'Error unzipping {path}, {unzip_dir} not found. ' \
                                   f'path/to/abc.zip MUST unzip to path/to/abc/'
        return True, str(unzip_dir), self._find_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f):
        """Saves a compressed image for HUB previews."""
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    def get_json(self, save=False, verbose=False):
        """Return dataset JSON for Ultralytics HUB."""
        from ultralytics.yolo.data import YOLODataset  # ClassificationDataset

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""
            if self.task == 'detect':
                coordinates = labels['bboxes']
            elif self.task == 'segment':
                coordinates = [x.flatten() for x in labels['segments']]
            elif self.task == 'pose':
                n = labels['keypoints'].shape[0]
                coordinates = np.concatenate((labels['bboxes'], labels['keypoints'].reshape(n, -1)), 1)
            else:
                raise ValueError('Undefined dataset task.')
            zipped = zip(labels['cls'], coordinates)
            return [[int(c), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue

            dataset = YOLODataset(img_path=self.data[split],
                                  data=self.data,
                                  use_segments=self.task == 'segment',
                                  use_keypoints=self.task == 'pose')
            x = np.array([
                np.bincount(label['cls'].astype(int).flatten(), minlength=self.data['nc'])
                for label in tqdm(dataset.labels, total=len(dataset), desc='Statistics')])  # shape(128x80)
            self.stats[split] = {
                'instance_stats': {
                    'total': int(x.sum()),
                    'per_class': x.sum(0).tolist()},
                'image_stats': {
                    'total': len(dataset),
                    'unlabelled': int(np.all(x == 0, 1).sum()),
                    'per_class': (x > 0).sum(0).tolist()},
                'labels': [{
                    Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)]}

        # Save, print and return
        if save:
            stats_path = self.hub_dir / 'stats.json'
            LOGGER.info(f'Saving {stats_path.resolve()}...')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """Compress images for Ultralytics HUB."""
        from ultralytics.yolo.data import YOLODataset  # ClassificationDataset

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in tqdm(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f'{split} images'):
                    pass
        LOGGER.info(f'Done. All images saved to {self.im_dir}')
        return self.im_dir


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the
    Python Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will
    not be resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.
        quality (int, optional): The image compression quality as a percentage. Default is 50%.

    Usage:
        from pathlib import Path
        from ultralytics.yolo.data.utils import compress_one_image
        for f in Path('/Users/glennjocher/Downloads/dataset').rglob('*.jpg'):
            compress_one_image(f)
    """
    try:  # use PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, 'JPEG', quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f'WARNING ⚠️ HUB ops PIL failure {f}: {e}')
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def delete_dsstore(path):
    """
    Deletes all ".DS_store" files under a specified directory.

    Args:
        path (str, optional): The directory path where the ".DS_store" files should be deleted.

    Usage:
        from ultralytics.yolo.data.utils import delete_dsstore
        delete_dsstore('/Users/glennjocher/Downloads/dataset')

    Note:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    # Delete Apple .DS_store files
    files = list(Path(path).rglob('.DS_store'))
    LOGGER.info(f'Deleting *.DS_store files: {files}')
    for f in files:
        f.unlink()


def zip_directory(dir, use_zipfile_library=True):
    """
    Zips a directory and saves the archive to the specified output path.

    Args:
        dir (str): The path to the directory to be zipped.
        use_zipfile_library (bool): Whether to use zipfile library or shutil for zipping.

    Usage:
        from ultralytics.yolo.data.utils import zip_directory
        zip_directory('/Users/glennjocher/Downloads/playground')

        zip -r coco8-pose.zip coco8-pose
    """
    delete_dsstore(dir)
    if use_zipfile_library:
        dir = Path(dir)
        with zipfile.ZipFile(dir.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in dir.glob('**/*'):
                if file_path.is_file():
                    zip_file.write(file_path, file_path.relative_to(dir))
    else:
        import shutil
        shutil.make_archive(dir, 'zip', dir)


# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import urllib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ultralytics import __version__



# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO
DEFAULT_CFG_PATH = ROOT / 'cfg/multitask.yaml'
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'  # global verbose mode
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
LOGGING_NAME = 'ultralytics'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ('Darwin', 'Linux', 'Windows'))  # environment booleans
HELP_MSG = \
    """
    Usage examples for running YOLOv8:

    1. Install the ultralytics package:

        pip install ultralytics

    2. Use the Python SDK:

        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.yaml')  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Use the model
        results = model.train(data="coco128.yaml", epochs=3)  # train the model
        results = model.val()  # evaluate model performance on the validation set
        results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
        success = model.export(format='onnx')  # export the model to ONNX format

    3. Use the command line interface (CLI):

        YOLOv8 'yolo' CLI commands use the following syntax:

            yolo TASK MODE ARGS

            Where   TASK (optional) is one of [detect, segment, classify]
                    MODE (required) is one of [train, val, predict, export]
                    ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                        See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

        - Train a detection model for 10 epochs with an initial learning_rate of 0.01
            yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

        - Predict a YouTube video using a pretrained segmentation model at image size 320:
            yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320

        - Val a pretrained detection model at batch-size 1 and image size 640:
            yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

        - Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
            yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

        - Run special commands:
            yolo help
            yolo checks
            yolo version
            yolo settings
            yolo copy-cfg
            yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Settings
torch.set_printoptions(linewidth=320, precision=4, profile='default')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for deterministic training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress verbose TF compiler warnings in Colab


class SimpleClass:
    """
    Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('_'):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f'{a}: {v.__module__}.{v.__class__.__name__} object'
                else:
                    s = f'{a}: {repr(v)}'
                attr.append(s)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """
    Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml
            """)

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend='Agg'):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Usage:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        callable: Decorated function with temporarily set rc parameters and backend.
    """

    if rcparams is None:
        rcparams = {'font.size': 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


class EmojiFilter(logging.Filter):
    """
    A custom logging filter class for removing emojis in log messages.

    This filter is particularly useful for ensuring compatibility with Windows terminals
    that may not support the display of emojis in log messages.
    """

    def filter(self, record):
        """Filter logs by emoji unicode characters on windows."""
        record.msg = emojis(record.msg)
        return super().filter(record)



def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.

    Returns:
        None: Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a yaml file or a yaml-formatted dictionary.

    Args:
        yaml_file: The file path of the yaml file or a yaml-formatted dictionary.

    Returns:
        None
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


# Default configuration
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        bool: True if running inside a Colab notebook, False otherwise.
    """
    return 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ


def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        bool: True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook.
    Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.
    """
    with contextlib.suppress(Exception):
        from IPython import get_ipython
        return get_ipython() is not None
    return False


def is_docker() -> bool:
    """
    Determine if the script is running inside a Docker container.

    Returns:
        bool: True if the script is running inside a Docker container, False otherwise.
    """
    file = Path('/proc/self/cgroup')
    if file.exists():
        with open(file) as f:
            return 'docker' in f.read()
    else:
        return False


def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    import socket

    for host in '1.1.1.1', '8.8.8.8', '223.5.5.5':  # Cloudflare, Google, AliDNS:
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            # If the connection was successful, close it to avoid a ResourceWarning
            test_connection.close()
            return True
    return False


ONLINE = is_online()


def is_pip_package(filepath: str = __name__) -> bool:
    """
    Determines if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        bool: True if the file is part of a pip package, False otherwise.
    """
    import importlib.util

    # Get the spec for the module
    spec = importlib.util.find_spec(filepath)

    # Return whether the spec is not None and the origin is not None (indicating it is a package)
    return spec is not None and spec.origin is not None


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def is_pytest_running():
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ('PYTEST_CURRENT_TEST' in os.environ) or ('pytest' in sys.modules) or ('pytest' in Path(sys.argv[0]).stem)


def is_github_actions_ci() -> bool:
    """
    Determine if the current environment is a GitHub Actions CI Python runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions CI Python runner, False otherwise.
    """
    return 'GITHUB_ACTIONS' in os.environ and 'RUNNER_OS' in os.environ and 'RUNNER_TOOL_CACHE' in os.environ


def is_git_dir():
    """
    Determines whether the current file is part of a git repository.
    If the current file is not part of a git repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return get_git_dir() is not None


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.

    Returns:
        (Path) or (None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d
    return None  # no .git dir found


def get_git_origin_url():
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str) or (None): The origin URL of the git repository.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'])
            return origin.decode().strip()
    return None  # if not git dir or on error


def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str) or (None): The current git branch name.
    """
    if is_git_dir():
        with contextlib.suppress(subprocess.CalledProcessError):
            origin = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
            return origin.decode().strip()
    return None  # if not git dir or on error


def get_default_args(func):
    """Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        dict: A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_user_config_dir(sub_dir='Ultralytics'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        Path: The path to the user config directory.
    """
    # Return the appropriate config directory for each operating system
    if WINDOWS:
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif LINUX:
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {platform.system()}')

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(str(path.parent)):
        path = Path('/tmp') / sub_dir
        LOGGER.warning(f"WARNING ⚠️ user config directory is not writeable, defaulting to '{path}'.")

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


USER_CONFIG_DIR = Path(os.getenv('YOLO_CONFIG_DIR', get_user_config_dir()))  # Ultralytics settings dir
SETTINGS_YAML = USER_CONFIG_DIR / 'settings.yaml'


def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')."""
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class TryExcept(contextlib.ContextDecorator):
    """YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager."""

    def __init__(self, msg='', verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """Multi-threads a target function and returns thread. Usage: @threaded decorator."""

    def wrapper(*args, **kwargs):
        """Multi-threads a given function and returns the thread."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def set_sentry():
    """
    Initialize the Sentry SDK for error tracking and reporting. Only used if sentry_sdk package is installed and
    sync=True in settings. Run 'yolo settings' to see and update settings YAML file.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)

    The function also configures Sentry SDK to ignore KeyboardInterrupt and FileNotFoundError
    exceptions and to exclude events with 'out of memory' in their exception message.

    Additionally, the function sets custom tags and user information for Sentry events.
    """

    def before_send(event, hint):
        """
        Modify the event before sending it to Sentry based on specific exception types and messages.

        Args:
            event (dict): The event dictionary containing information about the error.
            hint (dict): A dictionary containing additional information about the error.

        Returns:
            dict: The modified event or None if the event should not be sent to Sentry.
        """
        if 'exc_info' in hint:
            exc_type, exc_value, tb = hint['exc_info']
            if exc_type in (KeyboardInterrupt, FileNotFoundError) \
                    or 'out of memory' in str(exc_value):
                return None  # do not send event

        event['tags'] = {
            'sys_argv': sys.argv[0],
            'sys_argv_name': Path(sys.argv[0]).name,
            'install': 'git' if is_git_dir() else 'pip' if is_pip_package() else 'other',
            'os': ENVIRONMENT}
        return event

    if SETTINGS['sync'] and \
            RANK in (-1, 0) and \
            Path(sys.argv[0]).name == 'yolo' and \
            not TESTS_RUNNING and \
            ONLINE and \
            is_pip_package() and \
            not is_git_dir():

        # If sentry_sdk package is not installed then return and do not use Sentry
        try:
            import sentry_sdk  # noqa
        except ImportError:
            return

        sentry_sdk.init(
            dsn='https://5ff1556b71594bfea135ff0203a0d290@o4504521589325824.ingest.sentry.io/4504521592406016',
            debug=False,
            traces_sample_rate=1.0,
            release=__version__,
            environment='production',  # 'dev' or 'production'
            before_send=before_send,
            ignore_errors=[KeyboardInterrupt, FileNotFoundError])
        sentry_sdk.set_user({'id': SETTINGS['uuid']})  # SHA-256 anonymized UUID hash

        # Disable all sentry logging
        for logger in 'sentry_sdk', 'sentry_sdk.errors':
            logging.getLogger(logger).setLevel(logging.CRITICAL)


def get_settings(file=SETTINGS_YAML, version='0.0.3'):
    """
    Loads a global Ultralytics settings YAML file or creates one with default values if it does not exist.

    Args:
        file (Path): Path to the Ultralytics settings YAML file. Defaults to 'settings.yaml' in the USER_CONFIG_DIR.
        version (str): Settings version. If min settings version not met, new default settings will be saved.

    Returns:
        dict: Dictionary of settings key-value pairs.
    """
    import hashlib

    from ultralytics.yolo.utils.checks import check_version
    from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first

    git_dir = get_git_dir()
    root = git_dir or Path()
    datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()
    defaults = {
        'datasets_dir': str(datasets_root / 'datasets'),  # default datasets directory.
        'weights_dir': str(root / 'weights'),  # default weights directory.
        'runs_dir': str(root / 'runs'),  # default runs directory.
        'uuid': hashlib.sha256(str(uuid.getnode()).encode()).hexdigest(),  # SHA-256 anonymized UUID hash
        'sync': True,  # sync analytics to help with YOLO development
        'api_key': '',  # Ultralytics HUB API key (https://hub.ultralytics.com/)
        'settings_version': version}  # Ultralytics settings version

    with torch_distributed_zero_first(RANK):
        if not file.exists():
            yaml_save(file, defaults)
        settings = yaml_load(file)

        # Check that settings keys and types match defaults
        correct = \
            settings \
            and settings.keys() == defaults.keys() \
            and all(type(a) == type(b) for a, b in zip(settings.values(), defaults.values())) \
            and check_version(settings['settings_version'], version)
        if not correct:
            LOGGER.warning('WARNING ⚠️ Ultralytics settings reset to defaults. This is normal and may be due to a '
                           'recent ultralytics package update, but may have overwritten previous settings. '
                           f"\nView and update settings with 'yolo settings' or at '{file}'")
            settings = defaults  # merge **defaults with **settings (prefer **settings)
            yaml_save(file, settings)  # save updated defaults

        return settings


def set_settings(kwargs, file=SETTINGS_YAML):
    """
    Function that runs on a first-time ultralytics package installation to set up global settings and create necessary
    directories.
    """
    SETTINGS.update(kwargs)
    yaml_save(file, SETTINGS)


def deprecation_warn(arg, new_arg, version=None):
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    if not version:
        version = float(__version__[:3]) + 0.2  # deprecate after 2nd major release
    LOGGER.warning(f"WARNING ⚠️ '{arg}' is deprecated and will be removed in 'ultralytics {version}' in the future. "
                   f"Please use '{new_arg}' instead.")


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return urllib.parse.unquote(url).split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


# Run below code on yolo/utils init ------------------------------------------------------------------------------------

# Check first-install steps
PREFIX = colorstr('Ultralytics: ')
SETTINGS = get_settings()
DATASETS_DIR = Path(SETTINGS['datasets_dir'])  # global datasets directory
ENVIRONMENT = 'Colab' if is_colab() else 'Kaggle' if is_kaggle() else 'Jupyter' if is_jupyter() else \
    'Docker' if is_docker() else platform.system()
TESTS_RUNNING = is_pytest_running() or is_github_actions_ci()
set_sentry()

# Apply monkey patches if the script is being run from within the parent directory of the script's location
from patches import imread, imshow, imwrite

# torch.save = torch_save
if Path(inspect.stack()[0].filename).parent.parent.as_posix() in inspect.stack()[-1].filename:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow


# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
from tqdm import tqdm

from ultralytics.nn.autobackend import check_class_names
from ultralytics.yolo.utils import (DATASETS_DIR, LOGGER, NUM_THREADS, ROOT, SETTINGS_YAML, clean_url, colorstr, emojis,
                                    yaml_load)
from ultralytics.yolo.utils.checks import check_file, check_font, is_ascii
from ultralytics.yolo.utils.downloads import download, safe_download, unzip_file
from ultralytics.yolo.utils.ops import segments2boxes

HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
                    assert (lb[:, 5::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert (lb[:, 6::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                else:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb[:, 1:] <= 1).all(), \
                        f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # All labels
                max_cls = int(lb[:, 0].max())  # max label count
                assert max_cls <= num_cls, \
                    f'Label class {max_cls} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros(
                    (0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros((0, 5), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
                kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
                kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(imgsz, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def check_det_dataset(dataset, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    data = check_file(dataset)

    # Download (optional)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (zipfile.is_zipfile(data) or is_tarfile(data)):
        new_dir = safe_download(data, dir=DATASETS_DIR, unzip=True, delete=False, curl=False)
        data = next((DATASETS_DIR / new_dir).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)  # dictionary

    # Checks
    for k in 'train', 'val':
        if k not in data:
            raise SyntaxError(
                emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])

    data['names'] = check_class_names(data['names'])

    # Resolve paths
    path = Path(extract_dir or data.get('path') or Path(data.get('yaml_file', '')).parent)  # dataset root

    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
        data['path'] = path  # download scripts
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()]
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
                r = None  # success
            elif s.startswith('bash '):  # bash script
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:  # python script
                r = exec(s, {'yaml': data})  # return None
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}\n')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf')  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset: str, split=''):
    """
    Check a classification dataset such as Imagenet.

    This function takes a `dataset` name as input and returns a dictionary containing information about the dataset.
    If the dataset is not found, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str): Name of the dataset.
        split (str, optional): Dataset split, either 'val', 'test', or ''. Defaults to ''.

    Returns:
        data (dict): A dictionary containing the following keys and values:
            'train': Path object for the directory containing the training set of the dataset
            'val': Path object for the directory containing the validation set of the dataset
            'test': Path object for the directory containing the test set of the dataset
            'nc': Number of classes in the dataset
            'names': List of class names in the dataset
    """
    data_dir = (DATASETS_DIR / dataset).resolve()
    if not data_dir.is_dir():
        LOGGER.info(f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...')
        t = time.time()
        if dataset == 'imagenet':
            subprocess.run(f"bash {ROOT / 'yolo/data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip'
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / 'train'
    val_set = data_dir / 'val' if (data_dir / 'val').exists() else None  # data/test or data/val
    test_set = data_dir / 'test' if (data_dir / 'test').exists() else None  # data/val or data/test
    if split == 'val' and not val_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == 'test' and not test_set:
        LOGGER.info("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / 'train').iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))
    return {'train': train_set, 'val': val_set or test_set, 'test': test_set or val_set, 'nc': nc, 'names': names}


class HUBDatasetStats():
    """
    Class for generating HUB dataset JSON and `-hub` dataset directory

    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        task:           Dataset task. Options are 'detect', 'segment', 'pose', 'classify'.
        autodownload:   Attempt to download dataset if not found locally

    Usage
        from ultralytics.yolo.data.utils import HUBDatasetStats
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8.zip', task='detect')  # detect dataset
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8-seg.zip', task='segment')  # segment dataset
        stats = HUBDatasetStats('/Users/glennjocher/Downloads/coco8-pose.zip', task='pose')  # pose dataset
        stats.get_json(save=False)
        stats.process_images()
    """

    def __init__(self, path='coco128.yaml', task='detect', autodownload=False):
        """Initialize class."""
        LOGGER.info(f'Starting HUB dataset checks for {path}....')
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            # data = yaml_load(check_yaml(yaml_path))  # data dict
            data = check_det_dataset(yaml_path, autodownload)  # data dict
            if zipped:
                data['path'] = data_dir
        except Exception as e:
            raise Exception('error/HUB/dataset_stats/yaml_load') from e

        self.hub_dir = Path(str(data['path']) + '-hub')
        self.im_dir = self.hub_dir / 'images'
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {'nc': len(data['names']), 'names': list(data['names'].values())}  # statistics dictionary
        self.data = data
        self.task = task  # detect, segment, pose, classify

    @staticmethod
    def _find_yaml(dir):
        """Return data.yaml file."""
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(self, path):
        """Unzip data.zip."""
        if not str(path).endswith('.zip'):  # path is data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), f'Error unzipping {path}, {unzip_dir} not found. ' \
                                   f'path/to/abc.zip MUST unzip to path/to/abc/'
        return True, str(unzip_dir), self._find_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f):
        """Saves a compressed image for HUB previews."""
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    def get_json(self, save=False, verbose=False):
        """Return dataset JSON for Ultralytics HUB."""
        from ultralytics.yolo.data import YOLODataset  # ClassificationDataset

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""
            if self.task == 'detect':
                coordinates = labels['bboxes']
            elif self.task == 'segment':
                coordinates = [x.flatten() for x in labels['segments']]
            elif self.task == 'pose':
                n = labels['keypoints'].shape[0]
                coordinates = np.concatenate((labels['bboxes'], labels['keypoints'].reshape(n, -1)), 1)
            else:
                raise ValueError('Undefined dataset task.')
            zipped = zip(labels['cls'], coordinates)
            return [[int(c), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue

            dataset = YOLODataset(img_path=self.data[split],
                                  data=self.data,
                                  use_segments=self.task == 'segment',
                                  use_keypoints=self.task == 'pose')
            x = np.array([
                np.bincount(label['cls'].astype(int).flatten(), minlength=self.data['nc'])
                for label in tqdm(dataset.labels, total=len(dataset), desc='Statistics')])  # shape(128x80)
            self.stats[split] = {
                'instance_stats': {
                    'total': int(x.sum()),
                    'per_class': x.sum(0).tolist()},
                'image_stats': {
                    'total': len(dataset),
                    'unlabelled': int(np.all(x == 0, 1).sum()),
                    'per_class': (x > 0).sum(0).tolist()},
                'labels': [{
                    Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)]}

        # Save, print and return
        if save:
            stats_path = self.hub_dir / 'stats.json'
            LOGGER.info(f'Saving {stats_path.resolve()}...')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """Compress images for Ultralytics HUB."""
        from ultralytics.yolo.data import YOLODataset  # ClassificationDataset

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in tqdm(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f'{split} images'):
                    pass
        LOGGER.info(f'Done. All images saved to {self.im_dir}')
        return self.im_dir


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the
    Python Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will
    not be resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.
        quality (int, optional): The image compression quality as a percentage. Default is 50%.

    Usage:
        from pathlib import Path
        from ultralytics.yolo.data.utils import compress_one_image
        for f in Path('/Users/glennjocher/Downloads/dataset').rglob('*.jpg'):
            compress_one_image(f)
    """
    try:  # use PIL
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, 'JPEG', quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f'WARNING ⚠️ HUB ops PIL failure {f}: {e}')
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def delete_dsstore(path):
    """
    Deletes all ".DS_store" files under a specified directory.

    Args:
        path (str, optional): The directory path where the ".DS_store" files should be deleted.

    Usage:
        from ultralytics.yolo.data.utils import delete_dsstore
        delete_dsstore('/Users/glennjocher/Downloads/dataset')

    Note:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    # Delete Apple .DS_store files
    files = list(Path(path).rglob('.DS_store'))
    LOGGER.info(f'Deleting *.DS_store files: {files}')
    for f in files:
        f.unlink()


def zip_directory(dir, use_zipfile_library=True):
    """
    Zips a directory and saves the archive to the specified output path.

    Args:
        dir (str): The path to the directory to be zipped.
        use_zipfile_library (bool): Whether to use zipfile library or shutil for zipping.

    Usage:
        from ultralytics.yolo.data.utils import zip_directory
        zip_directory('/Users/glennjocher/Downloads/playground')

        zip -r coco8-pose.zip coco8-pose
    """
    delete_dsstore(dir)
    if use_zipfile_library:
        dir = Path(dir)
        with zipfile.ZipFile(dir.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in dir.glob('**/*'):
                if file_path.is_file():
                    zip_file.write(file_path, file_path.relative_to(dir))
    else:
        import shutil
        shutil.make_archive(dir, 'zip', dir)

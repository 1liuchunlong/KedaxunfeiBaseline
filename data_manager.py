from __future__ import print_function, absolute_import

import glob
import os
import os.path as osp
import re

import numpy as np

from utils import mkdir_if_missing, write_json, read_json
from IPython import embed


class FWreID(object):
    dataset_dir = 'FWreID'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_traindir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_testdir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_testdir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids





    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_traindir(self, dir_path, relabel=False):
        imgs = os.listdir(dir_path)
        pid_container = set()
        dataset = []
        for img in imgs:
            pid = int(img[0:4]) - 95
            pid_container.add(pid)
            dataset.append((dir_path + '/' + img, pid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

    def _process_testdir(self, dir_path, relabel=False):
        imgs = os.listdir(dir_path)
        pid_container = set()
        dataset = []
        for img in imgs:
            pid = int(img[0:4])
            pid_container.add(pid)
            dataset.append((dir_path + '/' + img, pid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


a = FWreID()

import transforms as T

transform_test = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
from torch.utils.data import DataLoader
from dataset_loader import ImageDataset

galleryloader = DataLoader(
    ImageDataset(a.gallery, transform=transform_test),
    batch_size=24, shuffle=False, num_workers=2,
    pin_memory=True, drop_last=False,
)

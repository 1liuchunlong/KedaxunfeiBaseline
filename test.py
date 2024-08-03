import csv
import sys
import time

import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import data_manager
import models
from dataset_loader import ImageDataset
import transforms as T
from utils import AverageMeter, Logger


seed = 1
gpu_devices = 0
root = './data/FWreID'
height = 256
width = 128
train_batch = 24
test_batch = 24
workers = 2
resume = './log/FWreID_ep50.pth.tar'
save_dir = './log'

def main():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pin_memory = True
    else:
        pin_memory = False

    sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))

    if use_gpu:
        print("Currently using GPU {}".format(gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    dataset = data_manager.FWreID()
    transform_train = T.Compose([
        T.Random2DTranslation(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False,
    )
    
    model = models.init_model(name='resnet50', num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))


    print("Loading checkpoint from '{}'".format(resume))
    if use_gpu:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        model = nn.DataParallel(model).cuda()
    else:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model = nn.DataParallel(model)

    test(model, queryloader, galleryloader, use_gpu)


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids = [], []
        for batch_idx, (imgs, pids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids = [], []
        end = time.time()
        for batch_idx, (imgs, pids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)

        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    indices = np.argsort(distmat, axis=1)
    with open('./indices.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(indices)


if __name__ == '__main__':
    main()

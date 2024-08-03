import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
import models
from dataset_loader import ImageDataset
import transforms as T
from models import ResNet
from utils import AverageMeter, Logger, save_checkpoint


seed = 1
gpu_devices = 0
root = './data/FWreID'
height = 256
width = 128
train_batch = 24
test_batch = 24
workers = 2
lR = 0.0002
Weight_decay = 2e-04
stepsize = 10
Gamma = 0.1
startepoch = 0
max_epoch = 50
resume = ''
save_dir = './log'
start_eval = 0
eval_step = -1
print_freq = 10

def main():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        pin_memory = True
    else:
        pin_memory = False

    sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))

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
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=train_batch, shuffle=True, num_workers=workers,
        pin_memory=pin_memory, drop_last=True,
    )

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lR, weight_decay=Weight_decay)
    if stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=Gamma)
    start_epoch = startepoch
    if resume:
        print("Loading checkpoint from '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
    for epoch in range(start_epoch, max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)
        if stepsize > 0: scheduler.step()

        if (epoch + 1) > start_eval and eval_step > 0 and (epoch + 1) % eval_step == 0 or (
                epoch + 1) == max_epoch:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
            }, osp.join(save_dir, 'FWreID_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    for batch_idx, (imgs, pids) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        outputs = model(imgs)
        loss = criterion(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    return 0


if __name__ == '__main__':
    main()

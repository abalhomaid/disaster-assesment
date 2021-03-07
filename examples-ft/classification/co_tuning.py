import random
import time
import warnings
import sys
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../..')
from ftlib.finetune.co_tuning import *
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        ResizeImage(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_dataset = dataset(root=args.root, split='train', sample_rate=args.sample_rate, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)


    determin_train_dataset = dataset(root=args.root, split='train', sample_rate=args.sample_rate, download=True,
                            transform=val_transform)
    determin_train_loader = DataLoader(determin_train_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers, drop_last=False)

    val_dataset = dataset(root=args.root, split='test', sample_rate=100, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__['ResNet50_F'](pretrained=True)
    num_classes = train_dataset.num_classes
    pretrained_head = models.__dict__['ResNet50_C'](pretrained=True)
    classifier = Classifier(backbone, num_classes, source_head=pretrained_head).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(args.lr), momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    milestones = []
    for milestone in args.lr_decay_epochs.split(','):
        milestones.append(int(milestone))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=args.lr_gamma)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return


    # compute relationship

    relationship_path = 'relationship_d.npy'
    # if not os.path.exists(relationship_path):
    if 1 == 1:
        train_source_labels, train_target_labels = get_feature(determin_train_loader, classifier)

        val_source_labels, val_target_labels = get_feature(val_loader, classifier)
        relationship = direct_relationship_learning(train_source_labels, train_target_labels, val_source_labels, val_target_labels)

        np.save(relationship_path, relationship)
    else:
        relationship = np.load(relationship_path)

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_loader, classifier, optimizer,
              epoch, relationship, args)
        lr_scheduler.step()
        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_loader: DataLoader, model: Classifier, optimizer: SGD,
          epoch: int, relationship: dict, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))
    train_iter = iter(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        try:
            x, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, labels = next(train_iter)

        x = x.to(device)
        label = labels.to(device)
        pretrained_targets = torch.from_numpy(relationship[labels]).cuda().float()
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        pretrained_outputs, train_outputs, f = model(x)
        cls_loss = F.cross_entropy(train_outputs, label)

        pretrained_loss = - pretrained_targets * nn.LogSoftmax(dim=-1)(pretrained_outputs)

        pretrained_loss = torch.mean(torch.sum(pretrained_loss, dim=-1))

        loss = cls_loss + args.t * pretrained_loss

        # loss = cls_loss
        cls_acc = accuracy(train_outputs, label)[0]

        losses.update(loss.item(), x.size(0))
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: Classifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            _, output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Baseline for Finetuning')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet50)')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('-sr', '--sample-rate', default=100, type=int,
                        metavar='N',
                        help='sample rate of training dataset (default: 100)')

    parser.add_argument('-t', '--t', default=2.3, type=float,
                        metavar='P', help='weight of pretrained loss')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay-epochs', type=str, default='40, 70', help='where to decay lr, can be a list')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    print(args)
    main(args)


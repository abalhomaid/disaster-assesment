# Domainbed code
class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)
    

class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}
    
class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)

### tllib code ###
        """
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.alignment.coral import CorrelationAlignmentLoss
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import tsne, a_distance


import torch
import torch.nn as nn


class CorrelationAlignmentLoss(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.

    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by

    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))

    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by

    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.


        # TODO: line 100-105 mmd function is same as forward function below
        # TODO: look at cdan.py train function, it loads target x_t and concatenates it with source x_s, same for ys.
        # TODO: apply the same for train function of coral here
    """

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

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
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                                random_color_jitter=True, random_gray_scale=True)
    val_transform = utils.get_val_transform(args.val_resizing)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_dataset, num_classes = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.sources,
                                                   split='train', download=True, transform=train_transform,
                                                   seed=args.seed)
    sampler = utils.RandomDomainSampler(train_dataset, args.batch_size, n_domains_per_batch=args.n_domains_per_batch)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              sampler=sampler, drop_last=True)
    val_dataset, _ = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.sources, split='val',
                                       download=True, transform=val_transform, seed=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_dataset, _ = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.targets, split='test',
                                        download=True, transform=val_transform, seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("train_dataset_size: ", len(train_dataset))
    print('val_dataset_size: ', len(val_dataset))
    print("test_dataset_size: ", len(test_dataset))
    train_iter = ForeverDataIterator(train_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = utils.ImageClassifier(backbone, num_classes, freeze_bn=args.freeze_bn, dropout_p=args.dropout_p,
                                       finetune=args.finetune, pool_layer=pool_layer).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(base_lr=args.lr), args.lr, momentum=args.momentum, weight_decay=args.wd,
                    nesterov=True)
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * args.iters_per_epoch)

    # define loss function
    correlation_alignment_loss = CorrelationAlignmentLoss().to(device)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = utils.collect_feature(val_loader, feature_extractor, device, max_num_features=100)
        target_feature = utils.collect_feature(test_loader, feature_extractor, device, max_num_features=100)
        print(len(source_feature), len(target_feature))
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_val_acc1 = 0.
    best_test_acc1 = 0.
    for epoch in range(args.epochs):

        print(lr_scheduler.get_lr())
        # train for one epoch
        train(train_iter, classifier, optimizer, lr_scheduler, correlation_alignment_loss, args.n_domains_per_batch,
              epoch, args)

        # evaluate on validation set
        print("Evaluate on validation set...")
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_val_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_val_acc1 = max(acc1, best_val_acc1)

        # evaluate on test set
        print("Evaluate on test set...")
        best_test_acc1 = max(best_test_acc1, utils.validate(test_loader, classifier, args, device))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test acc on test set = {}".format(acc1))
    print("oracle acc on test set = {}".format(best_test_acc1))
    logger.close()


def train(train_iter: ForeverDataIterator, model, optimizer, lr_scheduler: CosineAnnealingLR,
          correlation_alignment_loss: CorrelationAlignmentLoss, n_domains_per_batch: int, epoch: int,
          args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    losses_ce = AverageMeter('CELoss', ':3.2f')
    losses_penalty = AverageMeter('Penalty Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, losses_ce, losses_penalty, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_all, labels_all, _ = next(train_iter)
        x_all = x_all.to(device)
        labels_all = labels_all.to(device)

        # compute output
        y_all, f_all = model(x_all)

        # measure data loading time
        data_time.update(time.time() - end)

        # separate into different domains
        y_all = y_all.chunk(n_domains_per_batch, dim=0)
        f_all = f_all.chunk(n_domains_per_batch, dim=0)
        labels_all = labels_all.chunk(n_domains_per_batch, dim=0)

        loss_ce = 0
        loss_penalty = 0
        cls_acc = 0
        for domain_i in range(n_domains_per_batch):
            # cls loss
            y_i, labels_i = y_all[domain_i], labels_all[domain_i]
            loss_ce += F.cross_entropy(y_i, labels_i)
            # update acc
            cls_acc += accuracy(y_i, labels_i)[0] / n_domains_per_batch
            # correlation alignment loss
            for domain_j in range(domain_i + 1, n_domains_per_batch):
                f_i = f_all[domain_i]
                f_j = f_all[domain_j]
                loss_penalty += correlation_alignment_loss(f_i, f_j)

        # normalize loss
        loss_ce /= n_domains_per_batch
        loss_penalty /= n_domains_per_batch * (n_domains_per_batch - 1) / 2

        loss = loss_ce + loss_penalty * args.trade_off

        losses.update(loss.item(), x_all.size(0))
        losses_ce.update(loss_ce.item(), x_all.size(0))
        losses_penalty.update(loss_penalty.item(), x_all.size(0))
        cls_accs.update(cls_acc.item(), x_all.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CORAL for Domain Generalization')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: PACS)')
    parser.add_argument('-s', '--sources', nargs='+', default=None,
                        help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', default=None,
                        help='target domain(s)')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--freeze-bn', action='store_true', help='whether freeze all bn layers')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='only activated when freeze-bn is True')
    # training parameters
    parser.add_argument('--trade-off', default=1, type=float,
                        help='the trade off hyper parameter for correlation alignment loss')
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--n-domains-per-batch', default=3, type=int,
                        help='number of domains in each mini-batch')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='coral',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from common.modules.classifier import Classifier as ClassifierBase


class Classifier(ClassifierBase):
    """Classifier class for Self-Tuning.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, projection_dim=1024, finetune=True, pool_layer=None):
        # TODO: different head initialization
        head = nn.Linear(backbone.out_features, num_classes)
        head.weight.data.normal_(0, 0.01)
        head.bias.data.fill_(0.0)
        super(Classifier, self).__init__(backbone, num_classes=num_classes, head=head, finetune=finetune,
                                         pool_layer=pool_layer)
        self.projector = nn.Linear(backbone.out_features, projection_dim)
        self.projection_dim = projection_dim

    def forward(self, x: torch.Tensor):
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        # projections
        h = self.projector(f)
        h = normalize(h, dim=1)
        # predictions
        predictions = self.head(f)
        if self.training:
            return h, predictions
        else:
            return predictions

    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
            {"params": self.projector.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
        ]

        return params


class SelfTuning(nn.Module):
    """Self-Tuning module
    """

    def __init__(self, encoder_q: nn.DataParallel, encoder_k: nn.DataParallel, num_classes, K=32, m=0.999, T=0.07):
        super(SelfTuning, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.num_classes = num_classes

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue_list", torch.randn(encoder_q.module.projection_dim, K * self.num_classes))
        self.queue_list = normalize(self.queue_list, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, h, label):
        # gather keys before updating queue
        batch_size = h.shape[0]
        ptr = int(self.queue_ptr[label])
        real_ptr = ptr + label * self.K
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_list[:, real_ptr:real_ptr + batch_size] = h.T

        # move pointer
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[label] = ptr

    def forward(self, im_q, im_k, labels):
        batch_size = im_q.size(0)
        device = im_q.device

        # compute query features
        h_q, y_q = self.encoder_q(im_q)  # queries: h_q (N x projection_dim)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            h_k, _ = self.encoder_k(im_k)  # keys: h_k (N x projection_dim)

        # compute logits
        # positive logits: Nx1
        logits_pos = torch.einsum('nl,nl->n', [h_q, h_k]).unsqueeze(-1)  # Einstein sum is more intuitive

        # cur_queue_list: queue_size * class_num
        cur_queue_list = self.queue_list.clone().detach()

        logits_neg_list = torch.Tensor([]).to(device)
        logits_pos_list = torch.Tensor([]).to(device)

        for i in range(batch_size):
            neg_sample = torch.cat([cur_queue_list[:, 0:labels[i] * self.K],
                                    cur_queue_list[:, (labels[i] + 1) * self.K:]],
                                   dim=1)
            pos_sample = cur_queue_list[:, labels[i] * self.K: (labels[i] + 1) * self.K]
            ith_neg = torch.einsum('nl,lk->nk', [h_q[i:i + 1], neg_sample])
            ith_pos = torch.einsum('nl,lk->nk', [h_q[i:i + 1], pos_sample])
            logits_neg_list = torch.cat((logits_neg_list, ith_neg), dim=0)
            logits_pos_list = torch.cat((logits_pos_list, ith_pos), dim=0)
            self._dequeue_and_enqueue(h_k[i:i + 1], labels[i])

        # logits: 1 + queue_size + queue_size * (class_num - 1)
        pgc_logits = torch.cat([logits_pos, logits_pos_list, logits_neg_list], dim=1)
        pgc_logits = nn.LogSoftmax(dim=1)(pgc_logits / self.T)

        pgc_labels = torch.zeros([batch_size, 1 + self.K * self.num_classes]).to(device)
        pgc_labels[:, 0:self.K + 1].fill_(1.0 / (self.K + 1))
        return pgc_logits, pgc_labels, y_q

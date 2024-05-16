"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
import torch
import torch.nn as nn

from tllib.modules.classifier import Classifier as ClassifierBase
from typing import Optional

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
    """

    def __init__(self, gaussian):
        super(CorrelationAlignmentLoss, self).__init__()

        # MMD
        if gaussian:
            self.kernel_type = "gaussian"
        # CORAL
        else:
            self.kernel_type = "mean_cov"

    # MMD
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

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # MMD
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(f_s, f_s).mean()
            Kyy = self.gaussian_kernel(f_t, f_t).mean()
            Kxy = self.gaussian_kernel(f_s, f_t).mean()
            return Kxx + Kyy - 2 * Kxy
        # CORAL
        else:
            mean_s = f_s.mean(0, keepdim=True)
            mean_t = f_t.mean(0, keepdim=True)
            cent_s = f_s - mean_s
            cent_t = f_t - mean_t
            cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
            cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

            mean_diff = (mean_s - mean_t).pow(2).mean()
            cov_diff = (cov_s - cov_t).pow(2).mean()

            return mean_diff + cov_diff

class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

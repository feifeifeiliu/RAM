# requirements:
#    python 3.x
#    torch = 1.1.0
from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import pow
from math import e
# from .triplet_loss import normalize

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class Sigmoid(torch.nn.Module):
    def __init__(self, anneal=0.01, k=0., ta=False):
        """ temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        super(Sigmoid, self).__init__()
        self.temp = anneal if not ta else Parameter(torch.ones(1) * anneal)
        self.k = k

    def forward(self, tensor):
        exponent = -tensor / self.temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent)) + self.k * torch.log(torch.relu(tensor)+1)
        return y


def compute_aff(x, y=None):
    """computes the affinity matrix between an input vector and itself"""
    if y is None:
        return torch.mm(x, x.t())
    else:
        return torch.mm(x, y.t())


def compute_dist(x, type):
    """computes the euclidean distance between an input vector and itself"""
    if type == 'all':
        xb = x.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(xb, xb)
        dist = xx + xx.t()
        dist.addmm_(1, -2, x, x.t())
    elif type == 'pos':
        xb = x.size(1)
        xx = torch.pow(x, 2).sum(2, keepdim=True).expand(-1, xb, xb)
        dist = xx + xx.permute(0, 2, 1)
        dist = dist - 2 * torch.bmm(x, x.permute(0, 2, 1))
        # dist.addbmm_(1, -2, x, x.permute(0, 2, 1))

    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def compute_dist2(x, y, type):
    """computes the euclidean distance between an input vector and itself"""
    xb, n = x.shape[0], y.shape[0]
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(xb, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, xb).t()
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist



class AP_XBM_K2(torch.nn.Module):
    def __init__(self, anneal=0.01, k=0., batch_size=64, num_id=16, feat_dims=2048, top_k=100, ia=False, iam=5, max_epoch=120, power=2.0):
        super(AP_XBM_K2, self).__init__()

        assert(batch_size%num_id==0)

        # self.anneal = anneal
        bs = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims
        self.activition = Sigmoid(anneal=anneal, k=k)
        self.k = top_k
        self.max_epoch = max_epoch
        self.a = anneal
        self.ia = ia
        self.iam = iam
        self.power=power

    def forward(self, inputs_bs, targets_bs, inputs_xb, targets_xb, normalize_feature=False, epoch=1):
        if self.ia:
            self.activition.temp = self.a + self.a * (self.iam-1) * (epoch / self.max_epoch) ** self.power

        if normalize_feature:
            inputs_bs = F.normalize(inputs_bs, dim=1, p=2)
            inputs_xb = F.normalize(inputs_xb, dim=1, p=2)

        bs = inputs_bs.size(0)
        xb = inputs_xb.size(0)
        top_k = min(xb, self.k)

        mask = 1.0 - torch.eye(bs+top_k)
        mask = mask.unsqueeze(dim=0).repeat(bs, 1, 1).cuda()

        sim_bs_bs = compute_aff(inputs_bs)
        sim_bs_xb = compute_aff(inputs_bs, inputs_xb)
        # sim_bs_xb = torch.where(sim_bs_xb>0.9, torch.zeros(sim_bs_xb.size(), device='cuda').half()-1, sim_bs_xb)
        sim_sort, indice = torch.sort(sim_bs_xb, dim=1, descending=True)

        sim_sort = sim_sort[:, :top_k]
        sim = torch.cat([sim_bs_bs, sim_sort], dim=1)
        targets = targets_xb.unsqueeze(dim=0).repeat(bs, 1)
        label = torch.gather(targets, dim=1, index=indice[:, :top_k])
        label_bx = torch.cat([targets_bs.unsqueeze(dim=0).repeat(bs, 1), label], dim=1)

        sim_all_repeat = sim.unsqueeze(dim=1).repeat(1, bs+top_k, 1)

        sim_all_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)

        sim_all_sg = self.activition(sim_all_diff) * mask
        sim_all_rk = torch.sum(sim_all_sg, dim=-1) + 1

        a = targets_bs.unsqueeze(dim=1).repeat(1, bs + top_k) == label_bx
        label_all = \
            a.unsqueeze(dim=2).repeat(1, 1, bs+top_k) == \
            a.unsqueeze(dim=1).repeat(1, bs+top_k, 1)

        pos_all_rk = torch.sum(sim_all_sg * label_all, dim=-1) + 1

        a = (a * torch.cat([1.0 - torch.eye(bs), torch.ones([bs, top_k])], dim=1).cuda()).bool()

        # mean sum
        num = torch.sum(a)
        ap = torch.sum(pos_all_rk[a] / sim_all_rk[a]) / num

        # mean according to num per identity
        # npi = torch.sum(a, dim=1).unsqueeze(dim=1)
        # pos_all_rk = pos_all_rk / npi
        # ap = torch.sum(pos_all_rk[a] / sim_all_rk[a]) / bs

        return 1 - ap

        # weight sum
        # w = (bs+top_k)
        # ap = torch.sum(torch.softmax(pos_all_rk[a].detach()/(w), dim=0) * pos_all_rk[a] / sim_all_rk[a])



def main():
    loss_fn = AP_XBM_K2(0.01, 4, 2, 128)
    imgs = torch.randn([4, 128]).cuda()
    tars = torch.tensor([1, 1,
                         2, 2]).cuda()
    xbm_imgs = torch.randn([3, 128]).cuda()
    xbm_tars = torch.tensor([1, 2, 3]).cuda()
    loss = loss_fn(imgs, tars, xbm_imgs, xbm_tars)


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree

# Import other necessary components from your module
from .utils import *
from .basic_operators import _eps, _inf
from .blocks import *

def get_subscene_label(n, i, stage_list, target, stride, num_classes):
    """
    Function to extract labels for subscene based on the provided indices and stride.
    This is a placeholder and should be adapted to match your actual data structure and needs.
    
    Args:
    - n: stage level (not used in this placeholder)
    - i: index within the stage (not used in this placeholder)
    - stage_list: list or dictionary of data stages (not used in this placeholder)
    - target: tensor of target labels from which to extract subscene labels
    - stride: the stride or step to use when sampling labels
    - num_classes: the number of classes, used for one-hot encoding if necessary

    Returns:
    - torch.Tensor: the sampled or processed labels for the subscene
    """
    # Example: Subsampling labels based on the stride
    labels = target[..., ::stride]  # Subsample labels with the given stride
    return F.one_hot(labels, num_classes=num_classes).float()  # Convert to one-hot encoding


# Replace pointops with CPU-compatible operations
def knnquery(k, points_from, points_to):
    tree = cKDTree(points_from.cpu().numpy())
    dist, indices = tree.query(points_to.cpu().numpy(), k=k)
    return torch.from_numpy(indices).to(points_to.device), torch.from_numpy(dist).to(points_to.device)

def interpolate_features(points_from, points_to, features, k=1):
    tree = cKDTree(points_from.cpu().numpy())
    _, indices = tree.query(points_to.cpu().numpy(), k=k)
    indices = torch.from_numpy(indices).to(points_to.device).long()
    interpolated_features = features[indices].mean(dim=1)
    return interpolated_features

class MultiHead(nn.Module):
    def __init__(self, fdims, head_cfg, config, k):
        super().__init__()
        self.head_cfg = head_cfg
        self.ftype = get_ftype(head_cfg.ftype)[0]

        num_layers = config.num_layers
        infer_list = nn.ModuleList()
        ni_list = []

        for n, i in parse_stage(head_cfg.stage, num_layers):
            func = MLP(fdims[i], head_cfg, config, self.ftype)
            infer_list.append(func)
            ni_list.append((n, i))

        self.infer_list = infer_list
        self.ni_list = ni_list

        if head_cfg.combine.startswith('concat'):
            fdim = MLP.fkey_to_dims[head_cfg.ftype] * len(ni_list)
            self.comb_ops = torch.cat
        else:
            raise ValueError(f'not supported {head_cfg.combine}')
        
        if head_cfg.combine.endswith('mlp'):
            d = config.base_fdim
            self.cls = nn.Sequential(nn.Linear(fdim, d), nn.BatchNorm1d(d), nn.ReLU(inplace=True), nn.Linear(d, k))
        else:
            self.cls = nn.Linear(fdim, k)

    def upsample(self, stage_n, stage_i, stage_list):
        p, x, o = fetch_pxo(stage_n, stage_i, stage_list, self.ftype)
        if stage_i == 0:
            return x

        p0, _, o0 = fetch_pxo('up', 0, stage_list, self.ftype)
        x = interpolate_features(p, p0, x)
        return x

    def forward(self, stage_list):
        collect_list = []
        for (n, i), func in zip(self.ni_list, self.infer_list):
            rst = func(stage_list[n][i], 'f_out')  # process to desired fdim
            stage_list[n][i][self.ftype] = rst  # store back
            collect_list.append(self.upsample(n, i, stage_list))  # (n, c) - potentially upsampled
        x = self.comb_ops(collect_list, 1)  # combine - NCHW
        x = self.cls(x)
        return x, stage_list

class ContrastHead(nn.Module):
    def __init__(self, head_cfg, config):
        super().__init__()
        self.nsample = torch.tensor(config.nsample)
        self.nstride = torch.tensor(config.nstride)
        self.num_classes = torch.tensor(config.num_classes)
        self.head_cfg = head_cfg
        self.config = config
        self.stages = parse_stage(head_cfg.stage, config.num_layers)
        self.vx_size = [config.voxel_size * 2 ** i for i in range(config.num_layers)]
        self.ftype = get_ftype(head_cfg.ftype)[0]
        self.dist_func = getattr(self, f'dist_{head_cfg.dist}')
        self.posmask_func = getattr(self, f'posmask_{head_cfg.pos}')
        self.contrast_func = getattr(self, f'contrast_{head_cfg.contrast}')
        self.main_contrast = getattr(self, f'{head_cfg.main}_contrast') if 'main' in head_cfg and head_cfg.main else self.point_contrast

        self.temperature = None
        if 'temperature' in head_cfg:
            self.temperature = head_cfg.temperature

        self.project = None
        if 'project' in head_cfg and head_cfg.project:
            self.project = nn.ModuleDict({
                f'{n}{i}': MLPbyOps(head_cfg.project, config.base_fdim * 2 ** i, d_out=config.base_fdim) for n, i in self.stages
            })

    def forward(self, output, target, stage_list):
        loss_list = []
        for n, i in self.stages:
            loss = self.main_contrast(n, i, stage_list, target)
            loss_list.append(loss)
        return loss_list

class ContrastHead(nn.Module):
    def __init__(self, head_cfg, config):
        super().__init__()
        self.nsample = torch.tensor(config.nsample)
        self.nstride = torch.tensor(config.nstride)
        self.num_classes = torch.tensor(config.num_classes)
        self.head_cfg = head_cfg
        self.config = config
        self.stages = parse_stage(head_cfg.stage, config.num_layers)
        self.vx_size = [config.voxel_size * 2 ** i for i in range(config.num_layers)]
        self.ftype = get_ftype(head_cfg.ftype)[0]
        self.dist_func = getattr(self, f'dist_{head_cfg.dist}')
        self.posmask_func = getattr(self, f'posmask_{head_cfg.pos}')
        self.contrast_func = getattr(self, f'contrast_{head_cfg.contrast}')
        self.main_contrast = getattr(self, f'{head_cfg.main}_contrast') if 'main' in head_cfg and head_cfg.main else self.point_contrast
        self.temperature = head_cfg.temperature if 'temperature' in head_cfg else None
        self.project = nn.ModuleDict({
            f'{n}{i}': MLPbyOps(head_cfg.project, config.base_fdim * 2 ** i, d_out=config.base_fdim) for n, i in self.stages
        }) if 'project' in head_cfg and head_cfg.project else None

    def forward(self, output, target, stage_list):
        loss_list = []
        for n, i in self.stages:
            loss = self.main_contrast(n, i, stage_list, target)
            loss_list += [loss]
        return loss_list

    def point_contrast(self, n, i, stage_list, target):
        p, features, o = fetch_pxo(n, i, stage_list, self.ftype)
        if self.project:
            features = self.project[f'{n}{i}'](features)
        nsample = self.nsample[i]
        labels = get_subscene_label(n, i, stage_list, target, self.nstride, self.config.num_classes)
        neighbor_idx, _ = knnquery(nsample, p, p)
        neighbor_idx = neighbor_idx[..., 1:]  # Exclude self in neighbors
        neighbor_label = labels[neighbor_idx].view(-1, nsample - 1, labels.shape[1])
        neighbor_feature = features[neighbor_idx].view(-1, nsample - 1, features.shape[1])
        if 'norm' in self.head_cfg.dist or self.head_cfg.dist == 'cos':
            features = F.normalize(features, dim=-1)
        posmask = self.posmask_cnt(labels, neighbor_label)
        point_mask = (posmask.sum(dim=1) > 0) & (posmask.sum(dim=1) < nsample - 1)
        posmask = posmask[point_mask]
        features = features[point_mask]
        neighbor_feature = neighbor_feature[point_mask]
        dist = self.dist_func(features, neighbor_feature)
        loss = self.contrast_func(dist, posmask)
        loss = torch.mean(loss)
        loss *= float(self.head_cfg.weight[1:])
        return loss

    def dist_l2(self, features, neighbor_feature):
        dist = (features.unsqueeze(1) - neighbor_feature).pow(2).sum(dim=-1).sqrt() + _eps
        return dist

    def dist_kl(self, features, neighbor_feature):
        features = F.log_softmax(features, dim=-1)
        neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        dist = F.kl_div(neighbor_feature, features, reduction='none').sum(-1)
        return dist

    def posmask_cnt(self, labels, neighbor_label):
        labels = labels.argmax(dim=-1, keepdim=True)
        neighbor_label = neighbor_label.argmax(dim=-1)
        return labels == neighbor_label

    def contrast_softnn(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - dist.max(dim=1, keepdim=True)[0]
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)
        if invalid_mask is not None:
            exp *= (1 - invalid_mask)
        pos = torch.sum(exp * posmask, dim=1)
        neg = torch.sum(exp, dim=1)
        loss = -torch.log(pos / neg + _eps)
        return loss

    def contrast_nce(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - dist.max(dim=1, keepdim=True)[0]
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)
        if invalid_mask is not None:
            exp *= (1 - invalid_mask)
        neg = torch.sum(exp * (1 - posmask), dim=1)
        exp /= (exp + neg)
        loss = -torch.log(exp[posmask])
        return loss


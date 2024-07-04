import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree

_inf = 1e9
_eps = 1e-12

def knnquery(k, p_from, p_to, o_from, o_to):
    # Implementation of knn query using scipy's cKDTree
    tree = cKDTree(p_from.cpu().numpy())
    dist, idx = tree.query(p_to.cpu().numpy(), k=k)
    return torch.from_numpy(idx).to(p_to.device), torch.from_numpy(dist).to(p_to.device)

def get_subscene_label(stage_n, stage_i, stage_list, target, nstride, num_classes, **kwargs):
    x = F.one_hot(target, num_classes)  # (n, ncls)
    return get_subscene_features(stage_n, stage_i, stage_list, x, nstride, **kwargs)

def get_subscene_features(stage_n, stage_i, stage_list, x, nstride, kr=None, extend=False, return_neighbor=False):
    if stage_i == 0 and not extend:
        return x.float()

    if kr is None:  # infer from sub-sampling (nstride) as default
        i = 1 if stage_i == 0 and extend else stage_i
        kr = torch.prod(torch.tensor(nstride[:i])).item()

    stage_from = stage_list['up'][0]  # support
    p_from, o_from = stage_from['p_out'], stage_from['offset']

    stage_to = stage_list[stage_n][stage_i]  # query
    p_to, o_to = stage_to['p_out'], stage_to['offset']

    neighbor_idx, _ = knnquery(kr, p_from, p_to, o_from, o_to)  # (m, kr)

    neighbor_idx = neighbor_idx.view(-1).long()
    x = x[neighbor_idx, :].view(p_to.shape[0], kr, x.shape[1])  # (m, kr, ncls)
    x = x.float().mean(dim=-2)  # (m, ncls)

    if return_neighbor:
        return x, neighbor_idx, kr
    return x

def batch_stack(features, offset, return_detail=False):
    shape_tail = features[0].shape[1:]
    offset = torch.cat([offset.new_zeros([1]), offset])
    batches_len = offset[1:] - offset[:-1]
    batches_pad = batches_len.max() - batches_len

    batches_feat = []
    for i, pad_n in enumerate(batches_pad):
        zeros = features.new_zeros([pad_n, *shape_tail])
        batches_feat.append(torch.cat([features[offset[i]: offset[i+1]], zeros], 0))
    batches_feat = torch.stack(batches_feat)  # [B, Nmax, d]

    if return_detail:
        return batches_feat, offset, batches_pad, batches_len
    return batches_feat

def get_boundary_mask(labels, neighbor_label=None, neighbor_idx=None, valid_mask=None, get_plain=False, get_cnt=False):
    labels_shape = labels.shape
    if neighbor_label is None:
        shape = [*neighbor_idx.shape, *labels.shape[1:]]  # [BxN, kr, ncls]
        neighbor_label = labels[neighbor_idx.view(-1).long(), ...].view(shape)

    valid_neighbor = neighbor_label >= 0
    labels = labels.unsqueeze(-1)

    neq = labels != neighbor_label
    neq = torch.logical_and(neq, valid_neighbor)
    if get_cnt:
        bound = torch.sum(neq, dim=-1)
        bound = bound * valid_mask if valid_mask is not None else bound
    else:
        bound = torch.any(neq, dim=-1)
        bound = torch.logical_and(bound, valid_mask) if valid_mask is not None else bound

    if get_plain:
        eq = labels == neighbor_label
        eq = torch.logical_or(eq, torch.logical_not(valid_neighbor))
        plain = torch.all(eq, dim=-1)
        plain = torch.logical_and(plain, valid_mask) if valid_mask is not None else plain
        return bound, plain
    return bound

import itertools
import torch

import numpy as np
from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "motion-xplusplus":
        from data_loaders.humanml.data.dataset import Motion_Xplusplus
        return Motion_Xplusplus
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1, generate=False):
    if generate:
        from data_loaders.tensors import t2m_collate_contrastive
        return t2m_collate_contrastive 
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit", "motion-xplusplus"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, generate=False, cache_path=None, config=None): 
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit", "motion-xplusplus"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive, generate=generate, config=config)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0,
                       device=None, autoregressive=False, use_additional_dataset=False, loader_type='sequential', shuffle=True,
                       drop_last=True, generate=False, config=None, parallel=False):
    # 创建主数据集的 DataLoader
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len,
                          device=device, autoregressive=autoregressive, generate=generate, config=config)
    
    if use_additional_dataset:
       dataset.t2m_dataset.mean = np.load('/data/motion/stats_test/all_mean.npy')
       dataset.t2m_dataset.std = np.load('/data/motion/stats_test/all_std.npy')
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size, generate)
    
    if parallel:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        train_sampler = None

    loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=8, drop_last=drop_last, collate_fn=collate, persistent_workers=True, sampler=train_sampler,
            pin_memory=True
        )

    return loader



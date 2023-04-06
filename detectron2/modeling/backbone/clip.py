#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import math
from os.path import join
import numpy as np
import copy
from functools import partial

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone import FPN
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers.batch_norm import get_norm, FrozenBatchNorm2d
from detectron2.modeling.backbone import Backbone
from detectron2.layers import Conv2d, ShapeSpec, get_norm

from timm import create_model
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from timm.models.resnet import ResNet, Bottleneck
from timm.models.resnet import default_cfgs as default_cfgs_resnet
from timm.models.convnext import ConvNeXt, default_cfgs, checkpoint_filter_fn
from open_clip.transform import image_transform, AugmentationCfg
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


from open_clip import timm_model
import open_clip
import torch.nn as nn
import timm
import os


from timm.models.layers import Mlp, to_2tuple
try:
    # old timm imports < 0.8.1
    from timm.models.layers.attention_pool2d import RotAttentionPool2d
    from timm.models.layers.attention_pool2d import AttentionPool2d as AbsAttentionPool2d
except ImportError:
    # new timm imports >= 0.8.1
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d

from collections import OrderedDict
from open_clip.utils import freeze_batch_norm_2d


from open_clip.factory import load_state_dict, load_checkpoint

from open_clip import factory
import open_clip
__all__ = ["build_clip_backbone"]


class ModTimmModel(nn.Module):
    """ timm model adapter
    # FIXME this adapter is a work in progress, may change in ways that break weight compat
    """

    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool='avg',
            proj='linear',
            proj_bias=False,
            drop=0.,
            drop_path=None,
            pretrained=False,
    ):
        
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")

        self.image_size = to_2tuple(image_size)
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        self.trunk = timm.create_model(model_name, pretrained=pretrained, **timm_kwargs)
        feat_size = self.trunk.default_cfg.get('pool_size', None)
        feature_ndim = 1 if not feat_size else 2
        if pool in ('abs_attn', 'rot_attn'):
            assert feature_ndim == 2
            # if attn pooling used, remove both classifier and default pool
            self.trunk.reset_classifier(0, global_pool='')
        else:
            # reset global pool if pool config set, otherwise leave as network default
            reset_kwargs = dict(global_pool=pool) if pool else {}
            self.trunk.reset_classifier(0, **reset_kwargs)
        prev_chs = self.trunk.num_features

        head_layers = OrderedDict()
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim
        else:
            assert proj, 'projection layer needed if non-attention pooling is used.'

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

    def output_shape(self):
        return {
            'clip': ShapeSpec(
                        channels=self.trunk.num_features,
                    )
        }
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward(self, x):
        x = self.trunk.stem(x)
        x = self.trunk.stages(x)
        return x



class CLIPBackbone(Backbone):

    def __init__(self):
        super().__init__()
        
        ## TODO::: SUPER UGLY !!!CLEANUP!!!! + REduce memory footprint by not loading text encoder to GPU lol
        model_name = os.getenv('MODEL_NAME')
        pretrained = os.getenv('PRETRAINED')

        print(f"using the {model_name} model with {pretrained} pre-training")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir = "/cluster/project/zhang/umarka/.cache")
        
        self._out_features = ["clip"]
        self._out_feature_strides = {"clip": 16}
        self._out_feature_channels = {"clip": self.model.visual.trunk.num_features}

        '''
        ## Create a custom timm model
        self.timmmodel = ModTimmModel(
            model_name = model_name, 
            pretrained =  False, 
            pool =  '', 
            proj = 'linear',
            proj_bias =  False, 
            drop =  0.0, 
            drop_path =  0.1, 
            embed_dim = 512,
            image_size =  224 
        )

        ## Now load the pretrained weights from open_clip

        pretrained_cfg = factory.get_pretrained_cfg(model_name, pretrained)
        if pretrained_cfg:
            checkpoint_path = factory.download_pretrained(pretrained_cfg, cache_dir=".")
        elif os.path.exists(pretrained):
            checkpoint_path = pretrained
        
        ## Yeet the last layer as we need the embedding:
        self.timmmodel = torch.nn.Sequential(*(list(self.timmmodel.children())[:-1]))
        print(self.timmmodel)

        ## TODO: Ugly, clean this up
        k = list(factory.load_state_dict(checkpoint_path).keys())
        k = [x.split("visual.")[1] for x in k if x.startswith("visual")]
        dct = {}
        o_dict = factory.load_state_dict(checkpoint_path)
        for key in k:
            dct[key] = o_dict["visual." + key]

        print(self.timmmodel.load_state_dict(dct))

        self._out_features = ["clip"]
        ## TODO: Is this correct?????
        self._out_feature_strides = {"clip": 16}
        self._out_feature_channels = {"clip": self.timmmodel.trunk.num_features}

        self.preprocess = image_transform(
            self.timmmodel.image_size,
            is_train=True,
            mean=OPENAI_DATASET_MEAN,
            std=OPENAI_DATASET_STD,
        )

        '''

    def forward(self, x):
        #x = self.preprocess(x)
        #features = self.timmmodel.forward(x)

        x = self.model.visual.trunk.stem(x)
        features = self.model.visual.trunk.stages(x)
        return {"clip": features}



size2config = {
    'T': {
        'window_size': 7,
        'embed_dim': 96, 
        'depth': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'drop_path_rate': 0.2,
        'pretrained': 'models/swin_tiny_patch4_window7_224.pth'
    },
    'S': {
        'window_size': 7,
        'embed_dim': 96, 
        'depth': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'drop_path_rate': 0.2,
        'pretrained': 'models/swin_small_patch4_window7_224.pth'
    },
    'B': {
        'window_size': 7,
        'embed_dim': 128, 
        'depth': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'drop_path_rate': 0.3,
        'pretrained': 'models/swin_base_patch4_window7_224.pth'
    },
    'B-22k': {
        'window_size': 7,
        'embed_dim': 128, 
        'depth': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'drop_path_rate': 0.3,
        'pretrained': 'models/swin_base_patch4_window7_224_22k.pth'
    },
    'B-22k-384': {
        'window_size': 12,
        'embed_dim': 128, 
        'depth': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'drop_path_rate': 0.3,
        'pretrained': 'models/swin_base_patch4_window12_384_22k.pth'
    },
    'L-22k': {
        'window_size': 7,
        'embed_dim': 192, 
        'depth': [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        'drop_path_rate': 0.3, # TODO (xingyi): this is unclear
        'pretrained': 'models/swin_large_patch4_window7_224_22k.pth'
    },
    'L-22k-384': {
        'window_size': 12,
        'embed_dim': 192, 
        'depth': [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        'drop_path_rate': 0.3, # TODO (xingyi): this is unclear
        'pretrained': 'models/swin_large_patch4_window12_384_22k.pth'
    }
}

@BACKBONE_REGISTRY.register()
def build_clip_backbone(cfg, input_shape: ShapeSpec):
    """
    """
    return CLIPBackbone()
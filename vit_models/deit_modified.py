from functools import partial

import torch
import torch.nn as nn
import math
from einops import reduce, rearrange
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg

import torch.nn.functional as F

__all__ = [
    "tiny_patch16_224_hierarchical", "small_patch16_224_hierarchical", "base_patch16_224_hierarchical"
]


class TransformerHead(nn.Module):
    expansion = 1

    def __init__(self, token_dim, num_patches=196, num_classes=1000, stride=1):
        super(TransformerHead, self).__init__()

        self.token_dim = token_dim
        self.num_patches = num_patches
        self.num_classes = num_classes

        # To process patches
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)
        self.conv = nn.Conv2d(self.token_dim, self.token_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.token_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or self.token_dim != self.expansion * self.token_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.token_dim, self.expansion * self.token_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * self.token_dim)
            )

        self.token_fc = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, x):
        """
            x : (B, num_patches + 1, D) -> (B, C=num_classes)
        """
        cls_token, patch_tokens = x[:, 0], x[:, 1:]
        size = int(math.sqrt(x.shape[1]))

        patch_tokens = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=size, w=size)  # B, D, H, W
        features = F.relu(self.bn(self.conv(patch_tokens)))
        features = self.bn(self.conv(features))
        features += self.shortcut(patch_tokens)
        features = F.relu(features)
        patch_tokens = F.avg_pool2d(features, 14).view(-1, self.token_dim)
        cls_token = self.token_fc(cls_token)

        out = patch_tokens + cls_token

        return out


class VisionTransformer_hierarchical(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Transformer heads
        self.transformerheads = nn.Sequential(*[
            TransformerHead(self.embed_dim)
            for i in range(11)])

    def forward_features(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        # interpolate patch embeddings
        # dim = x.shape[-1]
        # w0 = w // self.patch_embed.patch_size[0]
        # h0 = h // self.patch_embed.patch_size[1]
        # class_pos_embed = self.pos_embed[:, 0]
        # N = self.pos_embed.shape[1] - 1
        # patch_pos_embed = self.pos_embed[:, 1:]
        # patch_pos_embed = nn.functional.interpolate(
        #     patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        #     scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        #     mode='bicubic',
        # )
        # if w0 != patch_pos_embed.shape[-2]:
        #     helper = torch.zeros(h0)[None, None, None, :].repeat(1, dim, w0 - patch_pos_embed.shape[-2], 1).to(x.device)
        #     patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-2)
        # if h0 != patch_pos_embed.shape[-1]:
        #     helper = torch.zeros(w0)[None, None, :, None].repeat(1, dim, 1, h0 - patch_pos_embed.shape[-1]).to(x.device)
        #     patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-1)
        # patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        # interpolate patch embeddings finish

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Store transformer outputs
        transformerheads_outputs = []

        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx <= 10:
                out = self.norm(x)
                out = self.transformerheads[idx](out)
                transformerheads_outputs.append(out)

        x = self.norm(x)
        return x, transformerheads_outputs

    def forward(self, x):
        x, transformerheads_outputs = self.forward_features(x)
        output = []
        for y in transformerheads_outputs:
            output.append(self.head(y))
        output.append(self.head(x[:, 0]))
        return output

@register_model
def tiny_patch16_224_hierarchical(pretrained=False, index=0, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('pretrained_models/ckpts/heir_tiny_001/model_best.pth.tar',
                                map_location="cpu"
                                )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def small_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('pretrained_models/ckpts/heir_small_001/model_best.pth.tar',
                                map_location="cpu"
                                )
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def base_patch16_224_hierarchical(pretrained=False, **kwargs):
    model = VisionTransformer_hierarchical(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('pretrained_models/ckpts/heir_base_001/model_best.pth.tar',
                                map_location="cpu"
                                )
        model.load_state_dict(checkpoint["state_dict"])
    return model

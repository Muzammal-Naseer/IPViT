from functools import partial

import torch
import torch.nn as nn
import math
from einops import reduce, rearrange
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg

import torch.nn.functional as F

__all__ = [
    "tiny_patch16_224_ensemble", "small_patch16_224_ensemble", "base_patch16_224_ensemble"
]


class FinalHead(nn.Module):
    def __init__(self, token_dim=192):
        super(FinalHead, self).__init__()

        self.token_dim = token_dim
        self.fc = nn.Linear(self.token_dim, self.token_dim)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.fc(x)


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


class VisionTransformerEnsemble(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Transformer heads
        self.transformerheads = nn.Sequential(*[
            TransformerHead(self.embed_dim)
            for i in range(11)])
        self.spatialheads = nn.Sequential(*[FinalHead(self.embed_dim) for _ in range(4)])

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

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

    def forward(self, x, get_average=False):
        x, transformer_heads_outputs = self.forward_features(x)
        final_heads_outputs = [self.head(x) for x in transformer_heads_outputs]
        patches = x[:, 1:, :]
        for idx in range(4):
            final_heads_outputs.append(self.head(self.spatialheads[idx](patches[:, idx * 49:(idx + 1) * 49, :])))
        final_heads_outputs.append(self.head(x[:, 0]))
        if get_average:
            return torch.mean(torch.stack(final_heads_outputs, 0), dim=0)
        return final_heads_outputs


@register_model
def tiny_patch16_224_ensemble(pretrained=False, index=0, **kwargs):
    model = VisionTransformerEnsemble(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('pretrained_models/ckpts/joint_tiny_01/checkpoint.pth.tar',
                                map_location="cpu"
                                )
        model.load_state_dict(checkpoint["state_dict"])

    return model


@register_model
def small_patch16_224_ensemble(pretrained=False, **kwargs):
    model = VisionTransformerEnsemble(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('pretrained_models/ckpts/joint_small_01/checkpoint.pth.tar',
                                map_location="cpu"
                                )
        model.load_state_dict(checkpoint["state_dict"])

    return model


@register_model
def base_patch16_224_ensemble(pretrained=False, **kwargs):
    model = VisionTransformerEnsemble(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load('pretrained_models/ckpts/joint_base_01/checkpoint.pth.tar',
                                map_location="cpu"
                                )
        model.load_state_dict(checkpoint["state_dict"])

    return model


if __name__ == '__main__':
    net = small_patch16_224_ensemble(pretrained=True)

    sample = torch.randn(1, 3, 224, 224)
    pred, _ = net(sample)

    print('Parameters:', sum(p.numel() for p in net.parameters()) / 1000000)
    print(f"Output shape: {pred.shape}")

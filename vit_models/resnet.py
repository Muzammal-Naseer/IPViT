import torch
import torchvision
from torchvision.models.resnet import Bottleneck, load_state_dict_from_url, model_urls


class NewResnet(torchvision.models.ResNet):

    def _forward_impl(self, x, drop_percent=None, drop_layer=0):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if drop_layer == 1:
            mask = torch.rand(x.shape[2:], device=x.device)
            mask = (mask > drop_percent).unsqueeze(0).unsqueeze(0)
            x = x * mask
        x = self.layer1(x)

        if drop_layer == 2:
            mask = torch.rand(x.shape[2:], device=x.device)
            mask = (mask > drop_percent).unsqueeze(0).unsqueeze(0)
            x = x * mask
        x = self.layer2(x)

        if drop_layer == 3:
            mask = torch.rand(x.shape[2:], device=x.device)
            mask = (mask > drop_percent).unsqueeze(0).unsqueeze(0)
            x = x * mask
        x = self.layer3(x)

        if drop_layer == 4:
            mask = torch.rand(x.shape[2:], device=x.device)
            mask = (mask > drop_percent).unsqueeze(0).unsqueeze(0)
            x = x * mask
        x = self.layer4(x)

        if drop_layer == 5:
            mask = torch.rand(x.shape[2:], device=x.device)
            mask = (mask > drop_percent).unsqueeze(0).unsqueeze(0)
            x = x * mask

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x, drop_percent=None, drop_layer=None):
        return self._forward_impl(x, drop_percent=drop_percent, drop_layer=drop_layer)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = NewResnet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def drop_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


if __name__ == '__main__':
    model = drop_resnet50(pretrained=True)
    sample = torch.randn((1, 3, 224, 224))
    out = model(sample, drop_layer=1, drop_percent=0.25)

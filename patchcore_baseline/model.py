from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, Wide_ResNet50_2_Weights, resnet50, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F


class PatchFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: list[str], pretrained: bool) -> None:
        super().__init__()
        if layers != ["layer2", "layer3"]:
            raise ValueError("This baseline currently expects layers = ['layer2', 'layer3'].")

        if backbone == "wide_resnet50_2":
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            model = wide_resnet50_2(weights=weights)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.extractor = create_feature_extractor(model, return_nodes={layer: layer for layer in layers})

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: OrderedDict[str, torch.Tensor] = self.extractor(x)
        layer2 = features["layer2"]
        layer3 = F.interpolate(features["layer3"], size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        patches = torch.cat([layer2, layer3], dim=1)
        patches = F.normalize(patches, p=2, dim=1)
        return patches


def flatten_patch_features(features: torch.Tensor) -> torch.Tensor:
    return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbones import BACKBONES


def create_encoder(backbone):
    try:
        if 'timm_' in backbone:
            backbone = backbone.split('_')[-1]
            timm.create_model(model_name=backbone, pretrained=True)
        else:
            model = BACKBONES[backbone](pretrained=True)
    except RuntimeError or KeyError:
        raise RuntimeError('Specify the correct backbone name. Either one of torchvision backbones, or a timm backbone.'
                           'For timm - add prefix \'timm_\'. For instance, timm_resnet18')

    layers = torch.nn.Sequential(*list(model.children()))
    try:
        potential_last_layer = layers[-1]
        while not isinstance(potential_last_layer, nn.Linear):
            potential_last_layer = potential_last_layer[-1]
    except TypeError:
        raise TypeError('Can\'t find the linear layer of the model')

    # Gradient clipping, applied to the final layer before the embedding topology transform
    clip_value = 1e2  # 1e12 is so large to be "no-op"; 1000 was my good guess; 100 applied in practice.
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    features_dim = potential_last_layer.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])

    return model, features_dim


class SupConModel(nn.Module):
    def __init__(self, backbone='resnet50', projection_dim=128, second_stage=False, num_classes=None, projmode=None):
        super(SupConModel, self).__init__()
        self.encoder, self.features_dim = create_encoder(backbone)
        self.second_stage = second_stage
        self.projection_head = True
        self.projection_dim = projection_dim
        self.embed_dim = projection_dim
        assert projmode in ['torus', 'torul', 'hyprs', 'torusC', 'torusN', 'sphere']
        self.projmode = projmode

        if self.second_stage:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.classifier = nn.Linear(self.features_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.features_dim, self.projection_dim * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.projection_dim * 4, self.projection_dim))

    def use_projection_head(self, mode):
        self.projection_head = mode
        if mode:
            self.embed_dim = self.projection_dim
        else:
            self.embed_dim = self.features_dim

    def forward(self, x, donormalise=True):
        if self.second_stage:
            feat = self.encoder(x).squeeze()
            return self.classifier(feat)
        else:
            feat = self.encoder(x).squeeze()
            if self.projection_head:
                feat = self.head(feat)
            if donormalise:
                if self.projmode in ['hyprs','sphere']:
                    feat = F.normalize(feat, dim=1)
                elif self.projmode in ['torul', 'torusN']:
                    # view the dims as pairwise, and L2-normalise the dim-pairs
                    # NB the L2-norm is thus normalised to (D/2), not 1.
                    (batchsize, ndims) = feat.shape
                    circlesview = feat.view((batchsize, ndims//2, 2))
                    circlesview = F.normalize(circlesview, dim=2)
                    feat = circlesview.view((batchsize, ndims))

            return feat

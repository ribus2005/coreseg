import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationViT(nn.Module):
    def __init__(self, encoder_weight_path, **args):
        super().__init__()
        self.encoder = torch.load(encoder_weight_path, weights_only = False)

        

    def forward(self, x):
        self.feature_maps = []
        with torch.no_grad():
            feature_maps = []
            x = self.model.patch_embed(x)
            for idx, block in enumerate(self.model.blocks):
                x = block(x)
                if idx in [7, 15, 23]:
                    feature_maps.append(torch.einsum('nhwc->nchw', x.view(x.shape[0], 14, 14, 1024)))
            x = self.model.norm(x)
        
        feature_maps = torch.concat(feature_maps, dim = 1) #[B, 3072, 14, 14]

        




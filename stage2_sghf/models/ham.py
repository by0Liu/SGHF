import torch
import torch.nn as nn
import math
from dbat import DBAT
from vit_model import LayerNorm2d


def LSE_Pooling(inputs):
    B, C, H, W = inputs.size()
    lse_r = 6.
    inputs = inputs.view(B, C, -1)
    inputs = lse_r * inputs
    outputs = (torch.logsumexp(inputs, dim=2) - math.log(H * W)) / lse_r
    return outputs


def sparsify_cam(heatmap, method='topk', k=0.35, threshold=None):
    """
    :param heatmap: Tensor [B, 1, H, W]
    :param method: 'topk' or 'threshold'
    :param k: top-k
    :param threshold
    """ 
    B, _, H, W = heatmap.shape
    if method == 'topk':
        flat = heatmap.view(B, -1)
        topk_val, _ = torch.topk(flat, int(k * H * W), dim=1)
        kth = topk_val[:, -1].view(B, 1, 1, 1)
        mask = (heatmap >= kth)
    elif method == 'threshold':
        mask = (heatmap >= threshold)
    else:
        raise ValueError("Unknown method")

    sparse_heatmap = heatmap * mask.float()
    return sparse_heatmap

class new_HAM(nn.Module):
    def __init__(self,channel = 512, num_classes = 14):
        super(new_HAM, self).__init__()
        self.fc_g = nn.Linear(1536, 512, bias=False)   # Triplet  1536 or 1024
        self.query = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, kernel_size=2, stride=2),
            LayerNorm2d(channel),
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, bias = True),
            LayerNorm2d(1024),
        )


        self.TripletAttention = DBAT()
        self.norm = nn.LayerNorm(512)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc_g2 = nn.Linear(512, num_classes, bias=False)


    def forward(self, img, model_resnet, model_vit, cam):  # model_resnet, model_vit
        spar_cam = sparsify_cam(cam)
        img_resnet = model_resnet(img, spar_cam)
        b, _, _, _ = img_resnet.shape
        img_vit = model_vit(img, cam)

        img_resnet = self.query(img_resnet)
        img_vit = self.key(img_vit)
        cat_feature = torch.cat((img_resnet, img_vit),dim=1)

        cat_feature = self.TripletAttention(cat_feature)

        fused_feature = LSE_Pooling(cat_feature)
        cat_feature = self.fc_g(fused_feature)
        cat_feature = self.norm(cat_feature)
        cat_feature = self.act(cat_feature)
        cat_feature = self.dropout(cat_feature)
        cat_feature = self.fc_g2(cat_feature)
        cat_feature = cat_feature.float()

        return cat_feature

def HAM_GI(channel = 512, num_classes = 8):
    model = new_HAM(channel, num_classes)
    return model

def HAM_CXR_14(channel = 512, num_classes = 4):
    model = new_HAM(channel, num_classes)
    return model

def HAM_CXR(channel = 512, num_classes = 14):
    model = new_HAM(channel, num_classes)
    return model

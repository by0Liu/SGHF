"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import torch
import torch.nn as nn
from multihead import MultiHeadAttention2D
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def make_fixed_roi_mask(roi_mask, target_hw, *, mode='soft', floor=0.3, tau=1.0, thr=0.5, eps=1e-6):

    if roi_mask.dim() == 3:
        roi_mask = roi_mask.unsqueeze(1)   # [B,1,Hr,Wr]
    m = F.interpolate(roi_mask.detach(), size=target_hw, mode='bilinear', align_corners=False)
    if mode == 'hard':
        m = (m > thr).float()
        if floor > 0.0:
            m = torch.clamp(m, min=floor, max=1.0)
        return m
    # soft
    mn = m.amin(dim=(-2,-1), keepdim=True)
    mx = m.amax(dim=(-2,-1), keepdim=True)
    m  = (m - mn) / (mx - mn + eps)
    if tau != 1.0:
        m = m.pow(tau)
    m = m.clamp(min=floor, max=1.0)
    return m


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):  # x: [B,C,H,W]
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1).contiguous()  # [B,H,W,C]
        x = self.ln(x)
        return x.permute(0,3,1,2).contiguous()

class ConvStem(nn.Module):
    def __init__(self, embed_dim=128, norm_layer=LayerNorm2d):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, embed_dim//2, 3, stride=2, padding=1, bias=False),  # 224→112
            nn.GELU(),
            norm_layer(embed_dim//2),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1, bias=False),  # 112→56
            norm_layer(embed_dim)
        )
    def forward(self, x):
        return self.proj(x)  # [B, embed_dim, 56, 56]


class Downsample_PatchMerging(nn.Module):
    def __init__(self, c_in, norm=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4*c_in, 2*c_in, bias=False)
        self.norm = norm(4*c_in)
    def forward(self, x):                       # x: [B,C,H,W]
        B,C,H,W = x.shape
        x = x.view(B, C, H//2, 2, W//2, 2)      # 2×2 patches
        x = x.permute(0,2,4,3,5,1).contiguous().view(B, H//2*W//2, 4*C) # [B,N,4C]
        x = self.norm(x)
        x = self.reduction(x)                   # [B,N,2C]
        return x.transpose(1,2).view(B, 2*C, H//2, W//2)



class Mlp(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.0, rank=16, bias=False, use_ln=True):
        super().__init__()
        self.dim  = dim
        self.rank = rank

        self.in_norm = LayerNorm2d(dim) if use_ln else nn.Identity()

        self.U_h = nn.Parameter(torch.randn(1, 1,  1,  rank))  # placeholder, lazy shape set at first forward
        self.V_h = nn.Parameter(torch.randn(1, 1,  rank, 1))
        self.U_w = nn.Parameter(torch.randn(1, 1,  1,  rank))
        self.V_w = nn.Parameter(torch.randn(1, 1,  rank, 1))
        self.alpha_h = nn.Parameter(torch.tensor(0.0))
        self.alpha_w = nn.Parameter(torch.tensor(0.0))

        hidden = int(dim * ffn_expansion_factor)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=bias),
        )

        self.out_norm = LayerNorm2d(dim) if use_ln else nn.Identity()

        self._inited = False

    def _maybe_init_spatial_params(self, H, W, device, dtype):
        if self._inited: return
        r = self.rank
        # H mixing: U_h[H,r], V_h[r,H]
        Uh = torch.empty(H, r, device=device, dtype=dtype)
        Vh = torch.empty(r, H, device=device, dtype=dtype)
        nn.init.xavier_uniform_(Uh); nn.init.xavier_uniform_(Vh)
        # W mixing: U_w[W,r], V_w[r,W]
        Uw = torch.empty(W, r, device=device, dtype=dtype)
        Vw = torch.empty(r, W, device=device, dtype=dtype)
        nn.init.xavier_uniform_(Uw); nn.init.xavier_uniform_(Vw)

        self.U_h = nn.Parameter(Uh)  # [H, r]
        self.V_h = nn.Parameter(Vh)  # [r, H]
        self.U_w = nn.Parameter(Uw)  # [W, r]
        self.V_w = nn.Parameter(Vw)  # [r, W]

        with torch.no_grad():
            self.alpha_h.fill_(0.0)
            self.alpha_w.fill_(0.0)
        self._inited = True

    def forward(self, x):
        """
        x: [B,C,H,W]
        return: [B,C,H,W]
        """
        B,C,H,W = x.shape
        self._maybe_init_spatial_params(H, W, x.device, x.dtype)

        x_in = self.in_norm(x)

        y_h = torch.einsum('b c h w, h r -> b c r w', x_in, self.U_h)
        y_h = torch.einsum('b c r w, r h -> b c h w', y_h, self.V_h)

        y_w = torch.einsum('b c h w, w r -> b c h r', x_in, self.U_w)
        y_w = torch.einsum('b c h r, r w -> b c h w', y_w, self.V_w)

        y = x_in + self.alpha_h * y_h + self.alpha_w * y_w

        y = self.ffn(y)

        out = self.out_norm(x + y)
        return out


class Block(nn.Module):
    """
        Encoder_Block
    """
    def __init__(self, *,
                 dim,
                 mlp_ratio    = 4.,
                 drop_path_ratio = 0.,
                 act_layer    = nn.GELU,
                 norm_layer   = nn.LayerNorm,
                 window_size  = 7,
                 overlap_ratio= 0.5,
                 num_heads    = 4,
                 dim_head     = 32,
                 add_atten    = False,
                 add_roi_bias = False,
                 roi_mode = 'soft',
                 roi_floor=0.3,
                 roi_tau=1.0,
                 roi_thr = 0.5,
                 roi_eps = 1e-6
                 ):
        super(Block, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()   # 是否使用droppath？

        self.norm2 = LayerNorm2d(dim)

        self.mlp = Mlp(dim=dim)

        self.att = add_atten
        self.use_roi  = add_roi_bias
        self.roi_mode = roi_mode
        self.roi_floor= roi_floor
        self.roi_tau  = roi_tau
        self.roi_thr  = roi_thr
        self.roi_eps  = roi_eps
        if self.att:
            self.attn  = MultiHeadAttention2D(dim)

    def forward(self, x, roi_mask=None):
        y = self.norm1(x)  # [B,C,H,W]

        if self.use_roi and (roi_mask is not None):
            m = make_fixed_roi_mask(
                roi_mask, target_hw=y.shape[-2:],
                mode=self.roi_mode, floor=self.roi_floor,
                tau=self.roi_tau, thr=self.roi_thr, eps=self.roi_eps
            )                              # [B,1,H,W]
            y = y * m

        if self.att:
            y = self.attn(y)
        x = x + self.drop_path(y)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):
    def __init__(self, *,
                 img_size  = 224,
                 in_c      = 3,
                 num_classes = 1000,
                 embed_dim = 128,
                 depth     = 4,
                 mlp_ratio = 4.,
                 drop_path_ratio = 0.,
                 act_layer = nn.GELU,
                 norm_layer= nn.LayerNorm,
                 window_sizes   = (8, 4, 7, 7),
                 overlap_ratios = (0.5, 0.5, 0.0, 0.0),
                 num_blocks=(4, 6, 6, 8),
                 num_heads_list = (4, 8, 8, 8),
                 dim_head_list  = (32, 32, 64, 128),
                 embed_layer=ConvStem):

        super(VisionTransformer, self).__init__()
        t_roi_mode = 'soft'  # or 'hard'
        t_roi_floor = 0.3
        t_roi_tau = 1.0
        t_roi_thr = 0.5


        self.patch_embed = embed_layer(embed_dim)
        dims = [embed_dim,                         # 128
                embed_dim*2,                       # 256
                embed_dim*4,                       # 512
                embed_dim*8]                       # 1024

        num_blocks = (4, 6, 6, 8)
        total_blocks = sum(num_blocks)  # 24
        drop_path_ratio = 0.2
        dp_rates = torch.linspace(0, drop_path_ratio, total_blocks).tolist()
        dp_iter = iter(dp_rates)

        self.stage1 = nn.Sequential(*[Block(dim=dims[0], mlp_ratio = mlp_ratio,drop_path_ratio = next(dp_iter),act_layer=act_layer,
                                                    norm_layer = norm_layer, window_size = window_sizes[0],overlap_ratio=overlap_ratios[0],
                                                    num_heads = num_heads_list[0],dim_head = dim_head_list[0]) for i in range(num_blocks[0])])
        self.down1 = Downsample_PatchMerging(dims[0])  #

        self.stage2 = nn.Sequential(*[Block(dim=dims[1], mlp_ratio = mlp_ratio,drop_path_ratio = next(dp_iter),act_layer=act_layer,
                                                    norm_layer = norm_layer, window_size = window_sizes[1],overlap_ratio=overlap_ratios[1],
                                                    num_heads = num_heads_list[1],dim_head = dim_head_list[1]) for i in range(num_blocks[1])])
        self.down2 = Downsample_PatchMerging(dims[1])  #

        self.stage3 = nn.Sequential(*[Block(dim=dims[2], mlp_ratio = mlp_ratio,drop_path_ratio = next(dp_iter),act_layer=act_layer,
                                                    norm_layer = norm_layer, window_size = window_sizes[2],overlap_ratio=overlap_ratios[2],
                                                    num_heads = num_heads_list[2],dim_head = dim_head_list[2],add_roi_bias=True,
                                                    roi_mode=t_roi_mode, roi_floor=t_roi_floor, roi_tau=t_roi_tau, roi_thr=t_roi_thr) for i in range(num_blocks[2])])
        self.down3 = Downsample_PatchMerging(dims[2])  #

        self.stage4 = nn.Sequential(*[Block(dim=dims[3], mlp_ratio = mlp_ratio,drop_path_ratio = next(dp_iter),act_layer=act_layer,
                                                    norm_layer = norm_layer, window_size = window_sizes[3],overlap_ratio=overlap_ratios[3],
                                                    num_heads = num_heads_list[3],dim_head = dim_head_list[3],add_roi_bias=True,add_atten=True,
                                                    roi_mode=t_roi_mode, roi_floor=t_roi_floor, roi_tau=t_roi_tau, roi_thr=t_roi_thr) for i in range(num_blocks[3])])

        self.norm = LayerNorm2d(dims[3])


    def forward_features(self, x, cam):

        x = self.patch_embed(x)

        x = self.stage1(x)

        x = self.down1(x)

        x = self.stage2(x)

        x = self.down2(x)

        for blk in self.stage3:
            x = blk(x, roi_mask=cam)
        x = self.down3(x)

        for blk in self.stage4:
            x = blk(x, roi_mask=cam)
        x = self.norm(x)

        return x

    def forward(self, x, cam):
        x = self.forward_features(x, cam)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



def vit_mws_tiny_224(num_classes=1000):
    return VisionTransformer(
        img_size  = 224,
        in_c      = 3,
        num_classes = num_classes,
        embed_dim = 128,
        depth     = 4,
        window_sizes   = (8, 4, 7, 7),
        overlap_ratios = (0.5, 0.5, 0.0, 0.0),
        num_blocks=(4, 6, 6, 8),
        num_heads_list = (4, 8, 8, 8),
        dim_head_list  = (32, 32, 64, 128),
    )

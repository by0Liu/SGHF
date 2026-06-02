# ubuntu / pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention2D(nn.Module):
    """
    Standard Multi-Head Attention for 2D feature maps.
    Input:  x  -> (B, C, H, W)
    Output: y  -> (B, C_out, H, W)  #  C_out = C
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        out_dim: int | None = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_qk: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm_qk = norm_qk

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, out_dim or dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        HW = H * W

        qkv = self.qkv(x)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, HW, self.head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, HW, self.head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, HW, self.head_dim)

        if self.norm_qk:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if attn_mask is not None:
            mask = (attn_mask <= 0)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        y = y.reshape(B, self.num_heads, HW, self.head_dim).permute(0, 1, 3, 2).reshape(B, C, H, W)

        y = self.proj(y)
        y = self.proj_drop(y)
        return y



if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 256, 14, 14
    x = torch.randn(B, C, H, W)

    mha = MultiHeadAttention2D(dim=C, num_heads=8, out_dim=C, attn_drop=0.0, proj_drop=0.0, norm_qk=False)
    y = mha(x)
    print("input :", x.shape)
    print("output:", y.shape)
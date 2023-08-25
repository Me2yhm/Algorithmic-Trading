from typing import Callable

import torch as t
from einops import rearrange
from torch import einsum, nn, Tensor


class PreNorm(nn.Module):
    """
    进行预归一化，fn在下文中指代Attention，因此该作用为进行归一化后施加注意力
    """

    def __init__(self, dim: int, fn: Callable[[Tensor], Tensor]):
        super().__init__()
        self.norm: nn.Module = nn.LayerNorm(dim)
        self.fn: Callable[[Tensor], Tensor] = fn

    def forward(self, x) -> Tensor:
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Applies the Gaussian Error Linear Units function
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor):
        # if len(x.shape) == 2:
        #     x = t.unsqueeze(x, 0)
        b, n, _, h = *x.shape, self.heads
        # 在给定维度(轴)上将输入张量进行分块, 此处为在最后一维上分成三部分
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # rearrange: 进行维度转换， 此处为将原最后一维拆分后再将维度翻转
        q, k, v = map(
            lambda tensor: rearrange(tensor, "b n (h d) -> b h n d", h=h), qkv
        )

        """
        爱因斯坦求和
        res = zeros([b_num, h_num, i_num, j_num])
        for b in range(b_num):
            for h in range(h_num):
                for i in range(i_num):
                    for j in range(j_num):
                        tmp = 0
                        for d in range(d_num):
                             tmp += q[b,h,i,d] * k[b,h,j,d]
                        res[b, h, i, j] = tmp
        """
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class OCET(nn.Module):
    def __init__(
        self,
        #  / 用来指明函数形参必须使用指定位置参数，不能使用关键字参数的形式
        #  * 用来指明必须使用关键字参数
        *,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.oce = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(4, 1), padding="same"
            ),
            nn.Sigmoid(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=40, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(40),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc = nn.Linear(dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        res = x + self.oce(x)
        res = self.conv1(res)
        # 默认去除所有维度为1的dimension
        res = t.squeeze(res, dim = -1)
        res = self.transformer(res)
        # get the cls token
        res = res[:, 0]

        res = self.fc(res)

        res = t.clip(res, max=1.0, min=0.0)
        return res

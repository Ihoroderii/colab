import torch
from torch import  nn
from torch.nn import functional as F
from attention import SelfAttention

class VIE_AttentionBlock(nn.Module):
    def __init__(self, channels : int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        residue = x

        n, c, h, w = x.shape

        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x += residue

        return x
        


class VIE_ResidualBlock(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_chanels)
        self.conv_1 = nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_chanels)
        self.conv_2 = nn.Conv2d(out_chanels, out_chanels, kernel_size=3, stride=1, padding=1)

        #print(in_chanels,"=",out_chanels)
        #if in_chanels == out_chanels:
        #    self.residual_leyer = nn.Identity()
        #else:
        self.residual_leyer = nn.Conv2d(in_chanels, out_chanels, kernel_size=1, stride=1, padding=0)

    def forward(self, x : torch.Tensor) -> torch.Tensor :

        print("x.shape0 : ", x.shape)
        #`print("x.shape_resid : ", self.residual_leyer(x))
        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        print("x.shape1 : ", x.shape)
        

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        print("x.shape2 : ", x.shape)

        return x + self.residual_leyer(residue)
    







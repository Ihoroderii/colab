import torch
from torch import nn
from torch.nn import functional as F
from decoder import VIE_AttentionBlock, VIE_ResidualBlock

class VIE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(  
            # (Batch, Size, Hight, Weight) -> (Batch, 128, Hight, Weight)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch, 128, Hight, Weight) -> #(Batch, )
            VIE_ResidualBlock(128, 128),

            # (Batch, 128, Hight, Weight) -> #(Batch, )
            VIE_ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            VIE_ResidualBlock(128, 256),

            VIE_ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            VIE_ResidualBlock(256, 512),

            VIE_ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VIE_ResidualBlock(512, 512),

            VIE_ResidualBlock(512, 512),
            
            VIE_ResidualBlock(512, 512),

            VIE_AttentionBlock(512),

            VIE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            #nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=1),

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise : torch.Tensor) -> torch.Tensor:

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdiv = variance.sqrt()

        #x = mean + stdiv * noise

        #x *= 0.18215

        #return x
        if noise is None or noise.shape != mean.shape:
            noise = torch.randn_like(mean)

        return mean + stdiv * noise

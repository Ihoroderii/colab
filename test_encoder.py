# test_encoder.py
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from encoder import VIE_Encoder   # import your class

def test_encoder_forward_pass():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open("cat.jpg").convert("RGB")
    x = transform(img).unsqueeze(0)  # (1,3,256,256)

    model = VIE_Encoder()
# after model is built
    with torch.no_grad():
        # forward to get the shape of mean
        print(x)
        print(list(model.children()))
        partial_model = nn.Sequential(*list(model.children())[:-2])
        tmp = partial_model(x)
        print("Intermediate output shape:", tmp.shape)

        # sample noise matching latent shape
        latent_channels = tmp.shape[1] // 2   # because you split into mean + logvar
        noise = torch.randn(1, latent_channels, tmp.shape[2], tmp.shape[3])

        out = model(x, noise)
        print("Final output shape:", out.shape)



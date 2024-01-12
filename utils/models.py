import torch
from torch import nn
from tqdm import tqdm


class Autoencoders:
    def __init__(self,img_size:int,save_to:str,model:str):
        self.img_size=img_size
        self.save_to=save_to
        self.model=model
        
    class Simple_autoencoder(nn.Module):
        def __init__(self):
            super(Autoencoders.Simple_autoencoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(3,32, kernel_size=3, padding=1),
                nn.Conv2d(32,64, kernel_size=3, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64,32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            )

        def forward(self, x):
            x1 = self.encoder(x)
            x2 = self.decoder(x1)
            return x2

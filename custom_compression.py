import torch
import torch.nn as nn

from my_gfa import MyGFA


class CNNBaseline(nn.Module):
    """
    Simple learned image-compression-style autoencoder.

    This is our baseline:
        Image -> CNN Encoder -> Latent -> CNN Decoder -> Image
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                192, 128, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 3, 4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.encoder(x)
        x_hat = self.decoder(y)
        return x_hat


class GLICPrototype(nn.Module):
    """
    Our custom model:

        CNN Encoder
             ↓
           MyGFA
             ↓
           Latent
             ↓
        CNN Decoder
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.gfa = MyGFA(
            channels=192,
            window_size=8,
            sample_size=16,
            average_degree=16,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                192, 128, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 3, 4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.encoder(x)

        # Our graph-based feature aggregation.
        y = self.gfa(y)

        x_hat = self.decoder(y)

        return x_hat
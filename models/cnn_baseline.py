
import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

        self.activation = nn.LeakyReLU(
            0.1,
            inplace=True
        )

    def forward(self, x):

        # Residual connection
        return self.activation(x + self.block(x))


class CNNAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # ENCODER
        self.encoder = nn.Sequential(

            # 128 x 128 -> 64 x 64
            nn.Conv2d(
                3, 64,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.LeakyReLU(0.1, inplace=True),

            ResidualBlock(64),

            # 64 x 64 -> 32 x 32
            nn.Conv2d(
                64, 128,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.LeakyReLU(0.1, inplace=True),

            ResidualBlock(128),

            ResidualBlock(128)
        )

        # DECODER
        self.decoder = nn.Sequential(

            # 32 x 32 -> 64 x 64
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.1, inplace=True),

            ResidualBlock(64),

            # 64 x 64 -> 128 x 128
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                32, 3,
                kernel_size=3,
                padding=1
            ),

            # Keep reconstructed pixel values between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, x):

        features = self.encoder(x)

        reconstructed = self.decoder(features)

        return reconstructed

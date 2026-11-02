
import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate

from models.cnn_baseline import ResidualBlock


class SNNAutoencoder(nn.Module):

    def __init__(self, time_steps=8, beta=0.9):

        super().__init__()

        self.time_steps = time_steps

        # Initial image feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                3, 64,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Generate input current for LIF neurons
        self.spike_conv = nn.Conv2d(
            64, 128,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # Surrogate gradient
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # LIF neuron
        self.lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            threshold=1.0,
            reset_mechanism="subtract"
        )

        # Image decoder
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.1, inplace=True),

            ResidualBlock(64),

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

            nn.Sigmoid()
        )

    def encode_latent(self, x):

        """
        Convert an image into integer spike counts.

        Output:
            [B, 128, H/4, W/4]

        Every symbol is an integer between 0 and time_steps.
        """

        features = self.feature_extractor(x)

        current = F.relu(
            self.spike_conv(features)
        )

        membrane = torch.zeros_like(current)

        spike_counts = torch.zeros_like(current)

        # Simulate LIF neurons over time
        for step in range(self.time_steps):

            spikes, membrane = self.lif(
                current,
                membrane
            )

            spike_counts = spike_counts + spikes

        return spike_counts

    def decode_latent(self, spike_counts):

        """
        Convert integer spike counts back into
        spike-rate features and reconstruct the image.
        """

        spike_rate = (
            spike_counts.float() / self.time_steps
        )

        reconstructed = self.decoder(spike_rate)

        return reconstructed

    def forward(self, x, return_spikes=False):

        # Generate discrete latent representation
        spike_counts = self.encode_latent(x)

        # Reconstruct image
        reconstructed = self.decode_latent(
            spike_counts
        )

        if return_spikes:

            spike_rate = (
                spike_counts / self.time_steps
            )

            return reconstructed, spike_counts, spike_rate

        return reconstructed

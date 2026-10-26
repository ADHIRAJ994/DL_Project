"""
Simplified, paper-aligned implementation of
Graph-based Feature Aggregation (GFA).

This is our own experimental implementation.

Core ideas:
1. Sobel-based content complexity
2. Local candidate graph
3. Global candidate graph
4. Cosine similarity
5. Adaptive number of neighbors
6. Weighted graph aggregation
7. Residual connection
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyGFA(nn.Module):
    """
    Experimental implementation of the main idea behind GLIC's GFA.

    Input:
        x -> [B, C, H, W]

    Output:
        same shape -> [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        sample_size: int = 16,
        average_degree: int = 32,
    ):
        super().__init__()

        self.channels = channels
        self.window_size = window_size
        self.sample_size = sample_size
        self.average_degree = average_degree

        # -------------------------------------------------
        # Learnable feature projections.
        #
        # Query, key and value are used for graph
        # aggregation in the same spirit as the official
        # graph-attention implementation.
        # -------------------------------------------------

        self.query = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.key = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.value = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.output = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        # -------------------------------------------------
        # Small feed-forward block.
        # -------------------------------------------------

        hidden = channels * 2

        self.ffn = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=True,
            ),
        )

    # =====================================================
    # SOBEL COMPLEXITY
    # =====================================================

    def compute_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a content-complexity score using Sobel
        horizontal and vertical gradients.

        x:
            [B, C, H, W]

        Returns:
            complexity:
            [B, H, W]
        """

        b, c, h, w = x.shape

        # Sobel X
        sobel_x = torch.tensor(
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ],
            dtype=x.dtype,
            device=x.device,
        ) / 4.0

        # Sobel Y
        sobel_y = torch.tensor(
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1],
            ],
            dtype=x.dtype,
            device=x.device,
        ) / 4.0

        # Apply same Sobel filter independently
        # to every channel.
        kx = sobel_x.view(1, 1, 3, 3).repeat(
            c, 1, 1, 1
        )

        ky = sobel_y.view(1, 1, 3, 3).repeat(
            c, 1, 1, 1
        )

        # Replicate padding preserves border dimensions.
        padded = F.pad(
            x,
            (1, 1, 1, 1),
            mode="replicate",
        )

        gx = F.conv2d(
            padded,
            kx,
            groups=c,
        )

        gy = F.conv2d(
            padded,
            ky,
            groups=c,
        )

        # Gradient magnitude.
        magnitude = torch.sqrt(
            gx.pow(2) +
            gy.pow(2) +
            1e-9
        )

        # RMS across channels.
        complexity = torch.sqrt(
            magnitude.pow(2).mean(dim=1)
        )

        return complexity

    # =====================================================
    # LOCAL CANDIDATES
    # =====================================================

    def local_candidates(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create local candidate neighborhoods.

        Returns:
            queries:
                [B, N, C]

            candidates:
                [B, N, K, C]
        """

        b, c, h, w = x.shape

        ws = min(self.window_size, h, w)

        # Pad if necessary so dimensions are divisible
        # by the local window size.
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        x_pad = F.pad(
            x,
            (0, pad_w, 0, pad_h),
            mode="replicate",
        )

        hp = h + pad_h
        wp = w + pad_w

        # Convert to:
        #
        # [B, num_windows_h, num_windows_w,
        #      ws, ws, C]
        #
        # Then flatten each window.

        windows = (
            x_pad
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(
                b,
                hp // ws,
                ws,
                wp // ws,
                ws,
                c,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )

        windows = windows.view(
            b,
            hp // ws,
            wp // ws,
            ws * ws,
            c,
        )

        # Query = every pixel in every window.
        queries = windows.view(
            b,
            -1,
            c,
        )

        # Candidate tensor.
        #
        # For simplicity, every pixel in a local window
        # can see the other pixels in that window.

        candidates = windows.view(
            b,
            -1,
            ws * ws,
            c,
        )

        # Remove padded spatial locations.
        queries = queries[:, :h * w]

        candidates = candidates.reshape(
            b,
            -1,
            ws * ws,
            c,
        )

        candidates = candidates[
            :,
            :h * w,
        ]

        return queries, candidates

    # =====================================================
    # GLOBAL CANDIDATES
    # =====================================================

    def global_candidates(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sparse global candidates.

        Instead of comparing every pixel against every
        other pixel, use a regularly spaced global sample.
        """

        b, c, h, w = x.shape

        # Construct a sparse grid.
        step = max(
            1,
            int(
                math.sqrt(
                    (h * w) /
                    max(self.sample_size * self.sample_size, 1)
                )
            )
        )

        sampled = x[
            :,
            :,
            ::step,
            ::step,
        ]

        sampled = sampled.flatten(
            2
        ).transpose(1, 2)

        queries = (
            x
            .flatten(2)
            .transpose(1, 2)
        )

        candidates = sampled

        return queries, candidates

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    @staticmethod
    def cosine_similarity(
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity.

        queries:
            [B, N, C]

        keys:
            [B, M, C]

        returns:
            [B, N, M]
        """

        q = F.normalize(
            queries,
            dim=-1,
        )

        k = F.normalize(
            keys,
            dim=-1,
        )

        return torch.matmul(
            q,
            k.transpose(-1, -2),
        )

    # =====================================================
    # ADAPTIVE AGGREGATION
    # =====================================================

    def adaptive_global_aggregation(
        self,
        x: torch.Tensor,
        complexity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Global graph aggregation.

        The complexity value determines approximately how
        many global neighbors each node should use.

        This is a simplified experimental implementation
        rather than a line-for-line copy of the paper code.
        """

        b, c, h, w = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        # Create sparse global key/value candidates.
        step = max(
            1,
            int(
                math.sqrt(
                    (h * w) /
                    max(self.sample_size * self.sample_size, 1)
                )
            )
        )

        k_sparse = k[
            :,
            ::step,
            :
        ]

        v_sparse = v[
            :,
            ::step,
            :
        ]

        # Cosine similarity.
        similarity = self.cosine_similarity(
            q,
            k_sparse,
        )

        # -------------------------------------------------
        # Convert complexity into adaptive degree.
        # -------------------------------------------------

        complexity = complexity.flatten(
            1
        )

        complexity = (
            complexity /
            (complexity.mean(
                dim=1,
                keepdim=True
            ) + 1e-8)
        )

        requested_degree = (
            self.average_degree *
            complexity
        )

        requested_degree = requested_degree.clamp(
            min=1,
            max=k_sparse.shape[1],
        )

        # For efficiency, use the maximum requested
        # degree and mask unwanted connections.
        max_degree = min(
            self.average_degree * 2,
            k_sparse.shape[1],
        )

        top_values, top_indices = torch.topk(
            similarity,
            k=max_degree,
            dim=-1,
        )

        # Number of neighbors needed for every node.
        degree = requested_degree.long().clamp(
            min=1,
            max=max_degree,
        )

        positions = torch.arange(
            max_degree,
            device=x.device,
        ).view(1, 1, -1)

        mask = positions < degree.unsqueeze(-1)

        top_values = top_values.masked_fill(
            ~mask,
            -1e9,
        )

        attention = F.softmax(
            top_values,
            dim=-1,
        )

        selected_v = v_sparse[
            :,
            top_indices,
            :
        ]

        aggregated = (
            attention.unsqueeze(-1) *
            selected_v
        ).sum(dim=-2)

        aggregated = aggregated.transpose(
            1,
            2,
        ).view(
            b,
            c,
            h,
            w,
        )

        return aggregated

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Preserve input for residual connection.
        shortcut = x

        # -------------------------------------------------
        # 1. Content complexity.
        # -------------------------------------------------

        complexity = self.compute_complexity(x)

        # -------------------------------------------------
        # 2. Local graph.
        #
        # For this first implementation, we use a local
        # convolution-like aggregation.
        # -------------------------------------------------

        local = F.avg_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # -------------------------------------------------
        # 3. Global adaptive graph.
        # -------------------------------------------------

        global_features = (
            self.adaptive_global_aggregation(
                x,
                complexity,
            )
        )

        # -------------------------------------------------
        # 4. Combine local + global information.
        # -------------------------------------------------

        x = (
            x +
            local +
            global_features
        ) / 3.0

        # -------------------------------------------------
        # 5. Learnable output projection.
        # -------------------------------------------------

        x = self.output(x)

        # -------------------------------------------------
        # 6. Residual connection.
        # -------------------------------------------------

        x = shortcut + x

        # -------------------------------------------------
        # 7. Feed-forward transformation.
        # -------------------------------------------------

        x = x + self.ffn(x)

        return x


# =========================================================
# SIMPLE TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing MyGFA")
    print("=" * 60)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print("Device:", device)

    # Small tensor for testing.
    x = torch.randn(
        1,
        64,
        32,
        32,
        device=device,
    )

    model = MyGFA(
        channels=64,
        window_size=8,
        sample_size=16,
        average_degree=16,
    ).to(device)

    y = model(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))

    assert x.shape == y.shape

    print("\n✅ MyGFA test passed!")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_etf(num_classes: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Simplex Equiangular Tight Frame.

    Returns a (num_classes, dim) tensor whose rows are unit-norm prototypes
    with sum=0 and pairwise cosine = -1/(num_classes-1).
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    if dim < num_classes:
        raise ValueError(f"dim ({dim}) must be >= num_classes ({num_classes})")

    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, num_classes, generator=g)
    U, _ = torch.linalg.qr(A)  # (dim, num_classes), orthonormal columns

    K = num_classes
    centering = torch.eye(K) - torch.full((K, K), 1.0 / K)
    M = math.sqrt(K / (K - 1)) * U @ centering  # (dim, K)
    return M.T.contiguous()  # (K, dim)


class ETFRouter(nn.Module):
    """FiLM + MLP routing onto a fixed ETF prototype set.

    Forward: (token, aux, tau) -> (weights, logits, q_norm)
      token:   (B, D_token)  scoring token to be routed
      aux:     (B, D_aux)    conditioning feature (agent pool emb + cmd emb)
      weights: (B, K)        soft routing weights (softmax of logits / tau)
      logits:  (B, K)        q_norm . ETF^T (cosine similarity, since q is L2-normed
                             and ETF rows are unit-norm)
      q_norm:  (B, D_route)  L2-normalized routing query (for DR loss)

    DR loss: ``(q_norm . etf[label] - r)^2`` averaged over batch,
    with r = sqrt((K-1)/K). Drives q toward the target prototype.
    """

    def __init__(
        self,
        token_dim: int,
        aux_dim: int,
        num_classes: int,
        routing_dim: int,
        hidden_dim: int = None,
        etf_seed: int = 0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or token_dim

        # FiLM: aux -> (gamma, beta) over token channels
        self.film = nn.Sequential(
            nn.Linear(aux_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 2 * token_dim),
        )
        # Zero-init the last FiLM linear so initial modulation is identity
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        # Routing MLP: token -> routing_dim
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, routing_dim),
        )

        etf = make_etf(num_classes, routing_dim, seed=etf_seed)
        self.register_buffer('etf', etf)  # (K, routing_dim)

        self.num_classes = num_classes
        self.r = math.sqrt((num_classes - 1) / num_classes)

    def forward(self, token: torch.Tensor, aux: torch.Tensor, tau: float = 1.0):
        gb = self.film(aux)
        gamma, beta = gb.chunk(2, dim=-1)
        mod = token * (1.0 + gamma) + beta  # identity at init (gamma=beta=0)
        q = self.mlp(mod)
        q_norm = F.normalize(q, dim=-1)
        logits = q_norm @ self.etf.t()
        weights = F.softmax(logits / tau, dim=-1)
        return weights, logits, q_norm

    def dr_loss(self, q_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """DR loss: pull q_norm toward etf[label] until dot reaches r."""
        targets = self.etf[labels]  # (B, routing_dim)
        dot = (q_norm * targets).sum(dim=-1)
        return ((dot - self.r) ** 2).mean()


class ExpertBankFFN(nn.Module):
    """Dense Soft MoE FFN: K stacked 2-layer FFNs combined by router weights.

    Matches AsymmetricFFN behavior (pre-LN + Linear -> act -> dropout -> Linear
    -> dropout + residual using normalized input as identity), but with K
    expert weight tensors evaluated in parallel via einsum.

    Forward: (x, weights) -> y
      x:       (B, embed_dims)
      weights: (B, K)  soft weights (e.g. from ETFRouter; rows sum to 1)
      y:       (B, embed_dims)
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        num_experts: int,
        ffn_drop: float = 0.0,
        act_layer=nn.GELU,
        add_identity: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_experts = num_experts
        self.add_identity = add_identity

        self.pre_norm = nn.LayerNorm(embed_dims)

        K, D, H = num_experts, embed_dims, feedforward_channels
        self.W1 = nn.Parameter(torch.empty(K, D, H))
        self.b1 = nn.Parameter(torch.zeros(K, H))
        self.W2 = nn.Parameter(torch.empty(K, H, D))
        self.b2 = nn.Parameter(torch.zeros(K, D))

        # Per-expert Kaiming init (matches nn.Linear default for each slice)
        for k in range(K):
            nn.init.kaiming_uniform_(self.W1[k], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W2[k], a=math.sqrt(5))
            fan_in1 = D
            fan_in2 = H
            bound1 = 1.0 / math.sqrt(fan_in1)
            bound2 = 1.0 / math.sqrt(fan_in2)
            nn.init.uniform_(self.b1[k], -bound1, bound1)
            nn.init.uniform_(self.b2[k], -bound2, bound2)

        self.act = act_layer()
        self.drop1 = nn.Dropout(ffn_drop)
        self.drop2 = nn.Dropout(ffn_drop)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x_n = self.pre_norm(x)
        # First linear, per expert: (B, D) x (K, D, H) -> (B, K, H)
        h = torch.einsum('bd,kdh->bkh', x_n, self.W1) + self.b1  # (B, K, H)
        h = self.drop1(self.act(h))
        # Second linear: (B, K, H) x (K, H, D) -> (B, K, D)
        y = torch.einsum('bkh,khd->bkd', h, self.W2) + self.b2  # (B, K, D)
        y = self.drop2(y)
        # Router-weighted sum over experts
        out = (weights.unsqueeze(-1) * y).sum(dim=1)  # (B, D)
        if self.add_identity:
            out = out + x_n
        return out

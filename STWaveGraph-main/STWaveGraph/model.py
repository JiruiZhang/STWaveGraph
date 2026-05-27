import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(x) + 1e-8)


class SpectralGraphWaveLayer(nn.Module):
    """
    Damped graph wave propagation with:
    - trainable multi-band damping/time/beta
    - explicit velocity term
    - spectral complement residual

    Wave equation on graph spectral domain:
        d2u/dt2 + alpha du/dt + beta * lambda * u = 0
    """
    def __init__(
        self,
        hidden_dim,
        n_bands=3,
        alpha_init=(0.05, 0.10, 0.20),
        beta_init=(0.8, 1.0, 1.2),
        time_init=(1.0, 1.0, 1.0),
        res_init=1.0,
        eps=1e-6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_bands = n_bands
        self.eps = eps

        alpha_init = self._expand_to_bands(alpha_init, n_bands, default=0.10)
        beta_init = self._expand_to_bands(beta_init, n_bands, default=1.00)
        time_init = self._expand_to_bands(time_init, n_bands, default=1.00)

        self.raw_alpha = nn.Parameter(inverse_softplus(alpha_init))
        self.raw_beta = nn.Parameter(inverse_softplus(beta_init))
        self.raw_time = nn.Parameter(inverse_softplus(time_init))
        self.raw_res_scale = nn.Parameter(inverse_softplus(res_init))

    @staticmethod
    def _expand_to_bands(x, n_bands, default=1.0):
        """
        Expand scalar / short list / tuple to length n_bands.
        """
        if x is None:
            return [default] * n_bands

        if isinstance(x, (float, int)):
            return [float(x)] * n_bands

        x = list(x)
        if len(x) == 0:
            return [default] * n_bands
        if len(x) == n_bands:
            return [float(v) for v in x]
        if len(x) < n_bands:
            last = float(x[-1])
            x = [float(v) for v in x] + [last] * (n_bands - len(x))
            return x
        return [float(v) for v in x[:n_bands]]

    def _positive(self, x):
        return F.softplus(x) + self.eps

    def _band_masks(self, eigvals):
        vals = eigvals.detach()
        k = vals.shape[0]

        if self.n_bands == 1:
            return [torch.ones_like(vals, dtype=torch.bool)]

        order = torch.argsort(vals)

        base = k // self.n_bands
        rem = k % self.n_bands

        masks = []
        start = 0
        for b in range(self.n_bands):
            size = base + (1 if b < rem else 0)

            if size == 0:
                idx = order[-1:]
            else:
                idx = order[start:start + size]
                start += size

            mask = torch.zeros_like(vals, dtype=torch.bool)
            mask[idx] = True
            masks.append(mask)

        return masks

    def forward(self, u0, v0, eigvecs, eigvals):
        """
        u0: [N, D]   initial semantic field
        v0: [N, D]   initial velocity field
        eigvecs: [N, K]
        eigvals: [K]
        """
        U = eigvecs
        lam = eigvals

        u_hat = U.T @ u0
        v_hat = U.T @ v0

        alpha = self._positive(self.raw_alpha).clamp(1e-3, 2.0)    # [B]
        beta = self._positive(self.raw_beta).clamp(1e-3, 5.0)      # [B]
        t = self._positive(self.raw_time).clamp(1e-3, 3.0)         # [B]
        res_scale = self._positive(self.raw_res_scale).clamp(0.0, 2.0)

        wave_hat = torch.zeros_like(u_hat)
        band_masks = self._band_masks(lam)

        for b, mask in enumerate(band_masks):
            if mask.sum() == 0:
                continue

            lam_b = lam[mask]
            alpha_b = alpha[b]
            beta_b = beta[b]
            time_b = t[b]

            delta = 0.5 * alpha_b
            omega2 = beta_b * lam_b - delta * delta

            coeff_u = torch.zeros_like(lam_b)
            coeff_v = torch.zeros_like(lam_b)

            under_mask = omega2 > self.eps
            over_mask = ~under_mask

            if under_mask.any():
                omega = torch.sqrt(torch.clamp(omega2[under_mask], min=self.eps))
                exp_term = torch.exp(torch.clamp(-delta * time_b, min=-60.0, max=20.0))

                coeff_u[under_mask] = exp_term * (
                    torch.cos(omega * time_b) +
                    (delta / (omega + self.eps)) * torch.sin(omega * time_b)
                )
                coeff_v[under_mask] = exp_term * (
                    torch.sin(omega * time_b) / (omega + self.eps)
                )

            if over_mask.any():
                gamma = torch.sqrt(torch.clamp(-omega2[over_mask], min=self.eps))
                r1 = -delta + gamma
                r2 = -delta - gamma

                e1 = torch.exp(torch.clamp(r1 * time_b, min=-60.0, max=20.0))
                e2 = torch.exp(torch.clamp(r2 * time_b, min=-60.0, max=20.0))

                delta_over_gamma = delta / (gamma + self.eps)

                coeff_u[over_mask] = 0.5 * (
                    (1.0 + delta_over_gamma) * e1 +
                    (1.0 - delta_over_gamma) * e2
                )
                coeff_v[over_mask] = 0.5 * (
                    (e1 - e2) / (gamma + self.eps)
                )

            wave_hat[mask] = (
                coeff_u.unsqueeze(1) * u_hat[mask] +
                coeff_v.unsqueeze(1) * v_hat[mask]
            )

        wave_low = U @ wave_hat

        # spectral complement residual
        u_low_plain = U @ u_hat
        spectral_complement = u0 - u_low_plain

        out = wave_low + res_scale * spectral_complement
        return out


class ResidualGraphConv(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_fc = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        neigh = adj @ x
        h = self.self_fc(x) + self.neigh_fc(neigh)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.norm(x + h)


class LocalGraphEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block1 = ResidualGraphConv(hidden_dim, dropout=dropout)
        self.block2 = ResidualGraphConv(hidden_dim, dropout=dropout)

    def forward(self, x, adj):
        h = self.block1(x, adj)
        h = self.block2(h, adj)
        return h


class WaveGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim=32, dropout=0.1, n_bands=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # more flexible than Tanh
        self.velocity_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.wave_layer = SpectralGraphWaveLayer(
            hidden_dim=hidden_dim,
            n_bands=n_bands
        )

        self.local_encoder = LocalGraphEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.skip_gate = nn.Parameter(torch.tensor(0.0))
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward_once(self, feat, adj_local, eigvecs, eigvals):
        h0 = self.input_proj(feat)
        v0 = self.velocity_proj(h0)

        hw = self.wave_layer(h0, v0, eigvecs, eigvals)
        hl = self.local_encoder(h0, adj_local)

        gate = self.fuse_gate(torch.cat([h0, hw, hl], dim=1))

        context_features = gate * hw + (1.0 - gate) * hl
        skip = torch.sigmoid(self.skip_gate)
        h = skip * h0 + (1.0 - skip) * context_features
        h = self.out_norm(h)

        z = F.normalize(self.projector(h), p=2, dim=1)
        rec = self.decoder(h)
        return h, z, rec


class PrototypeClusteringHead(nn.Module):
    def __init__(self, hidden_dim, n_clusters, alpha=1.0, normalize_input=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.normalize_input = normalize_input

        self.prototypes = nn.Parameter(torch.randn(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def initialize_centers(self, centers):
        with torch.no_grad():
            if self.normalize_input:
                centers = F.normalize(centers, p=2, dim=1)
            self.prototypes.copy_(centers)

    def forward(self, h):
        if self.normalize_input:
            h = F.normalize(h, p=2, dim=1)
            proto = F.normalize(self.prototypes, p=2, dim=1)
        else:
            proto = self.prototypes

        dist = torch.sum((h.unsqueeze(1) - proto.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / (torch.sum(q, dim=1, keepdim=True) + 1e-8)
        return q

    @staticmethod
    def target_distribution(q):
        weight = (q ** 2) / (torch.sum(q, dim=0, keepdim=True) + 1e-8)
        return weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)


class WaveGraphSTNet(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_clusters,
        proj_dim=32,
        dropout=0.1,
        n_bands=3
    ):
        super().__init__()
        self.encoder = WaveGraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            dropout=dropout,
            n_bands=n_bands
        )
        self.cluster_head = PrototypeClusteringHead(
            hidden_dim=hidden_dim,
            n_clusters=n_clusters,
            normalize_input=True
        )

    def forward(self, feat1, feat2, adj_local, eigvecs, eigvals):
        h1, z1, rec1 = self.encoder.forward_once(feat1, adj_local, eigvecs, eigvals)
        h2, z2, rec2 = self.encoder.forward_once(feat2, adj_local, eigvecs, eigvals)

        q1 = self.cluster_head(h1)
        q2 = self.cluster_head(h2)

        return h1, h2, z1, z2, rec1, rec2, q1, q2

    def inference(self, feat, adj_local, eigvecs, eigvals):
        h, z, rec = self.encoder.forward_once(feat, adj_local, eigvecs, eigvals)
        q = self.cluster_head(h)
        return h, z, rec, q
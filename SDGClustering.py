import matplotlib
import numpy as np

# Sørg for et interaktivt desktop-vindu (TkAgg / QtAgg) ved kjøring fra terminal/IDE
for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(backend, force=False)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt


class SDGClustering:
    """
    Symplectic Discrepancy-Lattice (SDG) Clustering.
    Implementert for d-dimensjonale data (optimalisert for d = 2, 3).
    """

    def __init__(
        self,
        grid_size: int = 56,
        beta: float = 30.0,
        max_iter: int = 80,
        cfl: float = 0.25,
        tol: float = 1e-3,
        mad_lambda: float = 2.5,
        theta_max_deg: float = 35.0,
    ):
        self.m = grid_size
        self.beta = beta
        self.max_iter = max_iter
        self.cfl = cfl
        self.tol = tol
        self.mad_lambda = mad_lambda
        self.cos_theta_max = np.cos(np.deg2rad(theta_max_deg))

    @staticmethod
    def _dct2_nd(x: np.ndarray) -> np.ndarray:
        res = x.astype(np.float64)
        for axis in range(res.ndim):
            N = res.shape[axis]
            res = np.moveaxis(res, axis, -1)
            v = np.empty_like(res)
            v[..., : (N + 1) // 2] = res[..., 0::2]
            v[..., (N + 1) // 2 :] = res[..., 1::2][..., ::-1]

            V = np.fft.fft(v, axis=-1)
            k = np.arange(N)
            factor = 2.0 * np.exp(-1j * np.pi * k / (2.0 * N))
            dct = np.real(V * factor)
            dct[..., 0] /= np.sqrt(2.0)
            dct *= np.sqrt(1.0 / (2.0 * N))
            res = np.moveaxis(dct, -1, axis)
        return res

    @staticmethod
    def _idct2_nd(x: np.ndarray) -> np.ndarray:
        res = x.astype(np.float64)
        for axis in range(res.ndim):
            N = res.shape[axis]
            res = np.moveaxis(res, axis, -1)
            c = res.copy()
            c[..., 0] *= np.sqrt(2.0)
            c *= np.sqrt(N / 2.0)

            k = np.arange(N)
            W = c * np.exp(1j * np.pi * k / (2.0 * N))
            v = np.real(np.fft.ifft(W, axis=-1))

            out = np.empty_like(res)
            out[..., 0::2] = v[..., : (N + 1) // 2]
            out[..., 1::2] = v[..., (N + 1) // 2 :][..., ::-1]
            res = np.moveaxis(out, -1, axis)
        return res

    def _compute_discrepancy_density(
        self, X_norm: np.ndarray, coords_1d: np.ndarray
    ) -> np.ndarray:
        n, d = X_norm.shape
        factors = []
        for j in range(d):
            z = self.beta * (coords_1d[None, :] - X_norm[:, j : j + 1])
            z = np.clip(z, -40.0, 40.0)
            sig = 1.0 / (1.0 + np.exp(-z))
            dsig = self.beta * sig * (1.0 - sig)
            factors.append(dsig)

        if d == 2:
            density = (factors[0].T @ factors[1]) / n
        elif d == 3:
            density = (
                np.einsum(
                    "ni,nj,nk->ijk",
                    factors[0],
                    factors[1],
                    factors[2],
                    optimize=True,
                )
                / n
            )
        else:
            density = np.zeros((self.m,) * d, dtype=np.float64)
            for idx in np.ndindex((self.m,) * d):
                prod = np.ones(n, dtype=np.float64)
                for j in range(d):
                    prod *= factors[j][:, idx[j]]
                density[idx] = np.mean(prod)

        density /= np.mean(density)
        return density - 1.0

    def _solve_poisson_neumann(self, rho: np.ndarray) -> np.ndarray:
        d = rho.ndim
        h = 1.0 / self.m
        rho_hat = self._dct2_nd(rho)

        grids = np.meshgrid(*[np.arange(self.m) for _ in range(d)], indexing="ij")
        eigvals = np.zeros_like(rho_hat)
        for k_grid in grids:
            eigvals += (2.0 * np.cos(np.pi * k_grid / self.m) - 2.0) / (h * h)

        psi_hat = np.zeros_like(rho_hat)
        mask = np.abs(eigvals) > 1e-12
        psi_hat[mask] = -rho_hat[mask] / eigvals[mask]
        psi_hat[tuple([0] * d)] = 0.0

        return self._idct2_nd(psi_hat)

    def _gradient_grid(self, psi: np.ndarray) -> np.ndarray:
        h = 1.0 / self.m
        grads = np.gradient(psi, h)
        if psi.ndim == 1:
            grads = [grads]
        return -np.stack(grads, axis=-1)

    def _sample_vector_field(
        self, field: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        d = points.shape[-1]
        h = 1.0 / self.m
        idx_float = np.clip((points - 0.5 * h) / h, 0.0, self.m - 1.0 - 1e-9)
        idx0 = np.floor(idx_float).astype(np.int64)
        frac = idx_float - idx0

        out = np.zeros_like(points)
        for corner in np.ndindex((2,) * d):
            weight = np.ones(points.shape[:-1], dtype=np.float64)
            gather_idx = []
            for dim, bit in enumerate(corner):
                w_dim = frac[..., dim] if bit == 1 else (1.0 - frac[..., dim])
                weight *= w_dim
                gather_idx.append(idx0[..., dim] + bit)
            out += weight[..., None] * field[tuple(gather_idx)]
        return out

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if d > 4:
            raise ValueError(
                f"SDG-clustering på tett gitter krever d <= 4 (fikk d={d})."
            )

        mins = X.min(axis=0)
        spans = np.maximum(X.max(axis=0) - mins, 1e-12)
        X_norm = 0.08 + 0.84 * (X - mins) / spans

        h = 1.0 / self.m
        coords_1d = (np.arange(self.m) + 0.5) * h
        mesh = np.meshgrid(*[coords_1d for _ in range(d)], indexing="ij")
        G0 = np.stack(mesh, axis=-1)
        G = G0.copy()

        rho = self._compute_discrepancy_density(X_norm, coords_1d)
        psi = self._solve_poisson_neumann(rho)
        vel_field = self._gradient_grid(psi)

        for _ in range(self.max_iter):
            v = self._sample_vector_field(vel_field, G)
            v_max = np.max(np.linalg.norm(v, axis=-1))
            if v_max < self.tol * h:
                break
            eta = self.cfl * h / max(v_max, 1e-12)
            G = np.clip(G + eta * v, 0.5 * h, 1.0 - 0.5 * h)

        J = np.zeros((self.m,) * d + (d, d), dtype=np.float64)
        for k in range(d):
            J[..., :, k] = np.gradient(G, h, axis=k)

        det_J = np.linalg.det(J)
        kappa = 1.0 / np.maximum(det_J, 1e-6)

        med = np.median(kappa)
        mad = np.median(np.abs(kappa - med))
        tau = med + self.mad_lambda * max(mad, 1e-6)
        active_mask = kappa > tau

        _, eigvecs = np.linalg.eigh(np.swapaxes(J, -1, -2) @ J)
        v_min = eigvecs[..., :, 0]

        cluster_grid = np.full((self.m,) * d, -1, dtype=np.int32)
        visited = np.zeros((self.m,) * d, dtype=bool)
        cluster_id = 0

        offsets = []
        for dim in range(d):
            for step in (-1, 1):
                off = [0] * d
                off[dim] = step
                offsets.append(tuple(off))

        for start_idx in np.ndindex((self.m,) * d):
            if not active_mask[start_idx] or visited[start_idx]:
                continue

            queue = [start_idx]
            visited[start_idx] = True
            cluster_grid[start_idx] = cluster_id

            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                v1 = v_min[curr]

                for off in offsets:
                    nxt = tuple(curr[k] + off[k] for k in range(d))
                    if any(nxt[k] < 0 or nxt[k] >= self.m for k in range(d)):
                        continue
                    if visited[nxt] or not active_mask[nxt]:
                        continue

                    v2 = v_min[nxt]
                    if abs(np.dot(v1, v2)) >= self.cos_theta_max:
                        visited[nxt] = True
                        cluster_grid[nxt] = cluster_id
                        queue.append(nxt)

            cluster_id += 1

        G_flat = G.reshape(-1, d)
        grid_labels_flat = cluster_grid.reshape(-1)
        labels = np.full(n, -1, dtype=np.int32)

        for i in range(n):
            dists_sq = np.sum((G_flat - X_norm[i]) ** 2, axis=1)
            labels[i] = grid_labels_flat[np.argmin(dists_sq)]

        self.X_norm_ = X_norm
        self.grid_init_ = G0
        self.grid_ = G
        self.rho_ = rho
        self.det_J_ = det_J
        self.kappa_ = kappa
        self.tau_ = tau
        self.cluster_grid_ = cluster_grid
        return labels


def show_sdg_window(X: np.ndarray, model: SDGClustering, labels: np.ndarray) -> None:
    """Åpner et eget interaktivt Matplotlib-vindu som visualiserer SDG-Clustering."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), num="SDG-Clustering Visualisering")
    fig.patch.set_facecolor("#111317")

    X_norm = model.X_norm_
    G = model.grid_
    m = model.m

    # Panel 1: Rådata og ortogonalt startgitter G_0
    ax0 = axes[0, 0]
    ax0.set_title("1. Normaliserte data & startgitter $G_0$", color="white", fontsize=11)
    step = max(1, m // 28)
    for i in range(0, m, step):
        ax0.plot(model.grid_init_[i, :, 0], model.grid_init_[i, :, 1], color="#3a414e", lw=0.6)
        ax0.plot(model.grid_init_[:, i, 0], model.grid_init_[:, i, 1], color="#3a414e", lw=0.6)
    ax0.scatter(X_norm[:, 0], X_norm[:, 1], s=14, color="#4cc9f0", alpha=0.85, edgecolors="none")

    # Panel 2: Symplektisk deformert gitter G_T
    ax1 = axes[0, 1]
    ax1.set_title("2. Deformert gitter $G_T$ (likevekt)", color="white", fontsize=11)
    for i in range(m):
        ax1.plot(G[i, :, 0], G[i, :, 1], color="#4a5568", lw=0.55, alpha=0.85)
        ax1.plot(G[:, i, 0], G[:, i, 1], color="#4a5568", lw=0.55, alpha=0.85)
    ax1.scatter(X_norm[:, 0], X_norm[:, 1], s=10, color="#f72585", alpha=0.65, edgecolors="none")

    # Panel 3: Kompresjonsindeks log10(kappa_i)
    ax2 = axes[1, 0]
    ax2.set_title(
        f"3. Kompresjonsfelt $\\log_{{10}}(\\kappa_i)$ (terskel $\\tau={model.tau_:.2f}$)",
        color="white",
        fontsize=11,
    )
    im = ax2.pcolormesh(
        G[..., 0],
        G[..., 1],
        np.log10(model.kappa_),
        shading="auto",
        cmap="magma",
    )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Panel 4: Klyngeresultat etter egenvektor-filtrert BFS
    ax3 = axes[1, 1]
    ax3.set_title("4. Ekstraherte klynger & støy (-1)", color="white", fontsize=11)
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax3.scatter(
            X[noise_mask, 0],
            X[noise_mask, 1],
            s=18,
            color="#6c757d",
            marker="x",
            label="Støy (-1)",
        )

    cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
    cmap_pts = plt.get_cmap("tab10")
    for idx, cid in enumerate(cluster_ids):
        mask = labels == cid
        ax3.scatter(
            X[mask, 0],
            X[mask, 1],
            s=22,
            color=cmap_pts(idx % 10),
            label=f"Klynge {cid} (n={np.sum(mask)})",
            alpha=0.9,
        )
    leg = ax3.legend(loc="upper right", facecolor="#1b1f27", edgecolor="#3a414e", fontsize=9)
    for text in leg.get_texts():
        text.set_color("white")

    for ax in axes.flat:
        ax.set_facecolor("#181b22")
        ax.tick_params(colors="#a0aec0")
        for spine in ax.spines.values():
            spine.set_color("#3a414e")

    min_det = float(np.min(model.det_J_))
    fig.suptitle(
        f"Symplectic Discrepancy-Lattice (SDG) Clustering  |  min det(J) = {min_det:.4f}",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Generer to kryssende sinuskurver + bakgrunnsstøy
    t1 = np.linspace(-1.0, 1.0, 250)
    c1 = np.column_stack([t1, 0.6 * np.sin(np.pi * t1)]) + rng.normal(0, 0.04, (250, 2))

    t2 = np.linspace(-1.0, 1.0, 250)
    c2 = np.column_stack([0.6 * np.sin(np.pi * t2), t2]) + rng.normal(0, 0.04, (250, 2))

    noise = rng.uniform(-1.2, 1.2, (40, 2))
    X_sample = np.vstack([c1, c2, noise])

    model = SDGClustering(
        grid_size=56,
        beta=30.0,
        max_iter=80,
        mad_lambda=2.5,
        theta_max_deg=35.0,
    )
    labels = model.fit_predict(X_sample)
    show_sdg_window(X_sample, model, labels)

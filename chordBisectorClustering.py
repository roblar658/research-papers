import matplotlib
import numpy as np

# Velg interaktivt GUI-vindu ved kjøring lokalt
for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(backend, force=False)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt


class ChordBisectorClustering:
    """
    Chord-Bisector Clustering:
    Rekursiv romdeling ved å finne midtnormaler mellom tilfeldige punktpar
    som passerer gjennom de tommeste korridorene i datarommet.
    """

    def __init__(
        self,
        n_candidates: int = 350,
        corridor_ratio: float = 0.08,
        max_corridor_density: float = 0.015,
        min_cluster_size: int = 12,
        seed: int = 42,
    ):
        self.n_candidates = n_candidates
        self.corridor_ratio = corridor_ratio
        self.max_corridor_density = max_corridor_density
        self.min_cluster_size = min_cluster_size
        self.seed = seed

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        labels = np.full(n, -1, dtype=np.int32)

        # Global korridorbredde r basert på median parvis avstand
        n_pairs = min(1000, n * 4)
        sample_a = X[rng.integers(0, n, n_pairs)]
        sample_b = X[rng.integers(0, n, n_pairs)]
        dists = np.linalg.norm(sample_a - sample_b, axis=1)
        self.corridor_radius_ = self.corridor_ratio * float(np.median(dists[dists > 0]))

        self.splits_ = []  # Lagrer historikk over kutt for visualisering
        next_cluster_id = 0

        def _recursive_split(indices: np.ndarray, depth: int) -> None:
            nonlocal next_cluster_id
            m_sub = len(indices)

            if m_sub < 2 * self.min_cluster_size:
                labels[indices] = next_cluster_id
                next_cluster_id += 1
                return

            X_sub = X[indices]
            idx_a = rng.integers(0, m_sub, self.n_candidates)
            idx_b = rng.integers(0, m_sub, self.n_candidates)
            valid = idx_a != idx_b
            idx_a, idx_b = idx_a[valid], idx_b[valid]

            best_score = np.inf
            best_record = None
            best_left_mask = None

            for i, j in zip(idx_a, idx_b):
                diff = X_sub[j] - X_sub[i]
                norm = float(np.linalg.norm(diff))
                if norm < 1e-12:
                    continue

                u = diff / norm
                mid = 0.5 * (X_sub[i] + X_sub[j])
                s = (X_sub - mid) @ u

                left_mask = s < 0.0
                n_left = int(np.sum(left_mask))
                n_right = m_sub - n_left

                # Balansekrav: unngå kutt helt i ytterkanten
                if n_left < self.min_cluster_size or n_right < self.min_cluster_size:
                    continue

                corridor_count = int(np.sum(np.abs(s) < self.corridor_radius_))
                if corridor_count < best_score:
                    best_score = corridor_count
                    best_left_mask = left_mask
                    best_record = {
                        "depth": depth,
                        "a": X_sub[i].copy(),
                        "b": X_sub[j].copy(),
                        "mid": mid.copy(),
                        "normal": u.copy(),
                        "corridor_count": corridor_count,
                        "subset_size": m_sub,
                    }

            if (
                best_left_mask is None
                or (best_score / m_sub) > self.max_corridor_density
            ):
                labels[indices] = next_cluster_id
                next_cluster_id += 1
                return

            self.splits_.append(best_record)
            _recursive_split(indices[best_left_mask], depth + 1)
            _recursive_split(indices[~best_left_mask], depth + 1)

        _recursive_split(np.arange(n), depth=0)
        return labels


def visualize_chord_bisector(
    X: np.ndarray, model: ChordBisectorClustering, labels: np.ndarray
) -> None:
    """Viser testdata, korder, midtnormal-korridorer og endelige klynger i eget vindu."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 6), num="Chord-Bisector Clustering Visualisering"
    )
    fig.patch.set_facecolor("#111317")

    pad = 0.8
    xmin, ymin = X.min(axis=0) - pad
    xmax, ymax = X.max(axis=0) + pad
    span = max(xmax - xmin, ymax - ymin) * 1.5
    r = model.corridor_radius_

    # --- Panel 1: Korder, midtnormaler og tomromskorridorer ---
    ax1.set_title(
        f"1. Utvalgte midtnormaler & tomromskorridorer (r = {r:.2f})",
        color="white",
        fontsize=11,
    )
    ax1.scatter(X[:, 0], X[:, 1], s=18, color="#94a3b8", alpha=0.85, zorder=2)

    split_cmap = plt.get_cmap("Set2")
    for idx, sp in enumerate(model.splits_):
        color = split_cmap(idx % 8)
        mid = sp["mid"]
        u = sp["normal"]
        # Tangentvektor t står vinkelrett på normalvektor u
        t = np.array([-u[1], u[0]])

        p1 = mid - span * t
        p2 = mid + span * t

        # Tegn selve korden (linjen mellom paret a og b) som stiplet linje
        ax1.plot(
            [sp["a"][0], sp["b"][0]],
            [sp["a"][1], sp["b"][1]],
            linestyle=":",
            color=color,
            lw=1.1,
            alpha=0.75,
        )

        # Tegn tomromskorridoren +/- r rundt midtnormalen
        poly = np.array([
            p1 - r * u,
            p2 - r * u,
            p2 + r * u,
            p1 + r * u,
        ])
        ax1.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.14, zorder=1)

        # Tegn midtnormalen (skillelinjen)
        ax1.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            lw=1.8,
            label=f"Kutt #{idx + 1} (dybde {sp['depth']}, i korridor={sp['corridor_count']})",
            zorder=3,
        )

    leg1 = ax1.legend(loc="upper left", facecolor="#1b1f27", edgecolor="#3a414e", fontsize=8.5)
    for txt in leg1.get_texts():
        txt.set_color("white")

    # --- Panel 2: Endelig klyngeresultat ---
    ax2.set_title(
        f"2. Ferdig klynging ({len(np.unique(labels))} klynger funnet)",
        color="white",
        fontsize=11,
    )
    cluster_cmap = plt.get_cmap("tab10")
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        ax2.scatter(
            X[mask, 0],
            X[mask, 1],
            s=26,
            color=cluster_cmap(cid % 10),
            label=f"Klynge {cid} (n={np.sum(mask)})",
            alpha=0.92,
            edgecolors="none",
        )

    leg2 = ax2.legend(loc="upper left", facecolor="#1b1f27", edgecolor="#3a414e", fontsize=9)
    for txt in leg2.get_texts():
        txt.set_color("white")

    for ax in (ax1, ax2):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_facecolor("#181b22")
        ax.tick_params(colors="#a0aec0")
        for spine in ax.spines.values():
            spine.set_color("#3a414e")

    fig.suptitle(
        "Chord-Bisector Clustering — Geometri og rekursiv romdeling",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)

    # Generer 5 adskilte klynger med ulik form, rotasjon og tetthet
    c0 = rng.normal([0.0, 0.0], 0.45, (90, 2))
    c1 = rng.normal([4.5, 0.8], 0.50, (85, 2))
    c2 = rng.normal([2.2, 4.2], 0.40, (80, 2))
    # Skråstilt elliptisk klynge
    rot = np.array([[0.8, -0.6], [0.6, 0.8]])
    c3 = (rng.normal([0.0, 0.0], [1.1, 0.25], (95, 2)) @ rot.T) + np.array([6.0, 4.5])
    # Kompakt høy-tetthetsklynge
    c4 = rng.normal([-1.2, 4.4], 0.28, (60, 2))

    X_test = np.vstack([c0, c1, c2, c3, c4])

    model = ChordBisectorClustering(
        n_candidates=400,
        corridor_ratio=0.07,
        max_corridor_density=0.015,
        min_cluster_size=15,
        seed=2026,
    )
    labels = model.fit_predict(X_test)
    visualize_chord_bisector(X_test, model, labels)

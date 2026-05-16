"""
BCC (Body-Centered Cubic) unit cell drawn together with its Voronoi cell
(the truncated octahedron), sharing a common center.

Terminology used here:
  - "Primary" graph: the BCC lattice itself.
      * Primary nodes = the 8 corner atoms + 1 body-center atom.
      * Primary edges = the 12 cube edges + the 8 corner-to-body-center bonds.
  - "Dual" graph: the Voronoi diagram of the BCC lattice.
      * Dual nodes = the 24 vertices of the truncated octahedron.
      * Dual edges = the 36 edges of the truncated octahedron.

The Voronoi cell of a point in a BCC lattice (cube side `a`) is the truncated
octahedron with vertices at all permutations of (0, +/- a/4, +/- a/2). With
the BCC cube also centered at the origin, the 6 square faces of the Voronoi
cell lie exactly on the 6 faces of the cube.
"""

from itertools import permutations, product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial import ConvexHull

plt.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def bcc_cell_centered(a: float = 1.0):
    """Return BCC nodes and all primary edges (cube edges + bonds).

    Cube is centered at the origin: corners at (+/- a/2)^3, body center at 0.
    """
    corners = np.array(list(product([-a / 2, a / 2], repeat=3)))
    center = np.zeros(3)
    all_nodes = np.vstack([corners, center[None, :]])

    primary_edges = []
    # 12 cube edges
    for i, p in enumerate(corners):
        for q in corners[i + 1:]:
            if np.count_nonzero(np.abs(p - q) > 1e-9) == 1:
                primary_edges.append([p, q])
    # 8 corner-to-body-center bonds
    for corner in corners:
        primary_edges.append([corner, center])

    return all_nodes, np.array(primary_edges)


def truncated_octahedron(s: float = 1.0):
    """Truncated octahedron centered at origin.

    24 vertices = all permutations of (0, +/- s, +/- 2s).
    Faces: 8 hexagons + 6 squares.
    For the BCC Voronoi cell with cube side `a`, pass s = a / 4.

    Returns (vertices, faces, edges) where `edges` is an (N, 2, 3) array of
    line segments along the polyhedron's 36 unique edges.
    """
    verts = set()
    for perm in permutations([0, 1, 2]):
        for sign_a in (-1, 1):
            for sign_b in (-1, 1):
                v = [0.0, 0.0, 0.0]
                v[perm[1]] = sign_a * s
                v[perm[2]] = sign_b * 2.0 * s
                verts.add(tuple(v))
    verts = np.array(sorted(verts))

    hull = ConvexHull(verts)
    face_groups: dict[tuple, set[int]] = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq, 5))
        face_groups.setdefault(key, set()).update(int(i) for i in simplex)

    faces = []
    edge_set: set[tuple[int, int]] = set()
    for key, idx_set in face_groups.items():
        pts_idx = np.array(sorted(idx_set))
        pts = verts[pts_idx]
        centroid = pts.mean(axis=0)
        normal = np.array(key[:3])
        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 \
            else np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, helper); u /= np.linalg.norm(u)
        w = np.cross(normal, u);      w /= np.linalg.norm(w)
        rel = pts - centroid
        order = np.argsort(np.arctan2(rel @ w, rel @ u))
        ordered_idx = pts_idx[order]
        faces.append(verts[ordered_idx])
        # collect edges of this face
        for i in range(len(ordered_idx)):
            a_i = int(ordered_idx[i])
            b_i = int(ordered_idx[(i + 1) % len(ordered_idx)])
            edge_set.add((min(a_i, b_i), max(a_i, b_i)))

    edges = np.array([[verts[i], verts[j]] for i, j in edge_set])
    return verts, faces, edges


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PRIMARY_COLOR = "#1f77b4"        # blue: primary nodes
PRIMARY_EDGE_COLOR = "#c0392b"   # red: primary edges
DUAL_COLOR = "#1a1a1a"           # near-black: dual nodes & edges
SQ_FACE = "#ffb84d"
HEX_FACE = "#7ec4ff"
PRIMARY_DASH = (0, (6, 3))


def plot_combined(ax, a: float = 1.0):
    primary_nodes, primary_edges = bcc_cell_centered(a)
    s = a / 4.0
    dual_verts, dual_faces, dual_edges = truncated_octahedron(s)

    # --- Dual cell (truncated octahedron) faces, transparent ---
    face_colors = [SQ_FACE if len(f) == 4 else HEX_FACE for f in dual_faces]
    ax.add_collection3d(Poly3DCollection(
        dual_faces, facecolors=face_colors,
        edgecolors="none", linewidths=0, alpha=1.0,
    ))

    # --- Dual edges (drawn explicitly so they appear in front of faces) ---
    ax.add_collection3d(Line3DCollection(
        dual_edges, colors=DUAL_COLOR, linewidths=1.2,
    ))

    # --- Primary edges: cube edges + nearest-neighbor bonds, red dashed ---
    ax.add_collection3d(Line3DCollection(
        primary_edges, colors=PRIMARY_EDGE_COLOR, linewidths=1.7,
        linestyles=PRIMARY_DASH, alpha=0.95,
    ))

    # --- Primary nodes (corners + body center, all blue) ---
    ax.scatter(*primary_nodes.T, s=240, c=PRIMARY_COLOR,
               edgecolors="#0b3d66", linewidths=1.2,
               depthshade=True, zorder=10)

    # --- Dual nodes ---
    ax.scatter(*dual_verts.T, s=28, c=DUAL_COLOR, zorder=9)

    # --- Legend ---
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="", color=PRIMARY_COLOR,
               markeredgecolor="#0b3d66", markersize=11,
               label="Primary nodes"),
        Line2D([0], [0], color=PRIMARY_EDGE_COLOR, linewidth=1.9,
               linestyle=PRIMARY_DASH, label="Primary edges"),
        Line2D([0], [0], marker="o", linestyle="", color=DUAL_COLOR,
               markersize=7, label="Dual nodes"),
        Line2D([0], [0], color=DUAL_COLOR, linewidth=1.4,
               label="Dual edges"),
        #Line2D([0], [0], marker="s", linestyle="", color=SQ_FACE,
        #       markeredgecolor=DUAL_COLOR, markersize=11,
        #       label="Dual square faces (6)"),
        #Line2D([0], [0], marker="h", linestyle="", color=HEX_FACE,
        #       markeredgecolor=DUAL_COLOR, markersize=13,
        #       label="Dual hexagonal faces (8)"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              fontsize=14, framealpha=0.92, bbox_to_anchor=(-0.05, 1.0))

    # --- Axes styling ---
    lim = a * 0.55
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor((0.75, 0.75, 0.75, 0.5))


def main(a: float = 1.0, save_path: str | None = None):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_combined(ax, a)
    ax.view_init(elev=19, azim=32, roll=-1)
    ax.set_proj_type('persp', focal_length=0.2)
    plt.grid(False)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
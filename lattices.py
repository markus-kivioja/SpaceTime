import numpy as np
from collections import defaultdict
from scipy.spatial import Voronoi, ConvexHull

SQ3 = np.sqrt(3.0)

def _fcc_centerings(base):
    """Add the four F-centering translations to a list of fractional sites."""
    shifts = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    out = []
    for b in base:
        for s in shifts:
            out.append(((b[0] + s[0]) % 1.0,
                        (b[1] + s[1]) % 1.0,
                        (b[2] + s[2]) % 1.0))
    return out


def get_structure(name):
    cubic = np.eye(3)                        # simple-cubic conventional cell

    if name == "Cubic":
        return cubic, np.array([(0, 0, 0)])

    if name == "FCC":
        basis = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
        return cubic, np.array(basis)

    if name == "BCC":
        basis = [(0, 0, 0), (.5, .5, .5)]
        return cubic, np.array(basis)

    if name == "A15":                        # Cr3Si / beta-W, Pm-3n (#223)
        basis = [(0, 0, 0), (.5, .5, .5),                       # 2a  CN12
                 (.25, 0, .5), (.75, 0, .5),                    # 6c  chain || x
                 (.5, .25, 0), (.5, .75, 0),                    #     chain || y
                 (0, .5, .25), (0, .5, .75)]                    #     chain || z
        return cubic, np.array(basis)

    if name == "C15":                        # MgCu2 cubic Laves, Fd-3m (#227)
        mg = [(.125, .125, .125), (.875, .875, .875)]
        cu = [(.5, .5, .5), (.5, .25, .25), (.25, .5, .25), (.25, .25, .5)]
        basis = _fcc_centerings(mg) + _fcc_centerings(cu)
        return cubic, np.array(basis)

    if name == "Z":                          # Frank-Kasper Z phase, Zr4Al3, P6/mmm (#191)
        c_over_a = 0.992                      # a = 5.433 A, c = 5.390 A
        L = np.array([[1.0, 0.0, 0.0],
                      [-0.5, SQ3 / 2, 0.0],
                      [0.0, 0.0, c_over_a]])
        zR = 0.293                            # free parameter of the 2e (CN14) site
        basis = [(1 / 3, 2 / 3, 0.0), (2 / 3, 1 / 3, 0.0),       # 2c  Zr  CN15
                 (0.0, 0.0, zR), (0.0, 0.0, 1.0 - zR),           # 2e  Zr  CN14
                 (.5, 0, .5), (0, .5, .5), (.5, .5, .5)]         # 3g  Al  CN12
        return L, np.array(basis)

    raise ValueError(name)


def build_points(L, basis, reps=range(-2, 4)):
    """Tile the conventional cell over `reps` along each axis -> Cartesian pts."""
    pts = []
    for i in reps:
        for j in reps:
            for k in reps:
                shift = i * L[0] + j * L[1] + k * L[2]
                for b in basis:
                    pts.append(b @ L + shift)
    return np.array(pts)

def _voronoi_neighbors(vor):
    nbrs = defaultdict(set)
    for a, b in vor.ridge_points:
        nbrs[a].add(b)
        nbrs[b].add(a)
    return nbrs


def _order_loop(P):
    """Order (assumed planar) vertices P (m,3) into a single polygon loop."""
    c = P.mean(axis=0)
    Q = P - c
    _, _, vt = np.linalg.svd(Q)
    u, v = vt[0], vt[1]
    ang = np.arctan2(Q @ v, Q @ u)
    return np.argsort(ang)

PRIMAL_K = {"FCC": 4, "BCC": 2, "A15": 2, "C15": 5, "Z": 5}
DUAL_K = {"FCC": 4, "BCC": 3, "A15": 5, "C15": 6, "Z": 6}


def _k_nearest(pts, center, k):
    """Indices of the k atoms nearest `center` (deterministic tie-breaking)."""
    return np.argsort(np.linalg.norm(pts - center, axis=1))[:k]


def coordination_meshes(name):
    """List of (vertices Nx3, tri_faces flat-array) coordination polyhedra."""
    L, basis = get_structure(name)
    pts = build_points(L, basis)
    center = pts.mean(axis=0)
    vor = Voronoi(pts)
    nbrs = _voronoi_neighbors(vor)
    sel = _k_nearest(pts, center, PRIMAL_K[name])

    meshes = []
    for i in sel:
        shell = pts[sorted(nbrs[i])]
        if len(shell) < 4:
            continue
        try:
            hull = ConvexHull(shell, qhull_options="Qt")     # triangulated facets
        except Exception:
            continue
        faces = np.hstack([[3, *tri] for tri in hull.simplices]).astype(np.int64)
        meshes.append((shell, faces))
    return meshes


def voronoi_mesh(name):
    """One combined mesh (vertices Nx3, polygon-faces flat-array) holding the
    unique polygonal facets of the Voronoi cells around the cluster centre."""
    L, basis = get_structure(name)
    pts = build_points(L, basis)
    center = pts.mean(axis=0)
    vor = Voronoi(pts)
    sel = set(_k_nearest(pts, center, DUAL_K[name]).tolist())

    verts, faces, seen = [], [], set()
    for k, (a, b) in enumerate(vor.ridge_points):
        if a not in sel and b not in sel:
            continue
        idx = vor.ridge_vertices[k]
        if -1 in idx:                         # unbounded facet -> skip
            continue
        key = frozenset(idx)
        if key in seen:
            continue
        seen.add(key)
        P = vor.vertices[idx]
        order = _order_loop(P)
        base = len(verts)
        verts.extend(P[order])
        faces.append(len(order))
        faces.extend(range(base, base + len(order)))
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def cube_mesh(center=(0.5, 0.5, 0.5), size=1.0):
    o = np.asarray(center) - size / 2
    v = o + size * np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                            dtype=float)
    quads = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
             [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    faces = np.hstack([[4, *q] for q in quads]).astype(np.int64)
    return v, faces

NAMES = ["Cubic", "FCC", "BCC", "A15", "C15", "Z"]

PRIMAL_COLOR = (0.72, 0.82, 0.72, 0.5)
DUAL_COLOR = (0.62, 0.62, 0.72, 0.5)

ELEV, AZIM = -18.0, -50.0
ZOOM = 2


def _view_vector(elev_deg, azim_deg):
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    cam_from_focal = np.array([np.cos(e) * np.cos(a),
                               np.cos(e) * np.sin(a),
                               np.sin(e)])
    return -cam_from_focal                    # direction camera looks (focal-ward)


def _add(p, verts, faces, color):
    import pyvista as pv
    mesh = pv.PolyData(verts, faces)
    actor = p.add_mesh(mesh, color=color, show_edges=True,
                       edge_color="black", line_width=1.0,
                       smooth_shading=False, split_sharp_edges=False,
                       ambient=0.15, diffuse=1, specular=0.0, opacity=0.8)
    # flat shading: one normal per facet, no Gouraud rounding
    actor.prop.interpolation = "flat"
    return mesh


def _render_panel(meshspecs, color, path, panel, scale):
    import pyvista as pv

    p = pv.Plotter(off_screen=True, border=False,
                   window_size=(panel * scale, panel * scale))
    p.set_background("white")
    p.enable_parallel_projection()            # orthographic: a crystallographic look
    for v, f in meshspecs:
        _add(p, v, f, color)
    p.view_vector(_view_vector(ELEV, AZIM), viewup=(0, 0, 1))
    p.reset_camera()
    p.camera.zoom(ZOOM)
    p.camera_set = True                       # keep show()/render from re-fitting
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    p.show(screenshot=path)                   # off-screen render + write PNG
    p.close()


def _panels_for(name, row):
    if name == "Cubic":
        return [cube_mesh()], (PRIMAL_COLOR if row == 0 else DUAL_COLOR)
    if row == 0:
        return coordination_meshes(name), PRIMAL_COLOR
    return [voronoi_mesh(name)], DUAL_COLOR


def build_figure(out="lattices_recreated.png", panel=420, scale=2):
    import os
    import tempfile
    from PIL import Image

    ncols, nrows = len(NAMES), 2
    pw = ph = panel * scale

    tmp = tempfile.mkdtemp(prefix="lattice_panels_")
    canvas = Image.new("RGB", (pw * ncols, ph * nrows), "white")
    try:
        for col, name in enumerate(NAMES):
            for row in range(nrows):
                meshspecs, color = _panels_for(name, row)
                ppath = os.path.join(tmp, f"{name}_{row}.png")
                _render_panel(meshspecs, color, ppath, panel, scale)
                tile = Image.open(ppath).convert("RGB")
                if tile.size != (pw, ph):      # guard against off-by-one sizing
                    tile = tile.resize((pw, ph), Image.LANCZOS)
                canvas.paste(tile, (col * pw, row * ph))
                print(f"  rendered {name:6s} {'primal' if row == 0 else 'dual'}")
    finally:
        for fn in os.listdir(tmp):
            os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp)

    canvas.save(out)
    print("wrote", out)
    return out


def view_interactive(structures=None, link=None, perspective=False):
    import pyvista as pv

    if structures is None:
        structures = NAMES
    if isinstance(structures, str):
        structures = [structures]
    if link is None:
        link = (len(structures) == 1)

    ncols, nrows = len(structures), 2
    p = pv.Plotter(shape=(nrows, ncols), border=True,
                   title="primal (top) / dual (bottom) — drag to rotate")
    p.set_background("white")

    for col, name in enumerate(structures):
        for row in range(nrows):
            p.subplot(row, col)
            meshspecs, color = _panels_for(name, row)
            for v, f in meshspecs:
                _add(p, v, f, color)
            p.add_text(f"{name}  {'primal' if row == 0 else 'dual'}",
                       font_size=9, color="black")
            if not perspective:
                p.enable_parallel_projection()
            p.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
            p.view_vector(_view_vector(ELEV, AZIM), viewup=(0, 0, 1))
            p.reset_camera()
            p.camera.zoom(ZOOM)                    # a little margin so nothing clips
            p.camera_set = True

    if link:
        p.link_views()
        p.subplot(0, 0)                       # frame to the (larger) primal cell
        if not perspective:
            p.enable_parallel_projection()
        p.view_vector(_view_vector(ELEV, AZIM), viewup=(0, 0, 1))
        p.reset_camera()
        p.camera.zoom(ZOOM)                    # a little margin so nothing clips
        p.camera_set = True                   # survive the render's auto-fit

    p.show()                                  # blocks until the window is closed


def main():
    import sys
    args = sys.argv[1:]
    view_interactive(args[1:] or None)
    #build_figure("lattices_recreated.png")


if __name__ == "__main__":
    main()
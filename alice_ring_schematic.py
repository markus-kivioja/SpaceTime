import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from mayavi.mlab import *
from mlabtex import mlabtex
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.modules.surface import Surface
from mayavi.filters.transform_data import TransformData

z_scale = 6
N = 100

vertical_cut = False
monopole = True
alice_string = True

condensate_R = 9

# Make data
if vertical_cut:
    u = np.linspace(np.pi, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
else:
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(np.pi / 2, np.pi, N)
x = condensate_R * np.outer(np.cos(u), np.sin(v))
y = condensate_R * np.outer(np.sin(u), np.sin(v))
z = z_scale * np.outer(np.ones(np.size(u)), np.cos(v))

mesh(x, y, z, color=(0.6, 0.8, 0.6))

angle = np.linspace(-np.pi * 2 / (N - 2), np.pi * 2, N)
plane_x = condensate_R * np.sin(angle)
if vertical_cut:
    plane_y = np.linspace(0, 0, N)
    plane_z = z_scale * np.cos(angle)
else:
    plane_y = condensate_R * np.cos(angle)
    plane_z = np.linspace(0, 0, N)

plane_x[0] = 0
plane_y[0] = 0

triangles = [(0, i, i + 1) for i in range(1, N - 1)]

s = triangular_mesh(
    plane_x, plane_y, plane_z, triangles, color=(1.0, 1.0, 1.0), opacity=0.6
)

if monopole:
    singularity_size = 0.007
    s2 = triangular_mesh(
        plane_x * singularity_size,
        plane_y * singularity_size + 0.008,
        plane_z * singularity_size * condensate_R / z_scale,
        triangles,
        color=(0.0, 0.0, 0.0),
        opacity=1.0
    )

R = 2.4
r = 0.5

theta = np.linspace(0, 2 * np.pi, N)
phi = np.linspace(0, 2 * np.pi, N)
torus = np.zeros((3, N, N))

if not monopole:
    for i in range(0, N):
        for j in range(0, N):
            torus[0][i][j] = (R + r * np.cos(phi[j])) * np.cos(theta[i])
            torus[1][i][j] = (R + r * np.cos(phi[j])) * np.sin(theta[i])
            torus[2][i][j] = r * np.sin(phi[j])

    mesh(torus[0], torus[1], torus[2], color=(1, 0, 0), opacity=1)

director_N = 19

if vertical_cut:
    field_x = np.linspace(-condensate_R, condensate_R, director_N)
    field_y = np.linspace(0, 0, director_N)
    field_z = np.linspace(-condensate_R, condensate_R, director_N)
    field_x, field_y, field_z = np.meshgrid(field_x, field_y, field_z)
else:
    field_x = np.linspace(-condensate_R, condensate_R, director_N)
    field_y = np.linspace(-condensate_R, condensate_R, director_N)
    field_z = np.linspace(0, 0, director_N)
    field_x, field_y, field_z = np.meshgrid(field_x, field_y, field_z)
    for x_idx in range(director_N):
        for y_idx in range(director_N):
            for z_idx in range(director_N):
                if (
                    np.sqrt(
                        field_x[z_idx][y_idx][x_idx] ** 2
                        + field_y[z_idx][y_idx][x_idx] ** 2
                    )
                    < R * 1.2
                ):
                    field_z[z_idx][y_idx][x_idx] = -0.01

dist = np.sqrt(field_x**2 + field_y**2 + (field_z / 0.72) ** 2)

if alice_string:
    angles = 0.5 * np.arctan2(field_y, -field_x)
    dx = np.cos(angles)
    dy = np.sin(angles)
    dz = field_z * 0
else:
    dx = field_x * 1
    dy = field_y * 1
    dz = -2 * field_z

if not monopole:
    for x_idx in range(director_N):
        for y_idx in range(director_N):
            for z_idx in range(director_N):
                if dist[z_idx][y_idx][x_idx] < condensate_R:
                    xy_dist = np.sqrt(
                        field_x[z_idx][y_idx][x_idx] ** 2
                        + field_y[z_idx][y_idx][x_idx] ** 2
                    )
                    if xy_dist < R * 0.8:
                        dx[z_idx][y_idx][x_idx] = 0
                        dy[z_idx][y_idx][x_idx] = 0
                        if not vertical_cut and xy_dist > R - r:
                            dz[z_idx][y_idx][x_idx] = 0
#if alice_string:
#    for x_idx in range(director_N):
#        for y_idx in range(director_N):
#            for z_idx in range(director_N):
#                if field_x[z_idx][y_idx][x_idx] < -r:
#                    dx[z_idx][y_idx][x_idx] = 0

mag = np.sqrt(dx**2 + dy**2 + dz**2)

for x_idx in range(director_N):
    for y_idx in range(director_N):
        for z_idx in range(director_N):
            if (
                dist[z_idx][y_idx][x_idx] <= condensate_R * 0.95
                and mag[z_idx][y_idx][x_idx] > 0
            ):
                dx[z_idx][y_idx][x_idx] = (
                    dx[z_idx][y_idx][x_idx] / mag[z_idx][y_idx][x_idx]
                )
                dy[z_idx][y_idx][x_idx] = (
                    dy[z_idx][y_idx][x_idx] / mag[z_idx][y_idx][x_idx]
                )
                dz[z_idx][y_idx][x_idx] = (
                    dz[z_idx][y_idx][x_idx] / mag[z_idx][y_idx][x_idx]
                )
            else:
                dx[z_idx][y_idx][x_idx] = 0
                dy[z_idx][y_idx][x_idx] = 0
                dz[z_idx][y_idx][x_idx] = 0

d_field = quiver3d(
    field_x,
    field_y,
    field_z,
    dx,
    dy,
    dz,
    mode="arrow",
    color=(0, 0, 0),
    scale_factor=0.9,
    opacity=1,
)
d_field.glyph.glyph_source.glyph_source.tip_length = 0.45
d_field.glyph.glyph_source.glyph_source.tip_radius = 0.15
if not vertical_cut:
    d_field.glyph.glyph_source.glyph_position = "center"
# else:
#    d_field.glyph.glyph_source.glyph_position = "tail"

my_azimuth = 58
my_elevation = 58  # 54.735610317245346

if alice_string or (vertical_cut and not monopole):
    line_N = 100

    for i in range(10):
        R2 = R + 0.01 * i

        if alice_string:
            line_t = np.linspace(-np.pi, np.pi, line_N)
            line_x = -R2 * np.cos(line_t)
            line_y = R2 * np.sin(line_t)
            line_z = np.linspace(0.15, 0.15, line_N)
        else:
            line_t = np.linspace(0, 2 * np.pi, line_N)
            line_x = -R2 * np.cos(line_t) + R
            line_y = np.linspace(0.15, 0.15, line_N)
            line_z = R2 * np.sin(line_t)

        ring = plot3d(line_x, line_y, line_z, line_t, colormap="bwr")
        ring.actor.property.lighting = False
        # if i == 0:
        #    colorbar(ring)

    line_N = 12

    if alice_string:
        line_t = np.linspace(-0.94 * np.pi, 0.94 * np.pi, line_N)
    else:
        line_t = np.linspace(0.3, 1.95 * np.pi, line_N)
    
    if alice_string:
        line_x = -R * np.cos(line_t)
        line_y = R * np.sin(line_t)
        line_z = np.linspace(0.395, 0.395, line_N)

        line_dx = np.cos(0.5 * line_t)
        line_dy = np.sin(0.5 * line_t)
        line_dz = line_z * 0
    else:
        line_x = -R * np.cos(line_t) + R
        line_y = np.linspace(0.395, 0.395, line_N)
        line_z = R * np.sin(line_t)

        line_dx = line_x * 1
        line_dy = line_y * 0
        line_dz = -2 * line_z

    if not alice_string:
        for idx in range(line_N):
            if np.abs(line_dx[idx]) < R * 0.8:
                line_dx[idx] = 0

    line_mag = np.sqrt(line_dx**2 + line_dy**2 + line_dz**2)

    for idx in range(line_N):
        if line_mag[idx] > 0:
            line_dx[idx] = line_dx[idx] / line_mag[idx] * (idx + 1)
            line_dy[idx] = line_dy[idx] / line_mag[idx] * (idx + 1)
            line_dz[idx] = line_dz[idx] / line_mag[idx] * (idx + 1)
        # else:
        #    line_dx[idx] = 0
        #    line_dy[idx] = 0
        #    line_dz[idx] = -0.00001

    ring_quiver = quiver3d(
        line_x,
        line_y,
        line_z,
        line_dx,
        line_dy,
        line_dz,
        # scalars=[i for i in range(len(line_t))],
        scale_mode="scalar",
        colormap="bwr",
        mode="arrow",
        # color=(1, 1, 1),
        scale_factor=1.2,
        # mask_points=6,
        # vmin=10,
        # vmax=13,
        opacity=1,
    )

    ring_arror_resolution = 60
    ring_arror_tip_length = 0.5
    ring_arror_tip_radius = 0.2

    # ring_quiver.actor.property.lighting = False
    ring_quiver.glyph.glyph_source.glyph_position = "center"
    ring_quiver.glyph.glyph_source.glyph_source.shaft_resolution = ring_arror_resolution
    ring_quiver.glyph.glyph_source.glyph_source.tip_resolution = ring_arror_resolution
    ring_quiver.glyph.glyph_source.glyph_source.tip_length = ring_arror_tip_length
    ring_quiver.glyph.glyph_source.glyph_source.tip_radius = ring_arror_tip_radius

    # ring_quiver2 = quiver3d(
    #    line_x,
    #    line_y,
    #    line_z,
    #    line_dx,
    #    line_dy,
    #    line_dz,
    #    # scalars=[i for i in range(len(line_t))],
    #    scale_mode="scalar",
    #    # colormap="bwr",
    #    mode="arrow",
    #    color=(0, 0, 0),
    #    scale_factor=1.2,
    #    # mask_points=6,
    #    # vmin=10,
    #    # vmax=13,
    #    # line_width=2,
    #    opacity=1,
    # )
    #
    # ring_quiver2.actor.property.lighting = False
    # ring_quiver2.actor.property.representation = "wireframe"
    # ring_quiver2.glyph.glyph_source.glyph_position = "center"
    # ring_quiver2.glyph.glyph_source.glyph_source.shaft_resolution = (
    #    ring_arror_resolution
    # )
    # ring_quiver2.glyph.glyph_source.glyph_source.tip_resolution = ring_arror_resolution
    # ring_quiver2.glyph.glyph_source.glyph_source.tip_length = ring_arror_tip_length
    # ring_quiver2.glyph.glyph_source.glyph_source.tip_radius = ring_arror_tip_radius

    TEXT = r"$\mathcal{L}^\prime$" if alice_string else r"$\mathcal{L}$"

    if alice_string:
        text_x = 3.0 * r
        text_y = 1.68 * R
        text_z = 0.49
    else:
        text_x = 1.08 * R
        text_y = 0.49
        text_z = -1.29 * R
    text_scale = 0.8
    tex = mlabtex(
        x=text_x,
        y=text_y,
        z=text_z,
        scale=text_scale,
        text=TEXT,
        color=(1, 1, 1),
        orientation=(my_elevation, 0.0, 90 + my_azimuth),
        dpi=1200,
        opacity=1,
    )
    tex.actor.property.lighting = False
    tex.actor.property.opacity = 1
    if alice_string:
        text_z = 0.5
    else:
        text_y = 0.5
    tex2 = mlabtex(
        x=text_x,
        y=text_y,
        z=text_z,
        scale=text_scale,
        text=TEXT,
        color=(1, 1, 1),
        orientation=(my_elevation, 0.0, 90 + my_azimuth),
        dpi=1200,
        opacity=1,
    )
    tex2.actor.property.lighting = False
    tex2.actor.property.opacity = 1

if alice_string:
    engine = get_engine()

    # Add a cylinder builtin source
    cylinder_src = BuiltinSurface()
    engine.add_source(cylinder_src)
    cylinder_src.source = 'cylinder'
    cylinder_src.data_source.center = np.array([ 0.,  0.,  0.])
    cylinder_src.data_source.radius = r
    cylinder_src.data_source.capping = False
    cylinder_src.data_source.resolution = 100
    cylinder_src.data_source.height = z_scale * 2
    cyl_surface = Surface()

    def rotMat3D(axis, angle, tol=1e-12):
        """Return the rotation matrix for 3D rotation by angle `angle` degrees about an
        arbitrary axis `axis`.
        """
        t = np.radians(angle)
        x, y, z = axis
        R = (np.cos(t))*np.eye(3) +\
        (1-np.cos(t))*np.matrix(((x**2,x*y,x*z),(x*y,y**2,y*z),(z*x,z*y,z**2))) + \
        np.sin(t)*np.matrix(((0,-z,y),(z,0,-x),(-y,x,0)))
        R[np.abs(R)<tol]=0.0
        return R

    # Add transformation filter to rotate cylinder about an axis
    transform_data_filter = TransformData()
    engine.add_filter(transform_data_filter, cylinder_src)
    Rt = np.eye(4)
    Rt[0:3,0:3] = rotMat3D((1,0,0), 90) # in homogeneous coordinates
    Rtl = list(Rt.flatten()) # transform the rotation matrix into a list

    transform_data_filter.transform.matrix.__setstate__({'elements': Rtl})
    transform_data_filter.widget.set_transform(transform_data_filter.transform)
    transform_data_filter.filter.update()
    transform_data_filter.widget.enabled = False   # disable the rotation control further.
    engine.add_filter(cyl_surface, transform_data_filter)

    cyl_surface.actor.property.color = (1.0, 0.0, 0.0)

    s2 = triangular_mesh(
        plane_x / condensate_R * r,
        plane_y / condensate_R * r,
        plane_z / condensate_R * r + z_scale,
        triangles,
        color=(1.0, 0.0, 0.0),
        opacity=1.0
    )

figure(gcf(), bgcolor=(1, 1, 1))

my_view = view(azimuth=my_azimuth, elevation=my_elevation, focalpoint=[2, 0, 0.7])

ori_axes = orientation_axes()
ori_axes.marker.orientation_marker.cone_resolution = 100
ori_axes.marker.orientation_marker.cylinder_resolution = 100
ori_axes.marker.orientation_marker.axis_labels = 0

show()

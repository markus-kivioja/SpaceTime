import numpy as np
from mayavi.mlab import *
from tvtk.api import tvtk
from scipy.special import sph_harm

filename = 'spin1_0.500201.vtk'
#pre_image_filename = 'pre_image_0.500201.vtk'
pre_image_filename = 'hyper_pre_image_0.500201.vtk'

figure(1, bgcolor=(1, 1, 1))
clf()

# Create a sphere
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

sphere_x = sin(phi) * cos(theta)
sphere_y = sin(phi) * sin(theta)
sphere_z = cos(phi)

Y11  = sph_harm( 1, 1, theta, phi)
Y10  = sph_harm( 0, 1, theta, phi)
Y1_1 = sph_harm(-1, 1, theta, phi)

normal_dirs_x = []
normal_dirs_y = []
normal_dirs_z = []

binormal_dirs_x = []
binormal_dirs_y = []
binormal_dirs_z = []

def draw_sph_harm(psi, location, spin):
    global sphere_z
    global sphere_y
    global sphere_z

    Y = psi[0] * Y11
    Y += psi[1] * Y10
    Y += psi[2] * Y1_1

    mag_sqr = (np.conj(Y) * Y).real
    phase = np.angle(Y)

    max_mag = 0
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if mag_sqr[i][j] > max_mag:
                max_mag = mag_sqr[i][j]

    min_phase = 9999.9
    for i in range(len(phase)):
        for j in range(len(phase[i])):
            if phase[i][j] < min_phase:
                min_phase = phase[i][j]

    if min_phase < 0:
        for i in range(len(phase)):
            for j in range(len(phase[i])):
                phase[i][j] += np.pi

    target_phase = np.pi / 3
    min_diff_to_target_phase = 9999.9
    target_phase_i = 0
    target_phase_j = 0
    for i in range(len(phase)):
        for j in range(len(phase[i])):
            if np.abs(phase[i][j] - target_phase) < min_diff_to_target_phase:
                min_diff_to_target_phase = np.abs(phase[i][j] - target_phase)
                target_phase_i = i
                target_phase_j = j

    temp_vec = [sphere_x[target_phase_i, target_phase_j], sphere_y[target_phase_i, target_phase_j], sphere_z[target_phase_i, target_phase_j]]
    normal = np.cross(spin, temp_vec)
    min_phase_dir = np.cross(spin, normal)
    min_phase_dir /= np.linalg.norm(min_phase_dir)

    normal_dirs_x.append(-min_phase_dir[0])
    normal_dirs_y.append(-min_phase_dir[1])
    normal_dirs_z.append(-min_phase_dir[2])

    binormal = np.cross(spin, min_phase_dir)

    binormal_dirs_x.append(binormal[0])
    binormal_dirs_y.append(binormal[1])
    binormal_dirs_z.append(binormal[2])

    #points = np.array([sphere_x, sphere_y, sphere_z]) * mag_sqr
    #mesh(
    #    points[0] + location[0],
    #    points[1] + location[1],
    #    points[2] + location[2],
    #    scalars = phase,
    #    colormap="bwr"
    #)

reader = tvtk.PolyDataReader( file_name=filename )
reader.read_all_scalars = True
poly_data = reader.get_output()
reader.update()

positions = poly_data.points
m1_r =  poly_data.point_data.get_array(0)
m1_i =  poly_data.point_data.get_array(1)
m0_r =  poly_data.point_data.get_array(2)
m0_i =  poly_data.point_data.get_array(3)
m_1_r = poly_data.point_data.get_array(4)
m_1_i = poly_data.point_data.get_array(5)

pre_image_reader = tvtk.PolyDataReader( file_name=pre_image_filename )
pre_image_reader.read_all_scalars = True
pre_image_poly_data = pre_image_reader.get_output()
pre_image_reader.update()

pre_image_positions = pre_image_poly_data.points
pre_image_m1_r =  pre_image_poly_data.point_data.get_array(0)
pre_image_m1_i =  pre_image_poly_data.point_data.get_array(1)
pre_image_m0_r =  pre_image_poly_data.point_data.get_array(2)
pre_image_m0_i =  pre_image_poly_data.point_data.get_array(3)
pre_image_m_1_r = pre_image_poly_data.point_data.get_array(4)
pre_image_m_1_i = pre_image_poly_data.point_data.get_array(5)

BLOCK_COUNT_X = 114
BLOCK_COUNT_Y = 114
BLOCK_COUNT_Z = 114

BLOCK_SIZE = 12
y_stride = BLOCK_SIZE
x_stride = y_stride * BLOCK_COUNT_Y
z_stride = x_stride * BLOCK_COUNT_X

GAP_SCALE = 0.2

axis_x = []
axis_y = []
axis_z = []

spin_dirs_x = []
spin_dirs_y = []
spin_dirs_z = []

def draw_on_axis():
    for x_idx in range(int(BLOCK_COUNT_X / 2), BLOCK_COUNT_X):
        if x_idx % 4 == 0:
            block_offset = 0
            local_offset = 1
            y_idx = int(BLOCK_COUNT_Y / 2)
            idx = (int(BLOCK_COUNT_Z / 2) + block_offset) * z_stride + y_idx * y_stride + x_idx * x_stride + local_offset

            psi = [
                m1_r[idx] + m1_i[idx]*1j,
                m0_r[idx] + m0_i[idx]*1j,
                m_1_r[idx] + m_1_i[idx]*1j
            ]
            location = [
                (x_idx - BLOCK_COUNT_X / 2) * GAP_SCALE,
                (y_idx - BLOCK_COUNT_Y / 2) * GAP_SCALE,
                0
            ]

            axis_x.append(location[0])
            axis_y.append(location[1])
            axis_z.append(location[2])
            
            normSq_s1 = psi[0].real * psi[0].real + psi[0].imag * psi[0].imag
            normSq_s_1 = psi[2].real * psi[2].real + psi[2].imag * psi[2].imag

            temp = np.sqrt(2) * (psi[0].conjugate() * psi[1] + psi[1].conjugate() * psi[2])

            spin = [temp.real, temp.imag, normSq_s1 - normSq_s_1]
            spin = spin / np.linalg.norm(spin)

            spin_dirs_x.append(spin[0])
            spin_dirs_y.append(spin[1])
            spin_dirs_z.append(spin[2])

            draw_sph_harm(psi, location, spin)

    for y_idx in range(int(BLOCK_COUNT_Y / 2), BLOCK_COUNT_Y):
        if y_idx % 4 == 0:
            block_offset = 0
            local_offset = 5
            x_idx = int(BLOCK_COUNT_X / 2)
            idx = (int(BLOCK_COUNT_Z / 2) + block_offset) * z_stride + y_idx * y_stride + x_idx * x_stride + local_offset

            psi = [
                m1_r[idx] + m1_i[idx]*1j,
                m0_r[idx] + m0_i[idx]*1j,
                m_1_r[idx] + m_1_i[idx]*1j
            ]
            location = [
                (x_idx - BLOCK_COUNT_X / 2) * GAP_SCALE,
                (y_idx - BLOCK_COUNT_Y / 2) * GAP_SCALE,
                0
            ]

            axis_x.append(location[0])
            axis_y.append(location[1])
            axis_z.append(location[2])
            
            normSq_s1 = psi[0].real * psi[0].real + psi[0].imag * psi[0].imag
            normSq_s_1 = psi[2].real * psi[2].real + psi[2].imag * psi[2].imag

            temp = np.sqrt(2) * (psi[0].conjugate() * psi[1] + psi[1].conjugate() * psi[2])

            spin = [temp.real, temp.imag, normSq_s1 - normSq_s_1]
            spin = spin / np.linalg.norm(spin)

            spin_dirs_x.append(spin[0])
            spin_dirs_y.append(spin[1])
            spin_dirs_z.append(spin[2])

            draw_sph_harm(psi, location, spin)

    for z_idx in range(int(BLOCK_COUNT_Z / 2), BLOCK_COUNT_Z):
        if z_idx % 4 == 0:
            block_offset = 0
            local_offset = 0
            idx = z_idx * z_stride + (int(BLOCK_COUNT_Y / 2) + block_offset) * y_stride + (int(BLOCK_COUNT_Z / 2) + block_offset) * x_stride + local_offset

            psi = [
                m1_r[idx] + m1_i[idx]*1j,
                m0_r[idx] + m0_i[idx]*1j,
                m_1_r[idx] + m_1_i[idx]*1j
            ]
            location = [
                0,
                0,
                (z_idx - BLOCK_COUNT_Z / 2) * GAP_SCALE
            ]

            axis_x.append(location[0])
            axis_y.append(location[1])
            axis_z.append(location[2])

            normSq_s1 = psi[0].real * psi[0].real + psi[0].imag * psi[0].imag
            normSq_s_1 = psi[2].real * psi[2].real + psi[2].imag * psi[2].imag

            temp = np.sqrt(2) * (psi[0].conjugate() * psi[1] + psi[1].conjugate() * psi[2])

            spin = [temp.real, temp.imag, normSq_s1 - normSq_s_1]
            spin = spin / np.linalg.norm(spin)

            spin_dirs_x.append(spin[0])
            spin_dirs_y.append(spin[1])
            spin_dirs_z.append(spin[2])
            
            draw_sph_harm(psi, location, spin)

def draw_preimage():
    point_count = len(pre_image_m1_r)
    print(point_count)
    for idx in range(point_count):
        psi = [
            pre_image_m1_r[idx] + pre_image_m1_i[idx]*1j,
            pre_image_m0_r[idx] + pre_image_m0_i[idx]*1j,
            pre_image_m_1_r[idx] + pre_image_m_1_i[idx]*1j
        ]
        location = np.array(pre_image_positions[idx]) / 12.0 * (BLOCK_COUNT_X / 2) * GAP_SCALE

        axis_x.append(location[0])
        axis_y.append(location[1])
        axis_z.append(location[2])
        
        normSq_s1 = psi[0].real * psi[0].real + psi[0].imag * psi[0].imag
        normSq_s_1 = psi[2].real * psi[2].real + psi[2].imag * psi[2].imag

        temp = np.sqrt(2) * (psi[0].conjugate() * psi[1] + psi[1].conjugate() * psi[2])

        spin = [temp.real, temp.imag, normSq_s1 - normSq_s_1]
        spin = spin / np.linalg.norm(spin)

        spin_dirs_x.append(spin[0])
        spin_dirs_y.append(spin[1])
        spin_dirs_z.append(spin[2])

        draw_sph_harm(psi, location, spin)

#draw_on_axis()
draw_preimage()

my_mask_points = 10
cylinder_radius = 0.06

spin_field = quiver3d(
        axis_x,
        axis_y,
        axis_z,
        spin_dirs_x,
        spin_dirs_y,
        spin_dirs_z,
        mode="arrow",
        scale_factor=0.4,
        color=(0, 1, 0),
        #mask_points = my_mask_points
        )

normal_field = quiver3d(
        axis_x,
        axis_y,
        axis_z,
        normal_dirs_x,
        normal_dirs_y,
        normal_dirs_z,
        mode="cylinder",
        scale_factor=0.2,
        color=(1, 0, 0),
        #mask_points = my_mask_points
        )

normal_field.glyph.glyph_source.glyph_source.resolution = 60
normal_field.glyph.glyph_source.glyph_source.radius = cylinder_radius

binormal_field = quiver3d(
        axis_x,
        axis_y,
        axis_z,
        binormal_dirs_x,
        binormal_dirs_y,
        binormal_dirs_z,
        mode="cylinder",
        scale_factor=0.2,
        color=(0, 0, 1),
        #mask_points = my_mask_points
        )

binormal_field.glyph.glyph_source.glyph_source.resolution = 60
binormal_field.glyph.glyph_source.glyph_source.radius = cylinder_radius

ori_axes = orientation_axes()
ori_axes.marker.orientation_marker.cone_resolution = 100
ori_axes.marker.orientation_marker.cylinder_resolution = 100
ori_axes.marker.orientation_marker.axis_labels = 0

show()
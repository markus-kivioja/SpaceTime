import numpy as np
from mayavi.mlab import *
from tvtk.api import tvtk
from scipy.special import sph_harm

#filename = '0.502515.vtk'
filename = '0.241425.vtk'

use_spherical_harmonics = True
#use_spherical_harmonics = False

figure(1, bgcolor=(1, 1, 1))
clf()

def dist(p1, p2):
    diff = p2 - p1
    return np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

prev_x = np.array([])
prev_y = np.array([])
prev_z = np.array([])

def get_nearest_perm(x, y, z):
    global prev_x
    global prev_y
    global prev_z

    perms = [
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1],
        [1, 2, 3, 0],

        [0, 1, 3, 2],
        [2, 0, 1, 3],
        [3, 2, 0, 1],
        [1, 3, 2, 0],

        [0, 2, 3, 1],
        [1, 0, 2, 3],
        [3, 1, 0, 2],
        [2, 3, 1, 0],
        
        [0, 2, 1, 3],
        [3, 0, 2, 1],
        [1, 3, 0, 2],
        [2, 1, 3, 0],

        [0, 3, 1, 2],
        [2, 0, 3, 1],
        [1, 2, 0, 3],
        [3, 1, 2, 0],

        [0, 3, 2, 1],
        [1, 0, 3, 2],
        [2, 1, 0, 3],
        [3, 2, 1, 0],
    ]

    min_dist = 9999999999999
    shortest_perm = 0
    for i in range(len(perms)):
        cur_dist = 0
        for j in range(4):
            p0 = np.array([prev_x[j], prev_y[j], prev_z[j]])
            p1 = np.array([x[perms[i][j]], y[perms[i][j]], z[perms[i][j]]])
            cur_dist += dist(p0, p1)

        if cur_dist < min_dist:
            min_dist = cur_dist
            shortest_perm = perms[i]
    return shortest_perm

def get_normal(psi):
    roots = np.roots([
        psi[0],
        2*psi[1],
        np.sqrt(6)*psi[2],
        2*psi[3],
        psi[4]
    ])
    for i in range(4 - len(roots)):
        roots = np.append([np.inf + np.inf * 1j], roots)

    phi = 2*np.arctan(abs(roots)) 
    theta = np.arctan2(roots.imag, roots.real)
    theta = np.nan_to_num(theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    corners = np.array(
        [
            np.array([x[0], y[0], z[0]]),
            np.array([x[1], y[1], z[1]]),
            np.array([x[2], y[2], z[2]]),
        ])
    normal = np.cross(corners[2] - corners[0], corners[1] - corners[0])
    return normal / np.linalg.norm(normal)

total_x = np.array([])
total_y = np.array([])
total_z = np.array([])
total_triangle = np.array([[0, 0, 0]])
scalars = np.array([])
counter = 0
needs_resetting = False

def draw_majorana(psi, location, reset=False):
    global prev_x
    global prev_y
    global prev_z
    global total_x
    global total_y
    global total_z
    global total_triangle
    global scalars
    global counter
    global needs_resetting

    if reset:
        needs_resetting = True

    roots = np.roots([
            psi[0],
            2*psi[1],
            np.sqrt(6)*psi[2],
            2*psi[3],
            psi[4]
        ])
    
    roots = np.array(sorted(roots, key=np.angle))

    resetted = False
    if needs_resetting and len(roots) == 4:
        roots = np.array(sorted(roots, key=np.angle))
        resetted = True

    for i in range(4 - len(roots)):
        roots = np.append([np.inf + np.inf * 1j], roots)

    phi = 2*np.arctan(abs(roots)) 
    theta = np.arctan2(roots.imag, roots.real)
    theta = np.nan_to_num(theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    new_x = x.copy()
    new_y = y.copy()
    new_z = z.copy()

    if not needs_resetting:
        perm = get_nearest_perm(x, y, z)
        for i in range(4):
            new_x[i] = x[perm[i]]
            new_y[i] = y[perm[i]]
            new_z[i] = z[perm[i]]

    if resetted:
        needs_resetting = False

    prev_x = new_x.copy()
    prev_y = new_y.copy()
    prev_z = new_z.copy()

    total_x = np.append(total_x, new_x + location[0])
    total_y = np.append(total_y, new_y + location[1])
    total_z = np.append(total_z, new_z + location[2])
    start_vert = counter * 4
    new_triangle = np.array([
        (start_vert + 0, start_vert + 1, start_vert + 2),
        #(start_vert + 0, start_vert + 1, start_vert + 3),
        (start_vert + 0, start_vert + 2, start_vert + 3),
        #(start_vert + 1, start_vert + 2, start_vert + 3),
        ])
    total_triangle = np.append(total_triangle, new_triangle, axis=0)
    scalars = np.append(scalars, np.array([0, 1, 2, 3]))
    counter += 1

# Create a sphere
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

sphere_x = sin(phi) * cos(theta)
sphere_y = sin(phi) * sin(theta)
sphere_z = cos(phi)

Y22  = sph_harm( 2, 2, theta, phi)
Y21  = sph_harm( 1, 2, theta, phi)
Y20  = sph_harm( 0, 2, theta, phi)
Y2_1 = sph_harm(-1, 2, theta, phi)
Y2_2 = sph_harm(-2, 2, theta, phi)

def draw_sph_harm(psi, location):
    global sphere_z
    global sphere_y
    global sphere_z

    Y =  psi[0] * Y22
    Y += psi[1] * Y21
    Y += psi[2] * Y20
    Y += psi[3] * Y2_1
    Y += psi[4] * Y2_2

    mag_sqr = (np.conj(Y) * Y).real
    phase = np.angle(Y)

    #for i in range(len(phase)):
    #    for j in range(len(phase[i])):
    #        if abs(abs(phase[i][j]) - np.pi) < 0.000001:
    #            phase[i][j] = np.pi

    points = np.array([sphere_x, sphere_y, sphere_z]) * mag_sqr
    mesh(
        points[0] + location[0],
        points[1] + location[1],
        points[2] + location[2],
        scalars = phase,
        colormap="bwr"
    )

reader = tvtk.PolyDataReader( file_name=filename )
reader.read_all_scalars = True
poly_data = reader.get_output()
reader.update()

positions = poly_data.points
m2_r =  poly_data.point_data.get_array(0)
m2_i =  poly_data.point_data.get_array(1)
m1_r =  poly_data.point_data.get_array(2)
m1_i =  poly_data.point_data.get_array(3)
m0_r =  poly_data.point_data.get_array(4)
m0_i =  poly_data.point_data.get_array(5)
m_1_r = poly_data.point_data.get_array(6)
m_1_i = poly_data.point_data.get_array(7)
m_2_r = poly_data.point_data.get_array(8)
m_2_i = poly_data.point_data.get_array(9)

BLOCK_COUNT_X = 114
BLOCK_COUNT_Y = 114
BLOCK_COUNT_Z = 114

BLOCK_SIZE = 12
y_stride = BLOCK_SIZE
x_stride = y_stride * BLOCK_COUNT_Y
z_stride = x_stride * BLOCK_COUNT_X

if use_spherical_harmonics:
    GAP_SCALE = 0.25
else:
    GAP_SCALE = 1.1

def draw_on_axis():
    first_iter = True
    for x_idx in range(BLOCK_COUNT_X - 1, -1, -1):
        if x_idx % 2 == 0:
            block_offset = 0
            local_offset = 1
            idx = (int(BLOCK_COUNT_Z / 2) + block_offset) * z_stride + x_idx * y_stride + x_idx * x_stride + local_offset

            psi = [
                m2_r[idx] + m2_i[idx]*1j,
                m1_r[idx] + m1_i[idx]*1j,
                m0_r[idx] + m0_i[idx]*1j,
                m_1_r[idx] + m_1_i[idx]*1j,
                m_2_r[idx] + m_2_i[idx]*1j
            ]
            location = [
                (x_idx - BLOCK_COUNT_X / 2) * GAP_SCALE,
                (x_idx - BLOCK_COUNT_Y / 2) * GAP_SCALE,
                0
            ]

            if use_spherical_harmonics:
                draw_sph_harm(psi, location)
            else:
                draw_majorana(psi, location, first_iter)
            first_iter = False

    for y_idx in range(BLOCK_COUNT_Y):
        if y_idx % 2 == 0:
            block_offset = 0
            local_offset = 5
            x_idx = (BLOCK_COUNT_Y - y_idx)
            idx = (int(BLOCK_COUNT_Z / 2) + block_offset) * z_stride + y_idx * y_stride + x_idx * x_stride + local_offset

            psi = [
                m2_r[idx] + m2_i[idx]*1j,
                m1_r[idx] + m1_i[idx]*1j,
                m0_r[idx] + m0_i[idx]*1j,
                m_1_r[idx] + m_1_i[idx]*1j,
                m_2_r[idx] + m_2_i[idx]*1j
            ]
            location = [
                (x_idx - BLOCK_COUNT_X / 2) * GAP_SCALE,
                (y_idx - BLOCK_COUNT_Y / 2) * GAP_SCALE,
                0
            ]

            if use_spherical_harmonics:
                draw_sph_harm(psi, location)
            else:
                draw_majorana(psi, location, y_idx == 0)

    for z_idx in range(BLOCK_COUNT_Z):
        if z_idx % 2 == 0:
            block_offset = 0
            local_offset = 0
            idx = z_idx * z_stride + (int(BLOCK_COUNT_Y / 2) + block_offset) * y_stride + (int(BLOCK_COUNT_Z / 2) + block_offset) * x_stride + local_offset

            psi = [
                m2_r[idx] + m2_i[idx]*1j,
                m1_r[idx] + m1_i[idx]*1j,
                m0_r[idx] + m0_i[idx]*1j,
                m_1_r[idx] + m_1_i[idx]*1j,
                m_2_r[idx] + m_2_i[idx]*1j
            ]
            location = [
                0,
                0,
                (z_idx - BLOCK_COUNT_Z / 2) * GAP_SCALE
            ]
            
            if use_spherical_harmonics:
                draw_sph_harm(psi, location)
            else:
                draw_majorana(psi, location, z_idx == 0)

def draw_preimage(vec):
    for x_idx in range(BLOCK_COUNT_X):
        #if x_idx % 2 == 0:
        for y_idx in range(BLOCK_COUNT_Y):
                #if y_idx % 2 == 0:
            for z_idx in range(BLOCK_COUNT_Z):
                        #if z_idx % 2 == 0:
                idx = z_idx * z_stride + y_idx * y_stride + x_idx * x_stride

                psi = [
                    m2_r[idx] + m2_i[idx]*1j,
                    m1_r[idx] + m1_i[idx]*1j,
                    m0_r[idx] + m0_i[idx]*1j,
                    m_1_r[idx] + m_1_i[idx]*1j,
                    m_2_r[idx] + m_2_i[idx]*1j
                ]
                location = [
                    (x_idx - BLOCK_COUNT_X / 2) * GAP_SCALE,
                    (y_idx - BLOCK_COUNT_Y / 2) * GAP_SCALE,
                    (z_idx - BLOCK_COUNT_Z / 2) * GAP_SCALE
                ]

                normal = get_normal(psi)
                epsilon = 0.1
                if np.linalg.norm(normal - vec) < epsilon or np.linalg.norm(-normal - vec) < epsilon:
                    draw_majorana(psi, location, False)

draw_on_axis()
#draw_preimage(np.array([0, 0, 1]))

if not use_spherical_harmonics:
    mesh = triangular_mesh(
        total_x,
        total_y,
        total_z, 
        total_triangle,
        scalars=scalars
    )
    #mesh.actor.property.representation = "wireframe"
    mesh.actor.property.lighting = False

show()
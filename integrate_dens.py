import numpy as np
import cupy as cp
from tvtk.api import tvtk
import math
import matplotlib.pyplot as plt
from matplotlib import widgets
from matplotlib.colors import ListedColormap

filename = 'build/Spin2GpeBCC/bn_vert/proj_z/spinor_vtks/0.250141.vtk'
#filename = '../cyclic_0.241425.vtk'
#filename = '../bn_0.502515.vtk'

N = 114

integ_dens_top = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    
    #define Z_QUANTIZED 0
    #define Y_QUANTIZED 1
    #define X_QUANTIZED 2

    #define BASIS Z_QUANTIZED
    
    #define VALUES_IN_BLOCK 12

    struct BlockPsi {
        complex<double> values[VALUES_IN_BLOCK];
    };
    
    extern "C" __global__
    void integ_dens_top(double* dens, const BlockPsi* m2, const BlockPsi* m1, const BlockPsi* m0, const BlockPsi* m_1, const BlockPsi* m_2, const int N, const double angle) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        int center_x = N / 2;
        int center_y = N / 2;

        int read_x = center_x + (double)(center_x - x) * cos(angle) - (double)(center_y - y) * sin(angle);
        int read_y = center_y + (double)(center_x - x) * sin(angle) + (double)(center_y - y) * cos(angle);

        if (x >= N || y >= N) return;

        double dens_s2  = 0;
        double dens_s1  = 0;
        double dens_s0  = 0;
        double dens_s_1 = 0;
        double dens_s_2 = 0;
        for (int z = 0; z < N; ++z)
        {
            const int idx = z * N * N + read_y * N + read_x;
            for (int dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
            {
                complex<double> s2  = m2[idx].values[dualNode];
                complex<double> s1  = m1[idx].values[dualNode];
                complex<double> s0  = m0[idx].values[dualNode];
                complex<double> s_1 = m_1[idx].values[dualNode];
                complex<double> s_2 = m_2[idx].values[dualNode];

#if BASIS == X_QUANTIZED
                double c = sqrt(6) * 0.25;
                complex<double> x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
                complex<double> x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
                complex<double> x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
                complex<double> x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
                complex<double> x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

                s2 =  x_s2;
                s1 =  x_s1;
                s0 =  x_s0;
                s_1 = x_s_1;
                s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
                double c = sqrt(6) * 0.25;
                complex<double> im = complex<double>( 0, 1 );
                complex<double> y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
                complex<double> y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
                complex<double> y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
                complex<double> y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
                complex<double> y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

                s2 =  y_s2;
                s1 =  y_s1;
                s0 =  y_s0;
                s_1 = y_s_1;
                s_2 = y_s_2;
#endif

                dens_s2 +=  (conj(s2) * s2).real();
                dens_s1 +=  (conj(s1) * s1).real();
                dens_s0 +=  (conj(s0) * s0).real();
                dens_s_1 += (conj(s_1) * s_1).real();
                dens_s_2 += (conj(s_2) * s_2).real();
            }	
        }
        dens[(N - y - 1) * 5 * N + 0 * N + x] = dens_s2;
        dens[(N - y - 1) * 5 * N + 1 * N + x] = dens_s1;
        dens[(N - y - 1) * 5 * N + 2 * N + x] = dens_s0;
        dens[(N - y - 1) * 5 * N + 3 * N + x] = dens_s_1;
        dens[(N - y - 1) * 5 * N + 4 * N + x] = dens_s_2;
    }
''', 'integ_dens_top')

integ_dens_side = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    
    #define Z_QUANTIZED 0
    #define Y_QUANTIZED 1
    #define X_QUANTIZED 2

    #define BASIS Z_QUANTIZED
    
    #define VALUES_IN_BLOCK 12

    struct BlockPsi {
        complex<double> values[VALUES_IN_BLOCK];
    };
    
    extern "C" __global__
    void integ_dens_side(double* dens, const BlockPsi* m2, const BlockPsi* m1, const BlockPsi* m0, const BlockPsi* m_1, const BlockPsi* m_2, const int N, const double angle) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int z = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= N || z >= N) return;

        int center_x = N / 2;
        int center_y = N / 2;

        double dens_s2  = 0;
        double dens_s1  = 0;
        double dens_s0  = 0;
        double dens_s_1 = 0;
        double dens_s_2 = 0;
        for (int y = 0; y < N; ++y)
        {
            int read_x = center_x + (double)(center_x - x) * cos(angle) - (double)(center_y - y) * sin(angle);
            int read_y = center_y + (double)(center_x - x) * sin(angle) + (double)(center_y - y) * cos(angle);
            if (read_x >= N || read_y >= N || 0 > read_x || 0 > read_y) continue;

            const int idx = z * N * N + read_y * N + read_x;
            for (int dualNode = 0; dualNode < VALUES_IN_BLOCK; ++dualNode)
            {
                complex<double> s2  = m2[idx].values[dualNode];
                complex<double> s1  = m1[idx].values[dualNode];
                complex<double> s0  = m0[idx].values[dualNode];
                complex<double> s_1 = m_1[idx].values[dualNode];
                complex<double> s_2 = m_2[idx].values[dualNode];

#if BASIS == X_QUANTIZED
                double c = sqrt(6) * 0.25;
                complex<double> x_s2  = 0.25 * s2 + 0.5 * s1 +   c * s0 + 0.5 * s_1 + 0.25 * s_2;
                complex<double> x_s1  = -0.5 * s2 - 0.5 * s1            + 0.5 * s_1 +  0.5 * s_2;
                complex<double> x_s0  =    c * s2            - 0.5 * s0             +    c * s_2;
                complex<double> x_s_1 = -0.5 * s2 + 0.5 * s1            - 0.5 * s_1 +  0.5 * s_2;
                complex<double> x_s_2 = 0.25 * s2 - 0.5 * s1 +   c * s0 - 0.5 * s_1 + 0.25 * s_2;

                s2 =  x_s2;
                s1 =  x_s1;
                s0 =  x_s0;
                s_1 = x_s_1;
                s_2 = x_s_2;
#elif BASIS == Y_QUANTIZED
                double c = sqrt(6) * 0.25;
                complex<double> im = complex<double>( 0, 1 );
                complex<double> y_s2  =      0.25 * s2 - im * 0.5 * s1 -   c * s0 + im * 0.5 * s_1 +     0.25 * s_2;
                complex<double> y_s1  = -im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 + im * 0.5 * s_2;
                complex<double> y_s0  =        -c * s2                 - 0.5 * s0                  -        c * s_2;
                complex<double> y_s_1 =  im * 0.5 * s2 -      0.5 * s1            -      0.5 * s_1 - im * 0.5 * s_2;
                complex<double> y_s_2 =      0.25 * s2 + im * 0.5 * s1 -   c * s0 - im * 0.5 * s_1 +     0.25 * s_2;

                s2 =  y_s2;
                s1 =  y_s1;
                s0 =  y_s0;
                s_1 = y_s_1;
                s_2 = y_s_2;
#endif

                dens_s2 +=  (conj(s2) * s2).real();
                dens_s1 +=  (conj(s1) * s1).real();
                dens_s0 +=  (conj(s0) * s0).real();
                dens_s_1 += (conj(s_1) * s_1).real();
                dens_s_2 += (conj(s_2) * s_2).real();
            }	
        }
        dens[5 * N * N + (N - z - 1) * 5 * N + 0 * N + x] = dens_s2;
        dens[5 * N * N + (N - z - 1) * 5 * N + 1 * N + x] = dens_s1;
        dens[5 * N * N + (N - z - 1) * 5 * N + 2 * N + x] = dens_s0;
        dens[5 * N * N + (N - z - 1) * 5 * N + 3 * N + x] = dens_s_1;
        dens[5 * N * N + (N - z - 1) * 5 * N + 4 * N + x] = dens_s_2;
    }
''', 'integ_dens_side')

reader = tvtk.PolyDataReader( file_name=filename )
reader.read_all_scalars = True
poly_data = reader.get_output()
reader.update()

positions = poly_data.points
m2 =  poly_data.point_data.get_array(0) + 1j * np.array(poly_data.point_data.get_array(1))
m1 =  poly_data.point_data.get_array(2) + 1j * np.array(poly_data.point_data.get_array(3))
m0 =  poly_data.point_data.get_array(4) + 1j * np.array(poly_data.point_data.get_array(5))
m_1 = poly_data.point_data.get_array(6) + 1j * np.array(poly_data.point_data.get_array(7))
m_2 = poly_data.point_data.get_array(8) + 1j * np.array(poly_data.point_data.get_array(9))

d_m2 =  cp.array(m2)
d_m1 =  cp.array(m1)
d_m0 =  cp.array(m0)
d_m_1 = cp.array(m_1)
d_m_2 = cp.array(m_2)

d_dens =  cp.zeros(5 * N * N * 2, dtype=cp.float64)

block_size = (8, 8,)
grid_size = (math.ceil(N / block_size[0]), math.ceil(N / block_size[1]),)

integ_dens_top(grid_size, block_size, (d_dens, d_m2, d_m1, d_m0, d_m_1, d_m_2, N, 0))
integ_dens_side(grid_size, block_size, (d_dens, d_m2, d_m1, d_m0, d_m_1, d_m_2, N, 0))

fig, ax = plt.subplots()
gray_scale = np.linspace(0, 1, 256)
colors = [[gray, gray, gray, 1.0] for gray in gray_scale]
my_cmap = ListedColormap(colors)
im = ax.imshow(d_dens.get().reshape((2 * N, 5 * N)), cmap=my_cmap, animated=True)

slider_ax = plt.axes([0.1, 0.01, 0.75, 0.04])
slider = widgets.Slider(
    ax=slider_ax,
    label='Angle',
    valmin=0,
    valmax=360,
    valinit=0,
    valfmt="%.2f°",
)

def update(angle):
    integ_dens_top(grid_size, block_size, (d_dens, d_m2, d_m1, d_m0, d_m_1, d_m_2, N, angle / 180 * np.pi))
    integ_dens_side(grid_size, block_size, (d_dens, d_m2, d_m1, d_m0, d_m_1, d_m_2, N, angle / 180 * np.pi))

    im.set_data(d_dens.get().reshape((2 * N, 5 * N)))


# register the update function with each slider
slider.on_changed(update)

plt.show()
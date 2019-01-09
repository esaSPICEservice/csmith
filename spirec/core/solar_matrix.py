import spiceypy
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.misc import imsave
import os

def plot_solar_ilum(utc, solarmax, pixels):
    mpl.rcParams['figure.figsize'] = (26.0, 26.0)

    spiceypy.furnsh("./kernels/rosmetak.txt")

    et = spiceypy.utc2et(utc)

    sensor_name = 'ROS_NAVCAM-A'

    sensor_id = spiceypy.bodn2c(sensor_name)
    (shape, frame, bsight, vectors, bounds) = spiceypy.getfov(sensor_id, 100)

    print(vectors, bounds)

    spiceypy.furnsh("./kernels/dsk/ROS_CG_M003_OSPCLPS_N_V1.BDS")

    nx, ny = (pixels, pixels)                                    # resolution of the image
    x = np.linspace(bounds[0][0], bounds[2][0], nx)
    y = np.linspace(bounds[0][1], bounds[2][1], ny)
    xv, yv = np.meshgrid(x, y)

    phase_matrix = np.zeros((nx, ny))
    emissn_matrix = np.zeros((nx, ny))
    solar_matrix = np.zeros((nx, ny))
    libsight = []

    nocross = []
    isvisible, isiluminated = [], []
    for i, x in enumerate(xv):
        for j, y in enumerate(yv):
            print('Processing point: ', i, j)
            ibsight = [x[i], y[j], bsight[2]]
            libsight.append(ibsight)
            try:
                (spoint, trgepc, srfvec ) = spiceypy.sincpt('DSK/UNPRIORITIZED', '67P/C-G', et, '67P/C-G_CK', 'NONE', 'ROSETTA', frame, ibsight)
                # (trgepc, srfvec, phase, solar, emissn) = spiceypy.ilumin('DSK/UNPRIORITIZED', '67P/C-G', et, '67P/C-G_CK', 'NONE', 'ROSETTA', spoint)
                (trgepc, srfvec, phase, solar, emissn, visiblef, iluminatedf) = spiceypy.illumf('DSK/UNPRIORITIZED', '67P/C-G', 'SUN', et, '67P/C-G_CK', 'LT+S', 'ROSETTA', spoint)
                #check visibility of spoint
                if visiblef == True:
                    isvisible.append(visiblef)
                if iluminatedf == True:
                    isiluminated.append(iluminatedf)
                    if solar > solarmax:
                        solar_matrix[i, j] = solarmax
                    else:
                        solar_matrix[i, j] = solar
                else:
                    solar_matrix[i, j] = solarmax           # not illuminated
                emissn_matrix[i,j] = emissn
                phase_matrix[i,j] = phase
            except:
                pass
                emissn_matrix[i,j] = 0
                phase_matrix[i,j] = math.pi
                solar_matrix[i,j] = 0
                nocross.append([i,j])

    print('total number of points: ', pixels*pixels)
    print('occulted points: ', pixels*pixels - len(isvisible))
    print('not iluminated points: ', pixels*pixels - len(isiluminated))

    nocross = np.asarray(nocross)
    solar_matrix[nocross[:,0], nocross[:,1]] = np.max(solar_matrix)
    fig, ax = plt.subplots()
    px = np.linspace(0, pixels-1, pixels)
    py = np.linspace(0, pixels-1, pixels)
    px, py = np.meshgrid(px, py)
    plt.pcolor(px, py, solar_matrix, cmap='Greys')
    # plt.pcolor(-xv/bounds[0][0]*1024/2+1024/2, -yv/bounds[0][0]*1024/2+1024/2, solar_matrix, cmap='Greys')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('equal')
    ax.set_xlim([0, pixels-1])
    ax.set_ylim([0, pixels-1])
    plt.savefig('churi_ilum_solar'+str(solarmax)+'.png', bbox_inches='tight')

    name = 'ROS_solar_matrix.png'
    limit = 256
    scaling = 255
    try:
        os.remove(name)
        print('modifying png file' + name)
    except:
        print('generating png file' + name)
    rescaled = (scaling / solar_matrix.max() * (solar_matrix - solar_matrix.min())).astype(np.uint8)
    rescaled = - np.flip(rescaled, 0) + scaling
    rescaled[rescaled > limit] = limit
    imsave(name, rescaled)

    return
# plot_solar_ilum(utc='2016-01-31T05:41:46.739', solarmax=np.pi/2, pixels=1024)
plot_solar_ilum(utc='2016-02-24T14:53:39.270', solarmax=np.pi/2, pixels=1024)
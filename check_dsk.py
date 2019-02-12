import spiceypy
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.misc import imsave
import os

def plot_solar_ilum(utc,mk, dsk, observer, target, target_frame,
                    pixels=150):
    mpl.rcParams['figure.figsize'] = (26.0, 26.0)

    spiceypy.furnsh(mk)

    utcstr = utc[10:13] + utc[14:16] + utc[17:19]
    et = spiceypy.utc2et(utc)

    spiceypy.furnsh(dsk)

    nx, ny = (pixels, pixels)                                    # resolution of the image
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    xv, yv = np.meshgrid(x, y)

    phase_matrix = np.zeros((nx, ny))
    emissn_matrix = np.zeros((nx, ny))
    solar_matrix = np.zeros((nx, ny))

    isvisible, isiluminated = [], []
    r, lt = spiceypy.spkpos(observer, et, 'J2000', 'NONE', target)

    #
    # We define a 'Nadir frame' w.r.t. J000 to make it general regardless of
    #
    #
    zN = r
    zN = zN/np.linalg.norm(zN)
    xN = np.array([1,0,0]) - np.dot(np.dot([1,0,0], zN), zN)/np.linalg.norm(zN)**2
    xN = xN/np.linalg.norm(xN)
    yN = np.cross(zN, xN)
    yN = yN/np.linalg.norm(yN)
    RotM = np.linalg.inv(np.array([xN, yN, zN]))
    spoints = []
    for i, x in enumerate(xv):
        for j, y in enumerate(yv):
            dpxy = [x[i], y[i], -np.linalg.norm(r)*1000]
            ibsight = spiceypy.mxv(RotM, dpxy)
            # ibsight = [x[i], y[j], -r*1000]
            try:
                (spoint, trgepc, srfvec ) = spiceypy.sincpt('DSK/UNPRIORITIZED', target, et, target_frame, 'NONE', observer, 'J2000', ibsight)
                spoints.append(spoint)
                (trgepc, srfvec, phase, solar, emissn, visiblef, iluminatedf) = spiceypy.illumf('DSK/UNPRIORITIZED', target, 'SUN', et, target_frame, 'NONE', observer, spoint)
                #check visibility of spoint
                if visiblef == True:
                    isvisible.append(visiblef)
                if iluminatedf == True:
                    isiluminated.append(iluminatedf)
                    if solar > np.pi/2:
                        solar_matrix[i, j] = np.pi - solar
                    else:
                        solar_matrix[i, j] = solar
                else:
                    solar_matrix[i, j] = np.pi/2          # not illuminated
                emissn_matrix[i,j] = emissn
                phase_matrix[i,j] = phase
            except:
                pass
                emissn_matrix[i,j] = 0
                phase_matrix[i,j] = math.pi
                solar_matrix[i,j] = np.pi/2

    spoints = np.asarray(spoints)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(spoints[:, 0], spoints[:, 1], spoints[:, 2], marker='.')
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title('')
    plt.axis('equal')
    plt.show()

    print('total number of points: ', pixels*pixels)
    print('occulted points: ', pixels*pixels - len(isvisible))
    print('not iluminated points: ', pixels*pixels - len(isiluminated))

    name = 'solar_matrix'
    try:
        os.remove(name)
        print('modifying png file' + name)
    except:
        print('generating png file' + name)
    plt.imshow(solar_matrix, cmap='viridis_r')
    plt.show()
    imsave(name+utcstr+'.png', solar_matrix)

    return


mk = "/Users/mcosta/Dropbox/SPICE/SPICE_CROSS_MISSION/apollo15/kernels/mk/ap15_ops_local.tm"
dsk = "/Users/mcosta/Dropbox/SPICE/SPICE_CROSS_MISSION/apollo15/kernels/dsk/apollo_csm_boom.bds"
observer = 'MOON'
target = 'A15'
target_frame = 'A15_SPACECRAFT'
utc = '1971-08-03T01:30:56'

#mk = "/Users/mcosta/JUICE/kernels/mk/juice_crema_4_0_ops_local.tm"
#dsk = "/Users/mcosta/JUICE/kernels/dsk/juice_sc_bus_v02.bds"
#observer = 'GANYMEDE'
#target = 'JUICE'
#target_frame = 'JUICE_SPACECRAFT'
#utc='2032-02-03T01:30:56'
#
mk = "/Users/mcosta/JUICE/kernels/mk/juice_crema_4_0_ops_local.tm"
dsk = "/Users/mcosta/JUICE/kernels/dsk/juice_sc_bus_v02.bds"
observer = 'JUPITER'
target = 'JUICE_SPACECRAFT'
target_frame = 'JUICE_SPACECRAFT'
utc='2030-02-08T00:20:19'


plot_solar_ilum(utc=utc, mk=mk, dsk=dsk, observer=observer, target=target,
                target_frame=target_frame, pixels=1000)

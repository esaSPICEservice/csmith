import os
import spiceypy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

from scipy.misc import imsave


def plot_solar_ilum(utc, metakernel, sensor, target, target_frame,
                    observer, pixel_lines=False, pixel_samples=False, dsk=False,
                    generate_image=False):
    '''

    :param solarmax:  Determination of this value is dependent on
    each image and
    :type solarmax:
    :param utc:
    :type utc:
    :param metakernel:
    :type metakernel:
    :param sensor:
    :type sensor:
    :param target:
    :type target:
    :param target_frame:
    :type target_frame:
    :param observer:
    :type observer:
    :param pixel_lines:
    :type pixel_lines:
    :param pixel_samples:
    :type pixel_samples:
    :param dsk:
    :type dsk:
    :return:
    :rtype:
    '''


    spiceypy.furnsh(metakernel)

    if dsk:
        spiceypy.furnsh(dsk)

    et = spiceypy.utc2et(utc)

    #
    # We set the size of the plot (not related to the actual image)
    #
    mpl.rcParams['figure.figsize'] = (26.0, 26.0)

    #
    # We retrieve the sensor information using GETFOV. More info available:
    #
    #   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/getfov_c.html
    #
    sensor_name = sensor
    sensor_id = spiceypy.bodn2c(sensor_name)
    (shape, frame, bsight, vectors, bounds) = spiceypy.getfov(sensor_id, 100)

    #
    # We check if the resolution of the sensor has been provided as an input
    # if not we try to obtain the resolution of the sensor from the IK
    #
    if not pixel_lines or not pixel_samples:
        try:
            pixel_samples = int(spiceypy.gdpool('INS'+str(sensor_id) + '_PIXEL_SAMPLES',0,1))
            pixel_lines = int(spiceypy.gdpool('INS' + str(sensor_id) + '_PIXEL_LINES',0,1))
        except:
            pass
            print("PIXEL_SAMPLES and/or PIXEL_LINES not defined for "
                  "{}".format(sensor))
            return

    #
    # We generate a matrix using the resolution of the framing camera as the
    # dimensions of the matrix
    #
    nx, ny = (pixel_samples, pixel_lines)
    x = np.linspace(bounds[0][0], bounds[2][0], nx)
    y = np.linspace(bounds[0][1], bounds[2][1], ny)
    xv, yv = np.meshgrid(x, y)

    #
    # We define the matrices that will be used as outputs and the
    #
    phase_matrix = np.zeros((nx, ny))
    emissn_matrix = np.zeros((nx, ny))
    solar_matrix = np.zeros((nx, ny))

    #
    # For each pixel we compute the possible intersection with the target, if
    # the target is intersected we then compute the illumination angles. We
    # use the following SPICE APIs: SINCPT and ILLUMF
    #
    #   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sincpt_c.html
    #   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/illumf_c.html
    #
    nocross = []
    isvisible, isiluminated = [], []
    for i, x in enumerate(xv):
        for j, y in enumerate(yv):

            #
            # List of pixel's boresight
            #
            ibsight = [x[i], y[j], bsight[2]]

            try:
                (spoint, trgepc, srfvec ) = spiceypy.sincpt('DSK/UNPRIORITIZED',
                                            target, et, target_frame, 'NONE', observer, frame, ibsight)
                (trgepc, srfvec, phase, solar,
                 emissn, visiblef, iluminatedf) = spiceypy.illumf('DSK/UNPRIORITIZED',
                                            target, 'SUN', et, target_frame, 'LT+S', observer, spoint)

                emissn_matrix[i, j] = emissn
                phase_matrix[i, j] = phase

                #
                # Add to list if the point is visible to the sensor
                #
                if visiblef == True:
                    isvisible.append(visiblef)
                #
                # Add to list if the point is illuminated and seen by the sensor
                #
                if iluminatedf == True:

                    isiluminated.append(iluminatedf)
                    solar_matrix[i, j] = solar

                else:
                    #
                    # And we set the not illuminated pixels with np.pi/2
                    #
                    solar_matrix[i, j] = np.pi/2

            except:
                pass

                #
                # If SINCPT raises an error, we set that we see nothing in
                # the pixel.
                #
                emissn_matrix[i,j] = 0
                phase_matrix[i,j] = np.pi
                solar_matrix[i,j] = np.pi/2

    print('Pixel report for {} w.r.t {} @ {}'.format(sensor,target,utc))
    print('   Total number of pixels: ', pixel_samples*pixel_lines)
    print('   Illuminated pixels:     ', len(isvisible))
    print('   Hidden pixels:          ', pixel_samples*pixel_lines - len(isvisible))
    print('   Shadowed points:        ', pixel_samples*pixel_lines - len(isiluminated))

    #
    # We transform the matrix from illumination angles to greyscale [0-255]
    #
    rescaled = (255 / solar_matrix.max() * (solar_matrix - solar_matrix.min())).astype(np.uint8)
    rescaled = - np.flip(rescaled, 0) + 255
    rescaled -= np.amin(rescaled)

    #
    # We generate the plot
    #
    if generate_image:
        name = '{}_{}_{}.png'.format(sensor.lower(),
                                     dsk.split('/')[-1].split('.')[0].lower(),
                                     utc.lower())
        imageio.imwrite(name, rescaled)
    else:
        plt.imshow(rescaled, cmap='gray')
        plt.show()

    return


target = 'PHOBOS'
target_frame = 'IAU_PHOBOS'
observer = 'MEX'
sensor = 'MEX_HRSC_SRC'
metakernel = '/Users/mcosta/MARS-EXPRESS/kernels/mk/MEX_OPS_LOCAL.TM'
dsk = '/Users/mcosta/MARS-EXPRESS/kernels/dsk/PHOBOS_K137_DLR_V01.BDS'
utc = '2010-08-27T20:31:56'

target = '67P/C-G'
target_frame = '67P/C-G_CK'
observer = 'ROSETTA'
sensor = 'ROS_NAVCAM-A'
metakernel = '/Users/mcosta/ROSETTA/kernels/mk/ROS_OPS_LOCAL.TM'
dsk = '/Users/mcosta/ROSETTA/kernels/dsk/ROS_CG_M001_OSPCLPS_N_V1.BDS'
utc = '2016-02-24T14:53:39'

plot_solar_ilum(utc=utc, metakernel=metakernel, sensor=sensor, target=target, target_frame=target_frame, observer=observer, dsk=dsk,
                generate_image=True, pixel_lines=100, pixel_samples=100)
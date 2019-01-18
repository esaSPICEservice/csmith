import os
import spiceypy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

from scipy.misc import imsave


def plot_solar_ilum(utc, metakernel, camera, target, target_frame,
                    pixel_lines=False, pixel_samples=False, dsk=False,
                    generate_image=False, report=False):
    '''

    :param utc: Image aquisition time in UTC format e.g.: 2016-01-01T00:00:00
    :type utc: str
    :param metakernel: SPICE Kernel Dataset Meta-Kernel
    :type metakernel: str
    :param camera: Name of the camera to be used. Usually found in the
    instrument kernel (IK) e.g.: 'ROS_NAVCAM-A'
    :type camera: str
    :param target: Target of the observation, e.g.:'67P/C-G'
    :type target: str
    :param target_frame: SPICE reference frame of the target. Usually found in
    the frames kernel (FK)
    :type target_frame: str
    :param pixel_lines: Number of pixel lines usually provided by the IK.
    :type pixel_lines: int
    :param pixel_samples: Number of pixel samples per line usually provided by
    the IK.
    :type pixel_samples: int
    :param dsk: Digital Shape Model to be used for the computation. Not required
    of included in the Meta-Kernel.
    :type dsk: str
    :param generate_image: Flag to determine whether if the image is saved or
    plotted.
    :type generate_image: bool
    :param report: Flag for processing repor.
    :type generate_image: bool
    :return:
    :rtype:
    '''


    spiceypy.furnsh(metakernel)

    if dsk:
        spiceypy.furnsh(dsk)
        method = 'DSK/UNPRIORITIZED'
    else:
        method = 'ELLIPSOID'

    et = spiceypy.utc2et(utc)

    #
    # We set the size of the plot (not related to the actual image)
    #
    mpl.rcParams['figure.figsize'] = (26.0, 26.0)

    #
    # We retrieve the camera information using GETFOV. More info available:
    #
    #   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/getfov_c.html
    #
    camera_name = camera
    camera_id = spiceypy.bodn2c(camera_name)
    (shape, frame, bsight, vectors, bounds) = spiceypy.getfov(camera_id, 100)


    #
    # TODO: In the future all the sensors should be epehmeris objects, see
    # https://issues.cosmos.esa.int/socci/browse/SPICEMNGT-77
    #
    if camera.split('_')[0] == 'ROS':
        observer = 'ROSETTA'
    elif camera.split('_')[0] == 'MEX':
        observer = 'MEX'
    elif camera.split('_')[0] == 'VEX':
        observer = 'VEX'


    #
    # We check if the resolution of the camera has been provided as an input
    # if not we try to obtain the resolution of the camera from the IK
    #
    if not pixel_lines or not pixel_samples:
        try:
            pixel_samples = int(spiceypy.gdpool('INS'+str(camera_id) + '_PIXEL_SAMPLES',0,1))
            pixel_lines = int(spiceypy.gdpool('INS' + str(camera_id) + '_PIXEL_LINES',0,1))
        except:
            pass
            print("PIXEL_SAMPLES and/or PIXEL_LINES not defined for "
                  "{}".format(camera))
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
                (spoint, trgepc, srfvec ) = spiceypy.sincpt(method, target, et,
                                            target_frame, 'NONE', observer, frame, ibsight)
                (trgepc, srfvec, phase, solar,
                 emissn, visiblef, iluminatedf) = spiceypy.illumf(method, target, 'SUN', et,
                                                  target_frame, 'LT+S', observer, spoint)

                emissn_matrix[i, j] = emissn
                phase_matrix[i, j] = phase

                #
                # Add to list if the point is visible to the camera
                #
                if visiblef == True:
                    isvisible.append(visiblef)
                #
                # Add to list if the point is illuminated and seen by the camera
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

    if report:
        print('Pixel report for {} w.r.t {} @ {}'.format(camera,target,utc))
        print('   Total number of pixels: ', pixel_samples*pixel_lines)
        print('   Illuminated pixels:     ', len(isvisible))
        print('   Hidden pixels:          ', pixel_samples*pixel_lines - len(isvisible))
        print('   Shadowed points:        ', pixel_samples*pixel_lines - len(isiluminated))

    #
    # We transform the matrix from illumination angles to greyscale [0-255]
    #
    rescaled = (255 / (solar_matrix.max()-solar_matrix.min()) * (solar_matrix - solar_matrix.min())).astype(np.uint8)
    rescaled = - np.flip(rescaled, 0) + 255

    #
    # We generate the plot
    #
    if generate_image:
        name = '{}_{}_{}.png'.format(camera.lower(),
                                     dsk.split('/')[-1].split('.')[0].lower(),
                                     utc.lower())
        imageio.imwrite(name, rescaled)
    else:
        plt.imshow(rescaled, cmap='gray')
        plt.axis('off')
        plt.show()

    return


target = 'PHOBOS'
target_frame = 'IAU_PHOBOS'
camera = 'MEX_HRSC_SRC'
metakernel = '/Users/mcosta/MARS-EXPRESS/kernels/mk/MEX_OPS_LOCAL.TM'
dsk = '/Users/mcosta/MARS-EXPRESS/kernels/dsk/PHOBOS_K137_DLR_V01.BDS'
utc = '2010-08-27T20:31:56'

target = '67P/C-G'
target_frame = '67P/C-G_CK'
camera = 'ROS_NAVCAM-A'
metakernel = '/Users/mcosta/ROSETTA/kernels/mk/ROS_OPS_LOCAL.TM'
dsk = '/Users/mcosta/ROSETTA/kernels/dsk/ROS_CG_M001_OSPCLPS_N_V1.BDS'
utc = '2016-02-24T14:53:39'

target = '67P/C-G'
target_frame = '67P/C-G_CK'
camera = 'ROS_OSIRIS_WAC_DIST'
metakernel = '/Users/mcosta/ROSETTA/kernels/mk/ROS_OPS_LOCAL.TM'
dsk = '/Users/mcosta/ROSETTA/kernels/dsk/ROS_CG_M001_OSPCLPS_N_V1.BDS'
utc = '2016-02-24T14:53:39'


plot_solar_ilum(utc=utc, metakernel=metakernel, camera=camera, target=target, target_frame=target_frame, dsk=dsk,
                pixel_lines=100, pixel_samples=100, generate_image=True, report=True)
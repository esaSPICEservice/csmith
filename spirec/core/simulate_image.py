import spiceypy
import numpy as np
import matplotlib.pyplot as plt
import imageio


def simulate_image(utc, metakernel, camera, target, target_frame,
                    pixel_lines=False, pixel_samples=False, dsk=False,
                    generate_image=False, plot_image=False, report=False, name=False):
    '''

    :param utc: Image acquisition time in UTC format e.g.: 2016-01-01T00:00:00
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
    :param plot_image: Flag to determine whether if the image is to be plotted or
    plotted.
    :type generate_image: bool
    :param report: Flag for processing report.
    :type generate_image: bool
    :param name: Name to be provided to the image
    :type generate_image: str
    :return: Name of the output image
    :rtype: str
    '''

    spiceypy.furnsh(metakernel)

    if dsk:
        spiceypy.furnsh(dsk)
        method = 'DSK/UNPRIORITIZED'
    else:
        method = 'ELLIPSOID'

    et = spiceypy.utc2et(utc)

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
    elif camera.split('_')[0] == 'JUICE':
        observer = 'JUICE'
    else:
        observer = camera

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
        if not name:

            name = '{}_{}_{}.png'.format(camera.lower(),
                                         name,
                                         utc.lower())

        imageio.imwrite(name, rescaled)

    if plot_image:
        plt.imshow(rescaled, cmap='gray')
        plt.axis('off')
        plt.show()

    return name
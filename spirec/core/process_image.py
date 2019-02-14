import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import scipy.optimize as opt
#from planetaryimage import PDS3Image

def ilum_bin(image, minlight=40, plot=False):
    """
    This function reads the input image. The image will be read GRAYSCALE.

    :param image: Path to the image
    :type image: str
    :param minlight: Real images do not have sharp edges on shadows. This value
    provides a threshold for indicating what we consider 'dark' in an image. By
    trial and error we have found that 40 fits most of the images.
    :type minlight int
    :return: Binary Image in Matrix format. Matrix is composed of 0
    (not illuminated) or 1 (illuminated)
    :rtype: ndarray
    """

    #if '.IMG' in image:
    #    image = PDS3Image.open(image)
    #    img = imread(image.image, mode='L')

    #else:
    img = imread(image, mode='L')
    #
    # Mode L indicates that is read in grayscale
    #


    print('Shape of the image:              ', np.shape(img))
    print('Maximum Black/White value [0-1]: ', img.max())

    img = np.asarray(img)

    black = (np.argwhere(img < minlight+1))
    img[black[:, 0], black[:, 1]] = 0
    white = (np.argwhere(img > 0))
    img[white[:, 0], white[:, 1]] = 1

    if plot:
        plt.imshow(img)
        plt.show()

    return img



def opt_fit(plane, plane_f, image, maxd = 1000):
    """
    This function compares the input images in order to find the required
    displacement to fit them together, providing you with an X-offset, a
    Y-offset and a rotation along the Z axis (in pixel coordinates). In
    order to do this the function optimizes the displacement + rotation such
    that when substracting the real and simulated gradient images, the
    sum of the values is minimised.

    :param plane: Actual image as a 'binary' matrix (processed with the ilum_bin
    function)
    :type plane: ndarray
    :param plane_f: Simulated image as a 'binary' matrix (generated with the
    solar_matrix function)
    :type plane_f: ndarray
    :param image: Name of the actual image
    :type image: str
    :param maxd: Maximum allowed displacement for the simulated image to fit the
    actual image. In order for the algorythim to work, pixel coordinates are
    provided within the matrix, if the coordinates are outside the matrix we would
    have an 'out of range', this parameter allows us to 'expand' the matrix in
    such way that the optimiser can iterate. Units are pixels. The larger the
    more computationally expensive.
    :type maxd: int
    :return: X and Y offset and rotation around the Z axis as a vector
    :rtype: ndarray
    """

    dx0, dy0 = [0, 0]

    #
    # The following image result is the first difference in between the
    # the simulated image and the actual image.
    #
    imsave(image.split('.')[0]+'_FIRST_GUESS.PNG', plane - plane_f)

    #
    # check if the real image is not completely in FOV (basically detects if
    # the target is crossing any FOV boundary) the image is modified as well.
    #
    if np.count_nonzero(plane[0, :]) != 0:

        #
        # space from origin of rows to first illuminated pixel.
        #
        # We check the first row and if an element is not 0, he understands
        # that there is a crossing. We look for the maximum illuminated pixel
        # closer to the zero row (such that we quantify the ammount of target
        # that we can see in the image). We do so with the actual and simulated
        # images.
        #
        vspace = max((np.argwhere(plane != 0))[:, 0])
        vspace_f = max((np.argwhere(plane_f != 0))[:, 0])

        if vspace < vspace_f:
            dx0 = vspace - vspace_f

            plane_f_aux = plane_f

            plane_f = np.zeros_like(plane_f)
            plane_f[0:vspace, :] = plane_f_aux[vspace_f - vspace:vspace_f, :]


    elif np.count_nonzero(plane[-1, :]) != 0:

        #
        # space from origin of rows to first illuminated pixel
        #
        vspace = min((np.argwhere(plane != 0))[:, 0])
        vspace_f = min((np.argwhere(plane_f != 0))[:, 0])

        if vspace > vspace_f:
            dx0 = vspace - vspace_f

            plane_f_aux = plane_f

            plane_f = np.zeros_like(plane_f)
            plane_f[vspace:, :] = plane_f_aux[
                                  vspace_f:vspace_f + len(plane) - vspace, :]


    if np.count_nonzero(plane[:, -1]) != 0:

        #
        # space from origin of columns to first illuminated pixel
        #
        hspace = min((np.argwhere(plane != 0))[:, 1])
        hspace_f = min((np.argwhere(plane_f != 0))[:, 1])

        if hspace > hspace_f:
            dy0 = hspace - hspace_f

            plane_f_aux = plane_f

            plane_f = np.zeros_like(plane_f)
            plane_f[:, hspace:] = plane_f_aux[:,
                                  hspace_f:hspace_f + len(plane) - hspace]


    elif np.count_nonzero(plane[:, 0]) != 0:

        #
        # space from origin of columns to first illuminated pixel
        #
        hspace = max((np.argwhere(plane != 0))[:, 1])
        hspace_f = max((np.argwhere(plane_f != 0))[:, 1])

        if hspace < hspace_f:
            dy0 = hspace - hspace_f

            plane_f_aux = plane_f

            plane_f = np.zeros_like(plane_f)
            plane_f[:, 0:hspace] = plane_f_aux[:, hspace_f - hspace:hspace_f]


    #
    # increment on the size of the image to allow large displacements
    #
    plane_ext = np.zeros((len(plane) + maxd, len(plane) + maxd))
    plane_f_ext = np.zeros((len(plane) + maxd, len(plane) + maxd))
    plane_ext[int(maxd/2):int(len(plane_ext)-maxd/2), int(maxd/2):int(len(plane_ext)-maxd/2)] = plane
    plane_f_ext[int(maxd / 2):int(len(plane_f_ext) - maxd / 2), int(maxd / 2):int(len(plane_f_ext) - maxd / 2)] = plane_f
    illumpoints_avg = (np.count_nonzero(plane_f_ext) + np.count_nonzero(plane_f)) / 2


    #
    # We obtain the number of pixels that 'fit'
    # such that 1 (actual) - 1 (sim) = 0
    #
    print('Initial % of fitted points: ' + str((1 - np.count_nonzero(plane_ext-plane_f_ext)/2 / illumpoints_avg) * 100))

    #
    # save as an image the illumination difference at the initial guess
    # displacement
    #
    imsave(image.split('.')[0]+'_INI_ILLUM_DIFF.PNG', plane_ext-plane_f_ext)

    #
    # center of illumination displacement from real to fake image to be used
    # as the initial guess for the optimiser
    #
    white = (np.argwhere(plane > 0))
    xcg = round(sum(white[:, 1]) / len(white))
    ycg = round(sum(white[:, 0]) / len(white))
    white = (np.argwhere(plane_f > 0))
    xcgf = round(sum(white[:, 1]) / len(white))
    ycgf = round(sum(white[:, 0]) / len(white))
    print('Displacement of center of illumination: ', ycg-ycgf, xcg-xcgf)


    #
    # cost function to be minimized
    #
    def cost_function(x0, M, Mf, plot=False):
        """
        Generates a difference in between the simulated image and the actual
        image and we minimise the number of non_zero elements.


        :param x0: Contains the optimization parameters, dx, dy, theta
        (displacements)
        :type x0: ndarray
        :param M: extened input actual image Matrix
        :type M: ndarray
        :param Mf: extended input simulated Matrix
        :type Mf: ndarray
        :param plot: Boolean to either save the iterations as images
        :type plot: bool
        :return: Number of elements of the matrix different than zero
        :rtype: int
        """


        #
        # rotations and displacements to be applied
        #
        dx, dy, theta = x0.astype(int)   # dx and dy in pixels, theta in degrees
        theta = np.deg2rad(theta/10)     # changes in theta of O(0.1) degrees

        print('Parameters at last iteration: ', x0)

        x = np.linspace(int(maxd/2), int(len(M)-maxd/2), len(plane)+1)
        y = np.linspace(int(maxd / 2), int(len(M) - maxd / 2), len(plane)+1)
        x, y = np.meshgrid(x, y)

        #
        # reshape as vectors for manipulation
        #
        x = np.reshape(x, (len(x)**2, 1))
        y = np.reshape(y, (len(y)**2, 1))
        xcgf = round(len(M)/2)
        ycgf = round(len(M)/2)

        #
        # write the points positions with respect to the center of the image
        #
        xaux, yaux = x - xcgf, y - ycgf

        #
        # rotate the image about the center and add displacement
        #
        yf = yaux*np.cos(theta) + xaux*np.sin(theta) + dy
        xf = - yaux*np.sin(theta) + xaux*np.cos(theta) + dx

        #
        # refer again the points with matrix index as coordinates (left down corner)
        #
        xf, yf = xf + xcgf, yf + ycgf

        #
        # reshape as square matrix
        #
        xf = np.reshape(xf, (len(xf), 1))
        yf = np.reshape(yf, (len(yf), 1))
        y, x = y.astype(int), x.astype(int)
        yf, xf = yf.astype(int), xf.astype(int)

        #
        # move the simulated image values to the new coordinates
        #
        Mf_aux = Mf
        Mf = np.zeros((len(Mf), len(Mf)))
        Mf[xf, yf] = Mf_aux[x, y]
        x = np.reshape(x, (int((len(x)**0.5)), int(len(x)**0.5)))
        y = np.reshape(y, (int((len(y)**0.5)), int(len(y)**0.5)))

        #
        # objective function to minimize (if the pitcures match perfectly, this should be 0)
        #
        C = M[y, x] - Mf[y, x]

        #
        # plot the objective function
        #
        if plot:
            imsave(image.split('.')[0]+'_CONVERGED_ILLUM_DIFF.PNG', C)

        return np.count_nonzero(C)


    #
    # optimization technique
    #
    def optimizer():
        """
        The method is useful for calculating the local minimum of a continuous
        but complex function, especially one without an underlying mathematical
        definition, because it is not necessary to take derivatives.

        :return: Optimization parameters after convergence (dx, dy, theta)
        :rtype: ndarray
        """

        res = opt.minimize(cost_function, np.array([0, 0, 0]),
                           args=((plane_ext, plane_f_ext, 0)), method='Powell',
                           options={'disp':True, 'maxiter':None, 'xtol':1})
        print(res)
        fval = res.fun

        #
        # print final result
        #
        print('Percentage of fitted points: ' + str((1 - fval/2 / illumpoints_avg) * 100))

        cost_function(res.x, plane_ext, plane_f_ext, plot=True)
        return res


    res = optimizer()

    dX = res.x + np.array([dx0, dy0, 0])

    return dX
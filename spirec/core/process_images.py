import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import scipy.optimize as opt
import spiceypy

def ilum_bin(image):
    img = imread(image, mode='L')
    print('shape of img: ', np.shape(img))
    print('maximum bw value: ', img.max())
    img = np.asarray(img)

    minlight = 40   # careful configuring this parameter, it may cause the blur of the image
    black = (np.argwhere(img < minlight+1))
    img[black[:, 0], black[:, 1]] = 0
    white = (np.argwhere(img > 0))
    img[white[:, 0], white[:, 1]] = 1
    plt.imshow(img)
    plt.show()
    return img
# plane = ilum_bin("ROS_CAM1_20160131T054146.jpg")
# plane_f = ilum_bin("ROS_solar_matrix_20160131T054146.png")
plane = ilum_bin("ROS_CAM1_20160211T112912.jpg")
plane_f = ilum_bin("ROS_solar_matrix_20160211T112912.png")
# plane = ilum_bin("ROS_CAM1_20160228T105856.jpg")
# plane_f = ilum_bin("ROS_solar_matrix_20160228T105856.png")

def opt_fit(plane, plane_f):
    # optimize displacement + rotation such that when substracting
    # the real and fake gradient images, the sum of the values is
    # minimized
    imsave('first_guess.png', plane - plane_f)
    #
    # check if the real image is not completely in FOV
    #
    if np.count_nonzero(plane[0, :]) != 0:
        # space from origin of rows to first illuminated pixel
        vspace = max((np.argwhere(plane != 0))[:, 0])
        vspace_f = max((np.argwhere(plane_f != 0))[:, 0])
        plane_f_aux = plane_f
        plane_f = np.zeros_like(plane_f)
        plane_f[0:vspace, :] = plane_f_aux[vspace_f - vspace:vspace_f, :]
    elif np.count_nonzero(plane[-1, :]) != 0:
        # space from origin of rows to first illuminated pixel
        vspace = min((np.argwhere(plane != 0))[:, 0])
        vspace_f = min((np.argwhere(plane_f != 0))[:, 0])
        plane_f_aux = plane_f
        plane_f = np.zeros_like(plane_f)
        plane_f[vspace:, :] = plane_f_aux[vspace_f:vspace_f + len(plane) - vspace, :]
    if np.count_nonzero(plane[:, -1]) != 0:
        # space from origin of columns to first illuminated pixel
        hspace = min((np.argwhere(plane != 0))[:, 1])
        hspace_f = min((np.argwhere(plane_f != 0))[:, 1])
        plane_f_aux = plane_f
        plane_f = np.zeros_like(plane_f)
        plane_f[:, hspace:] = plane_f_aux[:, hspace_f:hspace_f + len(plane) - hspace]
    elif np.count_nonzero(plane[:, 0]) != 0:
        # space from origin of columns to first illuminated pixel
        hspace = max((np.argwhere(plane != 0))[:, 1])
        hspace_f = max((np.argwhere(plane_f != 0))[:, 1])
        plane_f_aux = plane_f
        plane_f = np.zeros_like(plane_f)
        plane_f[:, 0:hspace] = plane_f_aux[:, hspace_f - hspace:hspace_f]
    #
    # increment on the size of the image to allow large displacements
    #
    maxd = 1000
    plane_ext = np.zeros((len(plane) + maxd, len(plane) + maxd))
    plane_f_ext = np.zeros((len(plane) + maxd, len(plane) + maxd))
    plane_ext[int(maxd/2):int(len(plane_ext)-maxd/2), int(maxd/2):int(len(plane_ext)-maxd/2)] = plane
    plane_f_ext[int(maxd / 2):int(len(plane_f_ext) - maxd / 2), int(maxd / 2):int(len(plane_f_ext) - maxd / 2)] = plane_f
    illumpoints_avg = (np.count_nonzero(plane_f_ext) + np.count_nonzero(plane_f)) / 2
    print('initial percentaje of fitted points: ' + str((1 - np.count_nonzero(plane_ext-plane_f_ext)/2 / illumpoints_avg) * 100))

    #
    # plot the illumination difference at the initial guess displacement
    #
    plt.imshow(plane_ext-plane_f_ext)
    plt.title('Initial illumination difference')
    plt.show()

    #
    # center of illumination displacement from real to fake image
    #
    white = (np.argwhere(plane > 0))
    xcg = round(sum(white[:, 1]) / len(white))
    ycg = round(sum(white[:, 0]) / len(white))
    white = (np.argwhere(plane_f > 0))
    xcgf = round(sum(white[:, 1]) / len(white))
    ycgf = round(sum(white[:, 0]) / len(white))
    print('displacement of center of illumination: ', ycg-ycgf, xcg-xcgf)

    #
    # cost function to be minimized
    #
    def cost_function(x0, M, Mf, plot):

        #
        # rotations and displacements to be applied
        #
        dx, dy, theta = x0.astype(int)   # dx and dy in pixels, theta in degrees
        theta = np.deg2rad(theta/10)     # changes in theta of O(0.1) degrees
        print('parameters at last iteration: ', x0)
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
        # move the fake image values to the new coordinates
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
        if plot == 'save':
            imsave('convergence_images_nadir/converged_illum_difference_' + str(x0) + '.png', C)
        elif plot == 'show':
            plt.imshow(C)
            plt.title('iterating on illumination difference, displacement: ' + str(round(dx)) + ', ' + str(round(dy)))
            plt.show()
        return np.count_nonzero(C)
    #
    # optimization technique
    #
    def optimizer():
        ''' The method is useful for calculating the local minimum of a continuous
        but complex function, especially one without an underlying mathematical
        definition, because it is not necessary to take derivatives. '''
        res = opt.minimize(cost_function, np.array([0, 0, 0]),
                           args=((plane_ext, plane_f_ext, 0)), method='Powell',
                           options={'disp':True, 'maxiter':None, 'xtol':1})
        print(res)
        fval = res.fun

        #
        # print final result
        #
        print('percentaje of fitted points: ' + str((1 - fval/2 / illumpoints_avg) * 100))
        cost_function(res.x, plane_ext, plane_f_ext, plot='show')
        return res
    res = optimizer()
    return res.x
rotation_ang = opt_fit(plane, plane_f)

def generate_quaternion(corrections):
    dy, dx, theta = corrections
    theta = theta/10
    print('vertical displacement: ', int(dy))
    print('horizontal displacement: ', int(dx))
    print('rotation: ', theta)
    dy = np.deg2rad(dy*5/1024)
    dx = np.deg2rad(dx*5/1024)
    theta = np.deg2rad(theta)

    M = spiceypy.eul2m(theta,dy,dx,3,2,1)
    q = spiceypy.m2q(M)
    print('quaternion defining the correction rotation: ', q)
    return
generate_quaternion(rotation_ang)

import numpy as np
import spiceypy

def generate_quaternion(corrections):
    """
    This function generates the quaternion from the resulting optimization
    parameters.

    :param corrections: Results of the optimisation
    :type corrections: ndarray
    :return: A unit quaternion representing the rotation matrix
    :rtype: 4-Element Array of floats
    """


    dy, dx, theta = corrections
    theta = theta/10

    print('Vertical displacement   [dX]: ', int(dy))
    print('Horizontal displacement [dY]: ', int(dx))
    print('Rotation             [theta]: ', theta)

    dy = np.deg2rad(dy*5/1024)
    dx = np.deg2rad(dx*5/1024)
    theta = np.deg2rad(theta)

    M = spiceypy.eul2m(theta,dy,dx,3,2,1)
    q = spiceypy.m2q(M)

    print('Quaternion defining the correction rotation: ', q)

    return q

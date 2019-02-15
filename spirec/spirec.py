from .core import process_image
from .core import generate_quaternion
from .core import simulate_image
from .core import read_pds_label

from .utils import target2frame

import glob

import os
import shutil

def spirec(config):

    os.chdir(config['images'])
    images = glob.glob('*.JPG')
    os.chdir(config['workspace'])
    simulations = glob.glob('*_SIMULATION.PNG')


    with open('quaternions.txt', 'w+') as f:
        for image in images:

            try:
                (utc, target, camera) = read_pds_label(config['labels']+os.sep+image.split('.')[0] + '.LBL')


                simulation = image.split('.')[0] + '_SIMULATION.PNG'
                if not simulation in simulations:

                    simulation = simulate_image(utc=utc, metakernel=config['mk'],
                            camera=camera, target=target,
                            target_frame=target2frame(target),
                            dsk=config[target], generate_image=True, report=True,
                            name=image.split('.')[0]+'_SIMULATION.PNG')

                shutil.copy(config['images']+os.sep+image, image)

                plane = process_image.ilum_bin(image)
                plane_f = process_image.ilum_bin(simulation)

                corrections = process_image.opt_fit(plane, plane_f, image)

                quaternion = generate_quaternion(corrections)

                f.write('{} {} {} {} {}\n'.format(utc, quaternion[0],
                                                       quaternion[1],
                                                       quaternion[2],
                                                       quaternion[3]))
            except Exception as e:
                print('{} not processed. Type Error: {}'.format(image, str(e)))

    return



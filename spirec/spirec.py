from .core import process_image
from .core import generate_quaternion
from .core import simulate_image
from .core import read_pds_label
import glob

import os
import shutil

def spirec(config):


    os.chdir(config['images_dir'])
    images = glob.glob('*.JPG')
    os.chdir(config['working_dir'])
    simulations = glob.glob('*_SIMULATION.PNG')

    with open('quaternions.txt', 'w+') as f:
        for image in images:

            try:
                (utc, target) = read_pds_label(config['labels_dir']+os.sep+image.split('.')[0] + '.LBL')

                if target == '67P/CHURYUMOV-GERASIMENKO 1 (1969 R1)':

                    simulation = image.split('.')[0] + '_SIMULATION.PNG'
                    if not simulation in simulations:

                        simulation = simulate_image(utc=utc, metakernel=config['mk'],
                            camera=config['camera'], target=config['target'],
                            target_frame=config['target_frame'],
                            dsk=config['dsk'], generate_image=True, report=True,
                            name=image.split('.')[0]+'_SIMULATION.PNG')

                    shutil.copy(config['images_dir']+os.sep+image, image)

                    plane = process_image.ilum_bin(image)
                    plane_f = process_image.ilum_bin(simulation)

                    corrections = process_image.opt_fit(plane, plane_f, image)

                    quaternion = generate_quaternion(corrections)

                    f.write('{} {} {} {} {}\n'.format(utc, quaternion[0],
                                                           quaternion[1],
                                                           quaternion[2],
                                                           quaternion[3]))
            except:
                print('{} not processed'.format(image))

    return



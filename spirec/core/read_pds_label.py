from spirec.utils import pds2spice

def read_pds_label(label):

    with open(label, 'r') as f:
        for line in f:

            if 'IMAGE_TIME        ' in line:
                utc = line.split('=')[-1].strip()
            #
            # OSIRIS labels do not have IMAGE TIME
            #
            elif 'START_TIME      ' in line:
                utc = line.split('=')[-1].strip()

            if 'TARGET_NAME       ' in line:
                target = line.split('"')[-2]

            if 'DETECTOR_ID       ' in line:
                camera = line.split('"')[-2]
            elif 'CHANNEL_ID      ' in line:
                camera = line.split('"')[-2]

    return utc, pds2spice(target), pds2spice(camera)
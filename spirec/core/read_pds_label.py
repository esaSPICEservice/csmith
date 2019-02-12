def read_pds_label(label):

    with open(label, 'r') as f:
        for line in f:

            if 'IMAGE_TIME' in line:
                utc = line.split('=')[-1].strip()
            if 'TARGET_NAME' in line:
                target = line.split('"')[-2]
                break

    return utc, target
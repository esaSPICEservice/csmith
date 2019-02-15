def pds2spice(pds_name):

    name_map = {
        "MEX_HRSC_SRC": "MEX_HRSC_SRC",
        "MEX_VMC": "MEX_VMC",
        "EEV-243": "ROS_OSIRIS_NAC_DIST",
        "CAM1":    "ROS_NAVCAM-A",
        "CAM2":    "ROS_NAVCAM-B",
        "EEV-242": "ROS_OSIRIS_WAC_DIST",
        "VEX_VMC_NIR-2": "VEX_VMC_NIR-2",
        "VEX_VMC_UV": "VEX_VMC_UV",
        "21 LUTETIA": "TLS",
        "EARTH": "EARTH",
        "PHOBOS": "PHOBOS",
        "MARS": "MARS",
        "67P/CHURYUMOV-GERASIMENKO 1 (1969 R1)":"67P/C-G"
    }

    return name_map[pds_name]

def target2frame(target):

    frame_map = {
        "67P/C-G": "67P/C-G_CK",
        "PHOBOS": "IAU_PHOBOS",
        "MARS": "IAU_MARS",
        "21 LUTETIA": "LUTETIA_FIXED",
        "EARTH": "IAU_EARTH",
        "VENUS": "IAU_VENUS"
    }

    return frame_map[target]
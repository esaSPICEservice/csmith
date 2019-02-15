from spirec import spirec
import json
import os

cwd = os.getcwd()

def test_navcam_earth():
    os.chdir(cwd)
    with open('RO-E-NAVCAM-2-EAR1-V1.1/spirec.json', 'r') as f:
        config = json.load(f)

    spirec(config)

def test_osiris_wac_chury():
    os.chdir(cwd)
    with open('RO-C-OSIWAC-3-EXT3-67PCHURYUMOV-M33-V1.0/spirec.json', 'r') as f:
        config = json.load(f)

    spirec(config)
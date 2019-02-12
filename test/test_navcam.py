from spirec import spirec
import json


def test_navcam():
    with open('test_navcam.json', 'r') as f:
        config = json.load(f)

    spirec(config)
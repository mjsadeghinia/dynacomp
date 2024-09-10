import json


def load_settings(fname):
    with open(fname, 'r') as file:
        settings = json.load(file)
    return settings
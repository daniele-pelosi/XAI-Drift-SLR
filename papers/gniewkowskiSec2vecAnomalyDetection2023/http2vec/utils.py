import os
import yaml


def file_generator(path):

    if os.path.isfile(path):
        yield path
        return 

    for root, dirs, files in os.walk(path):
        for name in files:
            yield os.path.join(root, name)


def load_config(config_path):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

from io import BytesIO

from urllib.request import urlopen
from zipfile import ZipFile

import os.path as osp

from mot_neural_solver.path_cfg import DATA_PATH

DATASET_URLS = {'': 'https://motchallenge.net/data/2DMOT2015.zip', # Zipped MOT15 Data contains a 2DMOT2015 directory inside
                'MOT17Det': 'https://motchallenge.net/data/MOT17Det.zip',
                'MOT17Labels': 'https://motchallenge.net/data/MOT17Labels.zip'}

for dir_name, url in DATASET_URLS.items():
    print(f"Downloading data from {url}")
    with urlopen(url) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(path = osp.join(DATA_PATH, dir_name))

    print("Done!")





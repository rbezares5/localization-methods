"""
This script generates a csv file with train images filenames
"""

import cv2 as cv
from skimage.feature import hog
import glob
from natsort import natsorted
import pandas as pd
import numpy as np
from scipy.spatial import distance
from math import sqrt
import time
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from torchvision.io import read_image
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import random

def get_file_names(dataset_path, codename):
    """
    Get filenames of images located in test_dataset_path
    """
    file_names=[]
    for file in natsorted(glob.glob(dataset_path)):
        file_names.append(file)

    pd.DataFrame(file_names).to_csv('{name}_dataset_filenames.csv'.format(name=codename), index=None)

def main():
    get_file_names('Friburgo/Friburgo_Test_ext/*.jpeg',codename='friburgo_test')


if __name__ == "__main__":
    main()
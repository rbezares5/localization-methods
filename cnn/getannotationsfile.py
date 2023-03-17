"""
This script takes filenames csv and labels csv, and generates a new, annotations csv that can be used to create a custom dataset

Then(in another file probably), I will define the custom dataset
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
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

def create_annotations(imgs_file, info_file, codename):
    """
    Create annotations file combining 2 csv files
    """
    df = pd.concat(map(pd.read_csv, [imgs_file, info_file]), ignore_index=True, axis=1)
    print(df)

    pd.DataFrame(df).to_csv('{name}_annotations_file.csv'.format(name=codename), index=None, header=None)

def main():
    create_annotations(imgs_file='friburgo_train_dataset_filenames.csv',
                       info_file='labels_gist50.csv',
                       codename='friburgo_train')


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import multivariate_normal as multinorm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from functions import readdata
from functions import plotmfccs


doublecleanedtrain = readdata(
    '/Users/benmatz/Box/Duke/Fall2022/ECE480/Project/ece480-final-project/Data/Train_Arabic_Digit.txt', 13)
plotmfccs(doublecleanedtrain, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
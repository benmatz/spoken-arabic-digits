import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import multivariate_normal as multinorm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from functions import readdata
from functions import get_k_distribution
from functions import get_em_distribution
from functions import accuracy
from functions import predictdigit


def plotmfccsvsanalysiswindow(doublecleaned, nums):
    for i in nums:
        edited_sample = doublecleaned[660*i]
        mfccs = []
        for j in range(len(edited_sample[0])):
            temp = []
            for k in range(len(edited_sample)):
                temp.append(edited_sample[k][j])
            mfccs.append(temp)
        windows = np.zeros(len(edited_sample))
        for l in range(windows.size):
            windows[l] = l
        fig, ax = plt.subplots(1, 1)
        for q in range(len(mfccs)):
            # if q == 0 or q == 1 or q == 2 or q == 3:
            ax.plot(windows, mfccs[q], label="MFCC" + str(q + 1))
        ax.set(title="Sample of 13 MFCCs in Utterance of Arabic Number " + str(i), xlabel="Analysis Window", ylabel="MFCC")
        ax.legend(loc='best', frameon=False)
        fig.tight_layout()
        plt.show()


def plotpairwisemfccs(doublecleaned, mfcc1, mfcc2, nums):
    for i in nums:
        mfccs = []
        for c in range(13):
            mfccs.append([])
        for sample in range(10):
            edited_sample = doublecleaned[660*i + sample]
            for j in range(len(edited_sample[0])):
                temp = []
                for k in range(len(edited_sample)):
                    mfccs[j].append(edited_sample[k][j])
        fig, ax = plt.subplots(1, 1)
        ax.scatter(mfccs[mfcc1 - 1], mfccs[mfcc2 - 1], s = 1)
        ax.set(title="Sample of 13 MFCCs in Utterance of Arabic Number " + str(i), xlabel="MFCC" + str(mfcc1), ylabel="MFCC" + str(mfcc2))
        fig.tight_layout()
        plt.show()


doublecleanedtrain = readdata('/Users/benmatz/Box/Duke/Fall2022/ECE480/Project/ece480-final-project/Data/Train_Arabic_Digit.txt', 13)
doublecleanedtest = readdata('/Users/benmatz/Box/Duke/Fall2022/ECE480/Project/ece480-final-project/Data/Test_Arabic_Digit.txt', 13)
plotmfccsvsanalysiswindow(doublecleanedtrain, [7])
#plotpairwisemfccs(doublecleanedtrain, 12, 13, [0, 1])
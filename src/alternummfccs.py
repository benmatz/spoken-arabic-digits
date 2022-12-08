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


fig, ax = plt.subplots(1, 1)

maxdigit = 10
ctype = 'full'
num_clusts = 3
num_avg = 1


km = []
em = []

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

for k in nums:
    doublecleanedtrain = readdata(
        '/Users/benmatz/Box/Duke/Fall2022/ECE480/Project/ece480-final-project/Data/Train_Arabic_Digit.txt', k)
    doublecleanedtest = readdata(
        '/Users/benmatz/Box/Duke/Fall2022/ECE480/Project/ece480-final-project/Data/Test_Arabic_Digit.txt', k)
    k_digits_distributions = []
    em_digits_distributions = []
    desired = int((maxdigit + 1) * (len(doublecleanedtrain) / 6600))
    for i in range(desired):
        k_digits_distributions = get_k_distribution(doublecleanedtrain[(i*660):(660*(i+1))], k_digits_distributions, num_clusts, ctype, num_avg)
        em_digits_distributions = get_em_distribution(doublecleanedtrain[(i*660):(660*(i+1))], em_digits_distributions, num_clusts, ctype, num_avg)
    km.append(accuracy(doublecleanedtest, k_digits_distributions))
    em.append(accuracy(doublecleanedtest, em_digits_distributions))

ax.plot(nums, km, 'r', label="K-Means")
ax.plot(nums, em, 'b', label="EM")
ax.set(title="Model Accuracy vs. Number of MFCCs", xlabel="Number of MFCCs", ylabel="Accuracy")
ax.legend(loc='best', frameon=False)
fig.tight_layout()
fig.savefig("num_mfccs")
plt.show()
print(km)
print(em)
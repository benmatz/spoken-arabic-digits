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

k = 13
maxdigit = 10
ctype = 'full'
num_avg = 1
num_clusts = [5, 5, 5, 6, 6, 5, 4, 4, 7, 4]
cov_types = ["syllables", "(syllables*2)-1", "phonemes", "(phonemes*2)-1"]
for i in range(len(num_clusts)):
    num_clusts[i] = (num_clusts[i]*2) - 1

doublecleanedtrain = readdata('/Users/benmatz/Box/Duke/Fall2022/ECE480/Project/ece480-final-project/Data/Train_Arabic_Digit.txt', k)
doublecleanedtest = readdata('/Users/benmatz/Box/Duke/Fall2022/ECE480/Project/ece480-final-project/Data/Test_Arabic_Digit.txt', k)


km = []
em = []

desired = int((maxdigit+1)*(len(doublecleanedtrain)/6600))
k_digits_distributions = []
em_digits_distributions = []
for i in range(desired):
    k_digits_distributions = get_k_distribution(doublecleanedtrain[(i*660):(660*(i+1))], k_digits_distributions, num_clusts[i], ctype, num_avg)
    em_digits_distributions = get_em_distribution(doublecleanedtrain[(i*660):(660*(i+1))], em_digits_distributions, num_clusts[i], ctype, num_avg)
km.append(accuracy(doublecleanedtest, k_digits_distributions))
em.append(accuracy(doublecleanedtest, em_digits_distributions))
# km = [0.8885857207821737, 0.8735788994997726, 0.8735788994997726, 0.872669395179627]
# em = [0.8799454297407913, 0.891768985902683, 0.8781264211005002, 0.8753979081400637]


ax.plot(cov_types, km, 'r', label="K-Means")
ax.plot(cov_types, em, 'b', label="EM")
ax.set(title="Model Accuracy vs. Number of Clusters", xlabel="Number of Clusters", ylabel="Accuracy")
ax.legend(loc='best', frameon=False)
fig.tight_layout()
fig.savefig("dig_spef_clusts")
plt.show()
print(km)
print(em)
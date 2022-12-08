import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import multivariate_normal as multinorm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def readdata(datapath, num_mfccs):
    with open(datapath) as f:
        lines = f.readlines()
    cleaned = []
    temp = []
    for line in lines:
        if line == '            \n':
            cleaned.append(temp)
            temp = []
        else:
            temp.append(line)
    doublecleaned = []
    for sample in cleaned:
        edited_sample = []
        for window in sample:
            i = 0
            window = window.split(" ")
            window[12] = window[12][0: len(window[12]) - 1]
            j = 0
            for mfcc in window:
                mfcc = float(mfcc)
                window[j] = mfcc
                j += 1
            i += 1
            edited_sample.append(window[0:num_mfccs])
        doublecleaned.append(edited_sample)
    doublecleaned = doublecleaned[1:len(doublecleaned)]
    return doublecleaned


def responsibility(pt, means, covs):
    c = len(means)
    res = np.zeros(c)
    for i in range(c):
        res[i] = multinorm.pdf(pt, means[i], covs[i])
    res = res/sum(res)
    return res


def variance(datapoint, center):
    var = 0
    for i in range(len(datapoint)):
        var = var + (datapoint[i] - center[i])**2
    return var


def kmeansdigit(digit, number_clusters, ctype, n_avg):
    thirteentrain = []
    for i in range(len(digit)):
        for j in range(0, int(n_avg/2)):
            thirteentrain.append(digit[i][j])
        for j in range(int(n_avg/2), len(digit[i]) - n_avg - int(n_avg/2)):
            tempavg = np.asarray(digit[i][j])
            for k in range(j - int(n_avg/2), j + int(n_avg/2)):
                tempavg = np.add(tempavg, np.asarray(digit[i][k]))
            thirteentrain.append(tempavg/n_avg)
        for j in range(len(digit[i]) - n_avg - int(n_avg/2), len(digit[i])):
            thirteentrain.append(digit[i][j])
    numclust = number_clusters
    kmeans = KMeans(init="k-means++", n_clusters=numclust, n_init=10, random_state=0)
    kmeans.fit(thirteentrain)
    centers = kmeans.cluster_centers_
    predictions = kmeans.predict(thirteentrain)
    pis = np.zeros(numclust)
    clustered = []
    for i in range(numclust):
        clustered.append([])
    variances = []
    if ctype == 'spherical':
        variances = np.zeros(numclust)
    for i in range(len(thirteentrain)):
        predicted = predictions[i]
        pis[predicted] = pis[predicted] + 1
        clustered[predicted].append(thirteentrain[i])
        if ctype == 'spherical':
            variances[predicted] = variances[predicted] + variance(thirteentrain[i], centers[predicted])
    pis = pis / len(thirteentrain)
    if ctype == 'spherical':
        variances = variances / len(thirteentrain)
    distributions = []
    for i in range(numclust):
        if ctype != 'spherical':
            c = np.cov((np.asarray(clustered[i])).T)
            if ctype == 'diag':
                c = np.diag(np.diag(c))
            variances.append(c)
        distributions.append(multinorm(centers[i], variances[i]))
    return [distributions, pis]


def emdigit(digit, number_clusters, ctype, n_avg):
    thirteentrain = []
    for i in range(len(digit)):
        for j in range(len(digit[i]) - n_avg):
            tempavg = np.asarray(digit[i][j])
            for k in range(j + 1, j + n_avg):
                tempavg = np.add(tempavg, np.asarray(digit[i][k]))
            thirteentrain.append(tempavg/n_avg)
    numclust = number_clusters
    gmm = GaussianMixture(n_components=numclust, covariance_type=ctype).fit(thirteentrain)
    means = gmm.means_
    covariances = gmm.covariances_
    # predictions = gmm.predict(thirteentrain)
    pis = np.zeros(numclust)
    responsibilities = np.zeros([len(thirteentrain), len(means)])
    for i in range(len(thirteentrain)):
        r = responsibility(thirteentrain[i], means, covariances)
        responsibilities[i] = r
        for j in range(len(responsibilities[i])):
            pis[j] = pis[j] + responsibilities[i][j]
    pis = pis / len(thirteentrain)
    distributions = []
    for i in range(numclust):
        distributions.append(multinorm(means[i], covariances[i]))
    return [distributions, pis]


def likelihood(data, distributions, pis):
    N = len(data)
    M = len(distributions)
    logproduct = 0
    for i in range(N):
        sum = 0
        for j in range(M):
            sum = sum + (pis[j]*distributions[j].pdf(data[i]))
        logsum = np.log(sum)
        logproduct = logproduct + logsum
    return -1*logproduct


def predictdigit(data, digdist):
    ML = float('inf')
    MLE = -1
    for i in range(len(digdist)):
        like = likelihood(data, digdist[i][0], digdist[i][1])
        if like < ML:
            ML = like
            MLE = i
    return MLE


def get_k_distribution(digdata, dd, n_clusts, ctype, n_avg):
    k = kmeansdigit(digdata, n_clusts, ctype, n_avg)
    dd.append([k[0], k[1]])
    return dd


def get_em_distribution(digdata, dd, n_clusts, ctype, n_avg):
    em = emdigit(digdata, n_clusts, ctype, n_avg)
    dd.append([em[0], em[1]])
    return dd


def printpredictions(data, dists):
    predictions = []
    for i in range(int(len(data) * len(dists) / 10)):
        predictions.append(predictdigit(data[i], dists))
    for i in range(len(dists)):
        print(predictions[(220 * i):(220 * (i + 1))])
        print(predictions[(220 * i):(220 * (i + 1))].count(i))


def accuracy(data, dists):
    predictions = []
    correct = 0
    for i in range(int(len(data) * len(dists) / 10)):
        predictions.append(predictdigit(data[i], dists))
    for i in range(len(dists)):
        correct += (predictions[(220 * i):(220 * (i + 1))].count(i))
    return correct/len(data)
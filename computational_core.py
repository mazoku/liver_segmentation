__author__ = 'tomas'

import matplotlib.pyplot as plt

import sklearn.mixture as skimix

import skimage.exposure as skiexp

import scipy.ndimage as scindi
import scipy.signal as scisig
import scipy.stats as scista

import numpy as np


def hist2gmm(data, smooth=True, sigma=1, order=5, debug=False):
    hist, bins = skiexp.histogram(data)
    hist[0] = hist[1:].max()

    if debug:
        hist_o = hist.copy()

    if smooth:
        hist = scindi.filters.gaussian_filter1d(hist, sigma=sigma)

    peaks_idx = scisig.argrelmax(hist, order=order)[0]
    peaks = bins[peaks_idx]
    peaks = np.hstack((0, peaks))

    n_components = len(peaks)
    gmm = create_gmm(data, peaks, debug=debug)

    if debug:
        plt.figure()
        plt.bar(bins, hist_o)
        plt.plot(bins, hist, 'g')

        for p in peaks:
            plt.plot(p, hist[p], 'ro')


def create_gmm(data, means, win_width=5, debug=False):
    n_comps = len(means)
    covs = np.zeros(n_comps)
    for i in range(n_comps):
        inners_m = (data >= (means[i] - win_width)) * (data <= (means[i] + win_width))
        inners = data[np.nonzero(inners_m)]
        covs[i] = np.cov(inners)

    gmm = skimix.GMM(n_comps)
    gmm.means_ = np.array(means.reshape(n_comps, 1))
    gmm.covars_ = np.array(covs.reshape(n_comps, 1))

    if debug:
        colors = 'rgbcmy'
        x = np.arange(0, 256, 1)
        plt.figure()
        for i in range(n_comps):
            plt.plot(x, scista.norm(means[i], covs[i]).pdf(x), colors[np.mod(i, len(colors))])

    return gmm
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
    comp = create_gmm(data, peaks, bins, hist)

    if debug:
        plt.figure()
        plt.bar(bins, hist_o)
        plt.plot(bins, hist, 'g')

        for p in peaks:
            plt.plot(p, hist[p], 'ro')
        plt.show()


def create_gmm(data, means, bins=None, hist=None, width=5):
    if (bins is None) or (hist is None):
        hist, bins = skiexp.histogram(data)

    n_comps = len(means)
    covs = np.zeros(n_comps)
    for i in range(n_comps):
        band_idx = ((bins-width) <= means[i]) * (means[i] <= (bins+width))

        # TODO: tady to musim predelat - takhle by to neslo
        covs[i] = np.cov(bins[band_idx])

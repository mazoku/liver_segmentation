__author__ = 'tomas'

import sys

import matplotlib.pyplot as plt

import sklearn.mixture as skimix

import skimage.exposure as skiexp

import scipy.ndimage as scindi
import scipy.signal as scisig
import scipy.stats as scista

import numpy as np

from PyQt4 import QtGui

import Viewer_3D


def get_hist_peaks(data, smooth=True, sigma=1, order=5, debug=False):
    hist, bins = skiexp.histogram(data)
    hist[0] = hist[1:].max()

    if debug:
        hist_o = hist.copy()

    if smooth:
        hist = scindi.filters.gaussian_filter1d(hist, sigma=sigma)

    peaks_idx = scisig.argrelmax(hist, order=order)[0]
    peaks = bins[peaks_idx]

    if debug:
        plt.figure()
        plt.bar(bins, hist_o)
        plt.plot(bins, hist, 'g')

        for p in peaks:
            plt.plot(p, hist[p], 'ro')

    return peaks


def estim_liver_prob_mod(data, win_width=5, debug=False):
    hist, bins = skiexp.histogram(data)
    ignore_band = 10
    peak = bins[hist[ignore_band:-ignore_band].argmax() + ignore_band]
    comp = estimate_comps(data, peak, win_width=win_width)[0]

    if debug:
        plt.figure()
        # plt.bar(bins, hist)
        plt.plot(bins, hist, 'g')
        plt.plot((peak, peak), (0, hist.max()), 'm-', linewidth=3)
        ax = plt.axis()
        plt.axis((ax[0], 255, ax[2], ax[3]))

        x = np.arange(0, 256, 1)
        y = comp.pdf(x)
        k = hist.max() / y.max()
        plt.plot(x, k * y, 'm-')

    return comp



def hist2comps(data, smooth=True, sigma=1, order=5, debug=False):
    peaks = get_hist_peaks(data, smooth, sigma, order, debug)
    peaks = np.hstack((0, peaks))

    n_components = len(peaks)
    comps = estimate_comps(data, peaks, debug=debug)

    return comps


def hist2gmm(data, smooth=True, sigma=1, order=5, debug=False):
    peaks = get_hist_peaks(data, smooth, sigma, order, debug)
    peaks = np.hstack((0, peaks))

    n_components = len(peaks)
    gmm = estimate_gmm(data, peaks, debug=debug)

    return gmm


def gmm_segmentation(data, gmm=None, debug=False):
    if gmm is None:
        gmm = hist2gmm(data, debug=debug)
    n_pts = np.prod(data.shape)
    respons = gmm.predict(data.reshape(n_pts, 1))

    if debug:
        # plt.figure()
        # plt.subplot(121), plt.imshow(data, 'gray')
        # plt.subplot(122), plt.imshow(respons.reshape(data.shape))
        app = QtGui.QApplication(sys.argv)
        viewer = Viewer_3D.Viewer_3D(data)
        viewer.show()

        viewer2 = Viewer_3D.Viewer_3D(respons.reshape(data.shape), range=True)
        viewer2.show()

        sys.exit(app.exec_())

    return respons


def estimate_comps(data, means, win_width=5, debug=False):
    means = np.atleast_1d(means)
    n_comps = len(means)
    covs = np.zeros(n_comps)
    comps = list()
    for i in range(n_comps):
        inners_m = (data >= (means[i] - win_width)) * (data <= (means[i] + win_width))
        inners = data[np.nonzero(inners_m)]
        covs[i] = np.cov(inners)
        comps.append(scista.norm(means[i], covs[i]))

    if debug:
        colors = 'rgbcmy'
        x = np.arange(0, 256, 1)
        plt.figure()
        for i in range(n_comps):
            plt.plot(x, comps[i].pdf(x), colors[np.mod(i, len(colors))])

    return comps


def estimate_gmm(data, means, win_width=5, debug=False):
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
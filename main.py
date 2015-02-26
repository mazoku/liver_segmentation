__author__ = 'tomas'

import matplotlib.pyplot as plt

import tools

import computational_core as cc

from PyQt4 import QtGui
import Viewer_3D
import sys


def run(fname, debug=False):
    ext_list = ('pklz', 'pickle')
    fname = fname
    if fname.split('.')[-1] in ext_list:
        data, mask, voxel_size = tools.load_pickle_data(fname)
    else:
        msg = 'Wrong data type, supported extensions: ', ', '.join(ext_list)
        raise IOError(msg)
    orig_shape = data.shape
    # data = tools.smoothing_tv(data, weight=0.05, sliceId=0)
    # data = tools.smoothing_gauss(data, sigma=1, sliceId=0)
    data = tools.smoothing_bilateral(data, sigma_space=5, sigma_color=0.02, sliceId=0)

    # cc.hist2gmm(data, debug=debug)
    # cc.gmm_segmentation(data, debug=debug)
    liver_rv = cc.estim_liver_prob_mod(data, debug=debug)

    liver_prob = liver_rv.pdf(data)
    app = QtGui.QApplication(sys.argv)
    viewer = Viewer_3D.Viewer_3D(data)
    viewer.show()
    viewer2 = Viewer_3D.Viewer_3D(liver_prob, range=True)
    viewer2.show()
    sys.exit(app.exec_())

    if debug:
        plt.show()


################################################################################
################################################################################
if __name__ == '__main__':

    # 2 hypo, 1 on the border --------------------
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_arterial_0.6_B30f-.pklz'
    # venous 0.6mm - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_venous_0.6_B20f-.pklz'
    # venous 5mm - ok, but wrong approach
    fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'

    # hypo in venous -----------------------
    # arterial - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_venous_0.6_B20f-.pklz'
    # venous - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz'

    # hyper, 1 on the border -------------------
    # arterial 0.6mm - not that bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Art_Abd_0.75_I26f_M_0.5-.pklz'
    # venous 5mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Ven_Abd_0.75_I26f_M_0.5-.pklz'

    # shluk -----------------
    # arterial 5mm
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_180_49509315_arterial_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_180_49509315_arterial_0.6_B20f-.pklz'

    # targeted

    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Arterial_0.75_I26f_3-.pklz'
    # venous 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Venous_0.75_I26f_3-.pklz'

    debug = True

    run(fname, debug=debug)
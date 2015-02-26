__author__ = 'tomas'

import sys

from PyQt4 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# from simple_viewer import Ui_MainWindow
from simple_viewer import Ui_Form

# class Viewer_3D(QtGui.QMainWindow):
class Viewer_3D(QtGui.QWidget):
    """Main class of the programm."""

    def __init__(self, data, range=False, parent=None):
        QtGui.QWidget.__init__(self, parent)
        # self.ui = Ui_MainWindow()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.actual_slice = 0
        self.data = data
        self.n_slices = self.data.shape[0]
        self.range = range

        # seting up the figure
        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0, 0, 1, 1])
        self.canvas = FigureCanvas(self.figure)

        # layout = QtGui.QVBoxLayout()
        # layout.addWidget(self.canvas, 1)
        # self.setLayout(layout)
        # conenction to wheel events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_figures()

        self.data = data
        # adding widget for displaying image data
        # self.form_widget = Form_widget.Form_widget(self)
        data_viewer_layout = QtGui.QHBoxLayout()
        # data_viewer_layout.addWidget(self.form_widget)
        data_viewer_layout.addWidget(self.canvas)
        self.ui.viewer_F.setLayout(data_viewer_layout)

        # seting up the range of the scrollbar to cope with the number of slices
        self.ui.slice_scrollB.setMaximum(self.n_slices - 1)

        # connecting slider
        self.ui.slice_scrollB.valueChanged.connect(self.slider_changed)


    def update_figures(self):
        if self.range:
            vmin = self.data.min()
            vmax = self.data.max()
        else:
            vmin = 0
            vmax = 255
        slice = self.data[self.actual_slice, :, :]

        plt.figure(self.figure.number)
        plt.subplot(111)
        # self.figure.gca().cla()  # clearing the contours, just to be sure
        plt.imshow(slice, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax)

        self.canvas.draw()


    def slider_changed(self, val):
        self.slice_change(val)
        self.actual_slice = val
        self.update_figures()


    def slice_change(self, val):
        self.ui.slice_scrollB.setValue(val)
        self.ui.slice_number_LBL.setText('slice # = %i' % (val + 1))


    def next_slice(self):
        self.actual_slice += 1
        if self.actual_slice >= self.n_slices:
            self.actual_slice = 0


    def prev_slice(self):
        self.actual_slice -= 1
        if self.actual_slice < 0:
            self.actual_slice = self.n_slices - 1


    def on_scroll(self, event):
        '''mouse wheel is used for setting slider value'''
        if event.button == 'up':
            self.next_slice()
        if event.button == 'down':
            self.prev_slice()
        self.update_figures()
        self.slice_change(self.actual_slice)


# def run(data):
#     app2 = QtGui.QApplication(sys.argv)
#     view = Viewer_3D(data)
#     view.show()
#     sys.exit(app2.exec_())


################################################################################
################################################################################
if __name__ == '__main__':

    # preparing data -----------------------------
    size = 100
    n_slices = 4
    data = np.zeros((n_slices, size, size))
    step = size / n_slices
    for i in range(n_slices):
        data[i, i * step:(i + 1) * step, :] = 150

    app = QtGui.QApplication(sys.argv)
    viewer = Viewer_3D(data)
    viewer.show()
    sys.exit(app.exec_())
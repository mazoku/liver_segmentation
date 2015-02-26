# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simple_viewer_widget.ui'
#
# Created: Wed Jan 14 16:04:38 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(499, 485)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.frame_2 = QtGui.QFrame(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.frame_2)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.viewer_F = QtGui.QFrame(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer_F.sizePolicy().hasHeightForWidth())
        self.viewer_F.setSizePolicy(sizePolicy)
        self.viewer_F.setMinimumSize(QtCore.QSize(0, 0))
        self.viewer_F.setFrameShape(QtGui.QFrame.Box)
        self.viewer_F.setObjectName(_fromUtf8("viewer_F"))
        self.verticalLayout_2.addWidget(self.viewer_F)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.slice_number_LBL = QtGui.QLabel(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice_number_LBL.sizePolicy().hasHeightForWidth())
        self.slice_number_LBL.setSizePolicy(sizePolicy)
        self.slice_number_LBL.setMinimumSize(QtCore.QSize(100, 0))
        self.slice_number_LBL.setObjectName(_fromUtf8("slice_number_LBL"))
        self.horizontalLayout_11.addWidget(self.slice_number_LBL)
        self.slice_scrollB = QtGui.QScrollBar(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice_scrollB.sizePolicy().hasHeightForWidth())
        self.slice_scrollB.setSizePolicy(sizePolicy)
        self.slice_scrollB.setOrientation(QtCore.Qt.Horizontal)
        self.slice_scrollB.setObjectName(_fromUtf8("slice_scrollB"))
        self.horizontalLayout_11.addWidget(self.slice_scrollB)
        self.verticalLayout_2.addLayout(self.horizontalLayout_11)
        self.verticalLayout.addWidget(self.frame_2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.slice_number_LBL.setText(_translate("Form", "slice # =", None))


#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-07-13

@author: Eugene Dvoretsky

Operates with image and statistics data on filesystem.
"""

import os
import datetime
from PySide import QtCore
from PySide import QtGui
from PIL import Image, TiffImagePlugin
TiffImagePlugin.WRITE_LIBTIFF = True


class Assay(object):
    """Photo and metadata container.
    """
    def __init__(self, cellfraction=None, dab_hemfraction=None,
            dab_dabhemfraction=None, photo=None):
        super(Assay, self).__init__()
        self.timestamp = datetime.datetime.now()  # Creation time
        self.cellfraction = cellfraction
        self.dab_hemfraction = dab_hemfraction
        self.dab_dabhemfraction = dab_dabhemfraction
        self.photo = photo  # Numpy image array
        '''
        self.id = 'UUID'  # Unique identifier
        self.name = str  # Arbitrary filename string

        self.objective = str  # Objective magnification string
        self.resolution = (int, int)
        self.exposure = int
        self.gain = float
        '''

    def __str__(self):
        s = "Time %s, CF %.2f, DAB/HEM %.2f, DAB / DAB || HEM %.2f" % ( 
            self.timestamp, self.cellfraction, self.dab_hemfraction,
            self.dab_dabhemfraction)
        return s

    def __unicode__(self):
        return self.__str__().decode('utf-8')


class StatDataModel(QtCore.QAbstractTableModel):
    """Handle images, metadata with filesystem.
    """
    def __init__(self):
        super(StatDataModel, self).__init__()
        self.__datadir = None
        self.__assays = list()
        self.__header = (
            'Timestamp', 'Cell fraction', 'DAB/HEM', 'DAB / DAB|HEM', 'Photo')
        self.isSaveImage = False

    def rowCount(self, index=QtCore.QModelIndex(), parent=QtCore.QModelIndex()):
        return len(self.__assays)

    def columnCount(self, index, parent=QtCore.QModelIndex()):
        return 5

    def data(self, index, role=QtCore.Qt.DisplayRole):
        """Returns the data stored under the given role for the item referred
        to by the index.
        """
        if not index.isValid() or index.row() >= self.rowCount():
            return None
        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0:
                return self.__assays[index.row()].timestamp.isoformat()
            elif index.column() == 1:
                return str(self.__assays[index.row()].cellfraction)
            elif index.column() == 2:
                return str(self.__assays[index.row()].dab_hemfraction)
            elif index.column() == 3:
                return str(self.__assays[index.row()].dab_dabhemfraction)
            elif index.column() == 4:
                if self.__assays[index.row()].photo is not None:
                    return 'Yes'
        return None

    def flags(self, index):
        """Returns the item flags for the given index.
        """
        if index.isValid():
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """Returns the data for the given role and section in the header with
        the specified orientation.
        """
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if len(self.__header) > section:
                    return self.__header[section]
            else:
                return '%.2d' % (section + 1)
        else:
            return None

#     def setData(self, index, value, role=QtCore.Qt.EditRole):
#         """Sets the role data for the item at index to value.
#         """
#         if index.isValid() and role == QtCore.Qt.EditRole:
#             print(value, type(value))
#             self.dataChanged.emit(index, index)
            # self.dataChanged.emit(
            #     self.createIndex(total_rows, 0),
            #     self.createIndex(total_rows, self.columnCount()))
#             return True  # If core accept data
#         else:
#             return False

    def insertRows(self, row, count, parent=QtCore.QModelIndex()):
        """Opposed to setData.
        """
        self.beginInsertRows(parent, row, row+count-1)
        print("Strange behavior.")
        self.endInsertRows()
        return True

    def appendAssay(self, assay):
        """Add statistics and image (and save it on disk) to end of the table.
        
        Replaces both insertRow and setData.
        """
        total_rows = self.rowCount()
        self.beginInsertRows(QtCore.QModelIndex(), total_rows, total_rows)
        self.__assays.append(assay)
#         self.__assays.append(Assay(cellfraction=0.7, dab_hemfraction=1.3, dab_dabhemfraction=0.9, photo='Maybe'))
        # im = misc.toimage(arr)
        # os.path.join()
        # im.save(filename, compression = "tiff_lzw")
        print("Appending assay %s" % assay)
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent=QtCore.QModelIndex()):
        """Remove selected assay statistics and image.
        """
        self.beginRemoveRows(parent, row, row+count-1)
        del(self.__assays[row:row+count])
        self.endRemoveRows()
        return True

#     def removeAssay(self, index):
#         self.beginRemoveRows(index, index.row(), index.row())
#         del(self.__assays[index.row()])
#         self.endRemoveRows()
#         return True

    def setDataDir(self, directory):
        """Set specified directory for image and metadata storing.
        """
        if os.path.exists(directory):
            self.__datadir = directory
            print("Datadir setted to '%s'" % directory)
        else:
            raise ValueError("Incorrect file path %s" % directory)
        
    def isDataDir(self):
        """Is data directory specified and exist?
        """
        if self.__datadir is not None:
            return os.path.exists(self.__datadir)
        else:
            return False

    def scanForAssays(self):
        """Scan filesystem directory for images and it's metadata.
        
        Don't forget `setDataDir` to appropriate directory before.
        """
        if not self.isDataDir():
            self.setDataDir(os.getcwdu())
        files = os.listdir(self.__datadir)
        print(files)
        for f in files:
            if f.endswith('.tif'):
                print('An TIFF image here! (%s)') % f


class FileWindow(QtGui.QWidget):
    """Allows discover and change Micromanager device properties.
    """
    def __init__(self, parent=None):
        super(FileWindow, self).__init__()
        self.parent = parent
        self.setMinimumSize(450, 600)
        
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.filemodel = QtGui.QFileSystemModel()
        self.view = QtGui.QTreeView()
        self.view.setSortingEnabled(True)
        self.view.setWordWrap(False)
        self.view.setModel(self.filemodel)
        self.view.setRootIndex(
            self.filemodel.setRootPath("."))
        self.vbox.addWidget(self.view)


class StatisticsBrowser(QtGui.QWidget):
    """Show statistics information.
    """
    wantAssay = QtCore.Signal()
    
    def __init__(self, model, parent=None):
        super(StatisticsBrowser, self).__init__()
        self.parent = parent
        self.model = model
#         self.setMinimumSize(600, 450)
        self.vbox = QtGui.QVBoxLayout()
        self.hButtonLayout = QtGui.QHBoxLayout()
        self.hButtonLayout.setAlignment(QtCore.Qt.AlignLeft)
        self.vbox.addLayout(self.hButtonLayout)
        self.setLayout(self.vbox)
        
        self.btnAdd = QtGui.QPushButton('&Add to account')
        self.hButtonLayout.addWidget(self.btnAdd)
        self.btnDel = QtGui.QPushButton('Delete')
        self.hButtonLayout.addWidget(self.btnDel)
        self.cbxSaveImage = QtGui.QCheckBox('Save specimen photo')
        self.hButtonLayout.addWidget(self.cbxSaveImage)
        self.btnSetDir = QtGui.QPushButton('Set path')
        self.hButtonLayout.addWidget(self.btnSetDir)
        
        self.view = QtGui.QTableView()
        self.view.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
#         self.view.horizontalHeader().setResizeMode(
#             QtGui.QHeaderView.ResizeToContents)
#         self.view.resizeColumnsToContents()
#         self.view.setSortingEnabled(True)
        self.view.setWordWrap(False)
        self.view.setModel(self.model)
        self.vbox.addWidget(self.view)
        
        self.btnAdd.clicked.connect(self.askAssay)
        self.btnDel.clicked.connect(self.deleteSelectedRows)
        self.btnSetDir.clicked.connect(self.setModelDataDir)
        self.cbxSaveImage.stateChanged.connect(self.toggleModelDataDir)

    @QtCore.Slot()
    def deleteSelectedRows(self):
        """Remove selected items rows.
        """
        # Get unique row number (user can select multiple cells in one row)
        uniqRows = set([idx.row() for idx in self.view.selectedIndexes()])
        # It's necessary to remove rows from the end, otherwise indexes become
        # outdated and useless.
        revRovs = sorted(list(uniqRows), reverse=True)
        for row in revRovs:
            self.model.removeRow(row)

    @QtCore.Slot()
    def setModelDataDir(self):
        """Set model data directory through GUI.
        """
        datadir = QtGui.QFileDialog.getExistingDirectory(parent=self,
             caption="Where to save image data and statistics?")
        if len(datadir) > 0:
            self.model.setDataDir(datadir)
        else:
            self.cbxSaveImage.setCheckState(QtCore.Qt.Unchecked)

    @QtCore.Slot()
    def toggleModelDataDir(self, state):
        """Force set model data directory if not specified yet and user wants
        to save statdata.
        """
        if self.model.isDataDir() is False:
            self.setModelDataDir()
        if self.model.isDataDir() and state == QtCore.Qt.Checked:
            self.model.isSaveImage = True
        else:
            self.model.isSaveImage = False
        print(self.model.isSaveImage)

    @QtCore.Slot()
    def askAssay(self):
        """Disable 'Add' button to prevent multiple clicking by mad user.
        """
        self.btnAdd.setEnabled(False)
        self.wantAssay.emit()

    @QtCore.Slot()    
    def ready(self):
        """Can ask for new assays.
        """
        self.btnAdd.setEnabled(True)


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    model = StatDataModel()
    model.appendAssay(Assay(cellfraction=0.7, dab_hemfraction=1.3, dab_dabhemfraction=0.9, photo='Maybe'))
    model.appendAssay(Assay(cellfraction=1.3, dab_hemfraction=2.9, dab_dabhemfraction=0.9, photo=None))
    model.appendAssay(Assay(cellfraction=2.2, dab_hemfraction=3.5, dab_dabhemfraction=0.1, photo=None))
    window = StatisticsBrowser(model)
    model.appendAssay(Assay(cellfraction=3.9, dab_hemfraction=4.9, dab_dabhemfraction=0.9, photo=None))
    window.show()
    model.appendAssay(Assay(cellfraction=4.9, dab_hemfraction=5.9, dab_dabhemfraction=1.9, photo=None))
    app.exec_()

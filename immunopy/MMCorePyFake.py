#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-05-28

@author: Eugene Dvoretsky
"""

import os
import Tkinter as tk
import ttk
import tkFileDialog
import threading
import numpy as np
from scipy import misc

try:
    import MMCorePy
    base = MMCorePy.CMMCore
    MM_INSTALLED = True
except ImportError:
    base = object
    MM_INSTALLED = False


class CMMCore(base):
    def __init__(self):
        super(CMMCore, self).__init__()
#         path = 'image/hdab256.tif'
#         path = 'image/2px_um.tif'
        path = 'image/Ki6720x_blue_filter.tif'
        curdir = os.path.dirname(os.path.abspath(__file__))
        self.set_image(os.path.join(curdir, path))
        MainWindow(mmcore=self)
        
    def getImageHeight(self):
        return self.frame.shape[0]
    def getImageWidth(self):
        return self.frame.shape[1]
    def setROI(self, x, y, w, h):
        print("setROI: %d %d %d %d") % (x, y, w, h)
        if self.RGB32.shape[0] < (y + h) or self.RGB32.shape[1] < (x + w):
            raise ValueError(
                "ROI %d, %d, %dx%d is bigger than image" % (x, y, w, h))
        self.frame = self.RGB32[y:y+h, x:x+w].copy()
    def clearROI(self):
        self.frame = self.RGB32.copy()
    def getLastImage(self):
        return self.frame.copy()
    def getImage(self):
        return self.frame.copy()
    def popNextImage(self):
        return self.frame.copy()
    def set_image(self, path):
        self.RGB = misc.imread(path)
        self.BGR = self.RGB[:,:,::-1]
        self.BGRA = np.dstack(
            (self.BGR, np.zeros((self.BGR.shape[0], self.BGR.shape[1]),
            dtype=np.uint8)))
        self.RGB32 = self.BGRA.view(dtype=np.uint32)
        self.frame = self.RGB32
    # If Micromanager isn't installed
    if not MM_INSTALLED:
        print("BAD NEWS")
        def loadSystemConfiguration(self, config_name):
            pass
        def getCameraDevice(self):
            return "Fake camera"
        def startContinuousSequenceAcquisition(self, bool_):
            pass
        def snapImage(self):
            pass
        def loadDevice(self, *device):
            self.input_video = ', '.join(device)
            print("Device '%s' loaded" % self.input_video)
        def initializeDevice(self, devname):
            print("Device '%s' initialized" % devname)
        def setCameraDevice(self, devname):
            print("Device camera '%s' initialized" % devname)
        def hasProperty(self, *props):
            pass
        def setProperty(self, *props):
            print("Props '%s' setted" % ', '.join([str(k) for k in props]))
        def setCircularBufferMemoryFootprint(self, value):
            pass
        def enableStderrLog(self, bool_):
            pass
        def enableDebugLog(self, bool_):
            pass
        def getBufferTotalCapacity(self):
            return 0.
        def getDevicePropertyNames(self, label):
            assert(label == "Fake camera")
            return ("Exposure", "Gain")
        def getImageBufferSize(self):
            return 0.
        def getRemainingImageCount(self):
            return 2
        def stopSequenceAcquisition(self):
            pass
        def reset(self):
            print("MMAdapterFake: Fake input_video `%s` reseted." % self.input_video)


class MainWindow(threading.Thread):
    def __init__(self, mmcore):
        super(MainWindow, self).__init__()
        self.mmc = mmcore
        self.image_list = None
        self.start()

    def choose_directory(self):
        """Returns a selected directoryname."""
        self.dir_opt = {
            'mustexist': True,
            'parent': self.root,
            'title': 'Images folder'}
        dname = tkFileDialog.askdirectory(**self.dir_opt)
        if dname:
            self.scan_dir(dname)

    def on_select(self, evt):
        w = evt.widget
        print('Image: %s' % [w.get(int(i)) for i in w.curselection()])
        if w.size() > 0:
#             return self.image_list[w.curselection()[0]]
            self.mmc.set_image(self.image_list[w.curselection()[0]])
        else:
            return None

    def scan_dir(self, directiry):
        self.image_list = list()
        self.list_bx.delete(0, tk.END)
        for f in sorted(os.listdir(directiry)):
            fpath = os.path.join(directiry, f)
            if os.path.isfile(fpath) and '.tif' in f:
                self.list_bx.insert(tk.END, f)
                self.image_list.append(fpath)

    def run(self):
        self.root = tk.Tk()
        self.root.wm_attributes('-topmost', 1)
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)
        
        self.root.title("Micromanager fake")
        self.label = ttk.Label(self.root, text="Which image should I return?")
        self.label.pack()

        self.list_bx = tk.Listbox(self.root, name='lb')
        self.list_bx.bind('<<ListboxSelect>>', self.on_select)
        self.list_bx.pack()

        self.open_btn = ttk.Button(self.root, text="Locate TIFF's", command=self.choose_directory)
        self.open_btn.pack(side='left')

        self.close_btn = ttk.Button(self.root, text="Close", command=self.root.destroy)
        self.close_btn.pack(side='right')
        self.root.mainloop()

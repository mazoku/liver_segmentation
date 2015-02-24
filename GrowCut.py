__author__ = 'Ryba'

import numpy as np
import Tkinter as tk
import ttk
import matplotlib.pyplot as plt
import warnings
import cv2


class GrowCut:
    def __init__( self, data, seeds, maxits=50 ,nghoodtype='sparse', gui=None ):
        '''
        data ... input data; should be 3D in form [slices, rows, columns]
        seeds ... seed points; same shape as data; background should have label 1
        '''
        self.data = data.astype(np.float64)
        self.data = np.array(self.data, ndmin=3)
        # self.data = self.smoothing( self.data )
        self.seeds = seeds
        self.maxits = maxits
        self.gui = gui
        self.visfig = None

        self.nslices, self.nrows, self.ncols = self.data.shape
        self.npixels = self.nrows * self.ncols
        self.nvoxels = self.npixels * self.nslices

        if self.nslices == 1: #2D image
            if nghoodtype == 'sparse':
                self.nghood = 4
            elif nghoodtype == 'full':
                self.nghood = 8
        else: #3D image
            if nghoodtype == 'sparse':
                self.nghood = 6
            elif nghoodtype == 'full':
                self.nghood = 26

        self.seeds = np.reshape( self.seeds, (1, self.nvoxels) ).squeeze()
        lind = np.ravel_multi_index( np.indices( self.data.shape ), self.data.shape ) #linear indices in array form
        self.lindv = np.reshape( lind, (1,self.nvoxels) ).squeeze() #linear indices in vector form
        self.coordsv = np.array( np.unravel_index( self.lindv, self.data.shape ) ) #coords in array [dim * nvoxels]

        self.strengths = np.zeros_like( self.seeds, dtype=np.float64 )
        self.strengths = np.where( self.seeds, 1., 0. )
        self.maxC = np.absolute( self.data.max() - self.data.min() )

        self.labels = self.seeds.copy()

        x = np.arange( 0, self.data.shape[2] )
        y = np.arange( 0, self.data.shape[1] )
        self.xgrid, self.ygrid = np.meshgrid( x, y )

        self.activePixs = np.argwhere( self.seeds > 0 ).squeeze()


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def run( self):
        self.neighborsM = self.make_neighborhood_matrix()

        converged = False
        it = 0
        if self.gui:
            self.gui.statusbar.config( text='Segmentation in progress...' )
            self.gui.progressbar.set( 0, '')
        while not converged:
            it += 1
            print 'iteration #%i' % it
            converged = self.iteration()

            if self.gui:
                self.redraw_fig()
                self.gui.progressbar.step( 1./self.maxits )
                self.gui.canvas.draw()

            if it == self.maxits and self.gui:
                self.gui.statusbar.config( text='Maximal number of iterations reached' )
                break

            if converged and self.gui:
                self.gui.statusbar.config( text='Algorithm converged after {0} iterations'.format(it) )

        print 'done'
        # qq = np.reshape( self.labels, self.data.shape )
        # plt.figure(), plt.imshow( qq[0,:,:] ), plt.show()


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def smoothing( self, data ):
        for i in range( data.shape[0] ):
            data[i,:,:] = cv2.GaussianBlur( data[i,:,:], (3,3), 0 )
        return data


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def redraw_fig(self):
        plt.hold( False )
        plt.imshow( self.gui.img[self.gui.currFrameIdx,:,:], aspect='equal' )
        plt.hold( True )
        # for i in self.linesL:
        #     self.ax.add_artist( i )

        if (self.labels == 1).any():
            self.blend_image( np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:] == 1, color='b' )
        if (self.labels == 2).any():
            self.blend_image( np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:] == 2, color='r' )
        if (self.labels == 3).any():
            self.blend_image( np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:] == 3, color='g' )

        self.gui.fig.canvas.draw()


    def get_labeled_im(self):
        return np.reshape(self.labels, self.data.shape)


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def blend_image( self, im, color, alpha=0.5 ):
        plt.hold( True )
        layer = np.zeros( np.hstack( (im.shape, 4)), dtype=np.uint8 )
        imP = np.argwhere( im )

        if color == 'r':
            layer[ imP[:,0], imP[:,1], 0 ] = 255
        elif color == 'g':
            layer[ imP[:,0], imP[:,1], 1 ] = 255
        elif color == 'b':
            layer[ imP[:,0], imP[:,1], 2 ] = 255
        elif color == 'c':
            layer[ imP[:,0], imP[:,1], (1,2) ] = 255
        elif color == 'm':
            layer[ imP[:,0], imP[:,1], (0,2) ] = 255
        elif color == 'y':
            layer[ imP[:,0], imP[:,1], (0,1) ] = 255

        layer[:,:,3] = 255 * im #alpha channel
        plt.imshow( layer, alpha=alpha  )


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def draw_contours(self):
        qq = np.reshape( self.labels, self.data.shape )[self.gui.currFrameIdx,:,:]
        plt.contour( self.xgrid, self.ygrid, qq==1, levels = [1], colors = 'b')
        plt.contour( self.xgrid, self.ygrid, qq==2, levels = [1], colors = 'r')

        plt.draw()


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def iteration( self):
        converged = True
        labelsN = self.labels.copy()

        # warnings.warn( 'For speed up - maintain list of active pixels and iterate only over them. '
        #                'Active pixel is pixel, that changed label or strength.' )

        newbies = list()
        for p in self.activePixs:
            pcoords = tuple(self.coordsv[:,p])
            for q in range(self.nghood):
                nghbInd = self.neighborsM[q,p]
                if np.isnan( nghbInd ):
                    continue
                nghbcoords = tuple(self.coordsv[:,nghbInd])
                # with warnings.catch_warnings(record=True) as w:
                C = np.absolute( self.data[pcoords] - self.data[nghbcoords] )

                g = 1 - ( C / self.maxC )

                #attack the neighbor
                #if g * self.strengths[nghbInd] > self.strengths[p]:
                if g * self.strengths[p] > self.strengths[nghbInd]:
                    self.strengths[nghbInd] = g * self.strengths[p]
                    labelsN[nghbInd] = self.labels[p]
                    newbies.append( nghbInd )
                    converged = False
        self.labels = labelsN
        self.activePixs = newbies
        return converged


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def make_neighborhood_matrix(self):
        # print 'start'
        if self.gui:
            self.gui.statusbar.config( text='Creating neighborhood matrix...' )
            self.gui.progressbar.set( 0, 'Creating neighborhood matrix...')
        if self.nghood == 8:
            nr = np.array( [-1, -1, -1, 0, 0, 1, 1, 1] )
            nc = np.array( [-1, 0, 1, -1, 1, -1, 0, 1] )
            ns = np.zeros( self.nghood )
        elif self.nghood == 4:
            nr = np.array( [-1, 0, 0, 1] )
            nc = np.array( [0, -1, 1, 0] )
            ns = np.zeros( self.nghood, dtype=np.int32 )
        elif self.nghood == 26:
            nrCenter = np.array( [-1, -1, -1, 0, 0, 1, 1, 1] )
            ncCenter = np.array( [-1, 0, 1, -1, 1, -1, 0, 1] )
            nrBorder = np.zeros( [-1, -1, -1, 0, 0, 0, 1, 1, 1] )
            ncBorder = np.array( [-1, 0, 1, -1, 0, 1, -1, 0, 1] )
            nr = np.array( np.hstack( (nrBorder, nrCenter, nrBorder) ) )
            nc = np.array( np.hstack( (ncBorder, ncCenter, ncBorder) ) )
            ns = np.array( np.hstack( (-np.ones_like(nrBorder), np.zeros_like(nrCenter), np.ones_like(nrBorder)) ) )
        elif self.nghood == 6:
            nrCenter = np.array( [-1, 0, 0, 1] )
            ncCenter = np.array( [0, -1, 1, 0] )
            nrBorder = np.array( [0] )
            ncBorder = np.array( [0] )
            nr = np.array( np.hstack( (nrBorder, nrCenter, nrBorder) ) )
            nc = np.array( np.hstack( (ncBorder, ncCenter, ncBorder) ) )
            ns = np.array( np.hstack( (-np.ones_like(nrBorder), np.zeros_like(nrCenter), np.ones_like(nrBorder)) ) )
        else:
            print 'Wrong neighborhood passed. Exiting.'
            return None

        neighborsM = np.zeros( (self.nghood, self.nvoxels) )
        for i in range( self.nvoxels ):
            s, r, c =  tuple( self.coordsv[:,i] )
            for nghb in range(self.nghood ):
                rn = r + nr[nghb]
                cn = c + nc[nghb]
                sn = s + ns[nghb]
                if rn < 0 or rn > (self.nrows-1) or cn < 0 or cn > (self.ncols-1) or sn < 0 or sn > (self.nslices-1):
                    neighborsM[nghb, i] = np.NaN
                else:
                    indexN = np.ravel_multi_index( (sn, rn, cn), self.data.shape )
                    neighborsM[nghb, i] = indexN
            if self.gui and np.floor(np.mod( i, self.nvoxels/20 )) == 0:
                self.gui.progressbar.step(1./20)
                self.gui.canvas.draw()
        if self.gui:
            self.gui.progressbar.set( 0, '')
            self.gui.statusbar.config( text='Neighborhood matrix created' )
        return neighborsM
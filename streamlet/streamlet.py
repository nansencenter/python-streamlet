# Name:         __init__.py
# Purpose:      Use the current folder as a package
# Authors:      Anton Korosov
# Created:      15.10.2014
# Copyright:    (c) NERSC 2014
# Licence:
# This file is part of Streamlet.
# Streamlet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# http://www.gnu.org/licenses/gpl-3.0.html
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
import numpy as np
import matplotlib.pyplot as plt

# Streamlet.status
OK = 0
NOGROW = 1
EMPTY = 2

class Streamlet(object):
    ''' Streamlet is essentially a short streamline segment'''
    pointsx = None
    pointsy = None
    moves = 0
    line = None
    status = OK
    doplot = True

    def __init__(self, u, v, x0=None, y0=None, factor=1,
                    lx=0, rx=1, ly=0, uy=1,
                    style='k-', doplot=True, **kwargs):
        ''' Create a Streamlet with only 2 points

        Parameters
        ----------
            u : numpy array
                field of eastward displacement
            v : numpy arra
                field of northward displacement
            x0 : float
                X-coordinate of the streamlet start (random, if not given)
            y0 : float
                Y-coordinate of the streamlet start (random, if not given)
            factor : float
                increase u/v vectors by <factor> to find the next point
            lx, rx : float, float
                X coordinates of the left and right borders
            ly, uy : float, float
                Y coordinates of the lower and upper borders
            style : str
                style to draw the line
            **kwargs : dict
                parameters for pyplot.plot()

        Modifies
        --------
            self.pointsx : list
                list with X-coordinates of points
            self.pointsy : list
                list with Y-coordinates of points
            self.plot : pyplot plot
                plot of the current streamlet

        '''
        # set coordinates of the borders
        self.lx = lx
        self.rx = rx
        self.ly = ly
        self.uy = uy
        self.width = u.shape[1]
        self.height =u.shape[0]
        self.factor = factor
        self.style = style
        self.doplot = doplot

        # get starting point
        if x0 is None:
            x0 = lx + np.random.random() * (rx - lx)
        if y0 is None:
            y0 = ly + np.random.random() * (uy - ly)

        # add starting point
        self.pointsx = [x0]
        self.pointsy = [y0]

        # add one more point
        self.grow(u, v, 1)

        # create plot
        self.plot(**kwargs)

    def __del__(self):
        ''' Clean up: remove line from plot'''
        self.line.remove()

    def get_next_point(self, u, v, x0, y0):
        ''' Find next point

        Parameters
        ----------
            u : numpy array
                field of eastward displacement
            v : numpy arra
                field of northward displacement
            x0 : float
                X-coordinate of the original point
            y0 : float
                Y-coordinate of the original point

        Returns
        -------
            x1, y1 : float, float
                next point

        '''
        ws = 1
        c0, r0 = self.cr_from_xy(x0, y0)
        if np.isnan(c0) or np.isnan(r0):
            return None, None

        # get matrix of U, V around the point
        try:
            subU = u[int(r0)-ws:int(r0)+ws+1, int(c0)-ws:int(c0)+ws+1]
            subV = v[int(r0)-ws:int(r0)+ws+1, int(c0)-ws:int(c0)+ws+1]
        except:
            import ipdb; ipdb.set_trace()

        if np.isnan(subU).any():
            return None, None

        if subU.shape != (3,3):
            return None, None

        # get matrix of weights based on vicinity of pixels to the point
        if c0 == int(c0) and r0 == int(r0):
            weights = np.array([[0,0,0],
                                [0,1,0],
                                [0,0,0]])
        else:
            xDist = np.array([[-1,0,1],
                              [-1,0,1],
                              [-1,0,1]]) - c0 + int(c0)

            yDist = np.array([[-1,-1,-1],
                              [ 0, 0, 0],
                              [ 1, 1, 1]]) - r0 + int(r0)
            weights = np.hypot(xDist, yDist)

        # get linear interpolation of U,V
        # from the sub array of U/V to the point
        # as a weighted mean of U and V
        uInterp = np.sum(subU * weights) / np.sum(weights)
        vInterp = np.sum(subV * weights) / np.sum(weights)

        # calculate next point
        x1 = x0 + uInterp * self.factor
        y1 = y0 + vInterp * self.factor

        return x1, y1

    def cr_from_xy(self, x, y):
        ''' Convert X, Y coordinates to column, row coordinates

        Parameters
        ----------
            x, Y : float
                X/Y - coordinates of the point
        Returns
        -------
            r, c : float
                Row/Column coordinates of the point

        '''
        c = self.width * (x - self.lx) / (self.rx - self.lx)
        r = self.height * (1 - (y - self.ly) / (self.uy - self.ly))
        return c, r

    def grow(self, u, v, steps):
        ''' Add points to the head of stream line

        Parameters
        ----------
            u, v : numpy arrays
                fields of eastward and northward displacement
            steps : int
                number of points to add in one go
        Modifies
        --------
            self.pointsx, self.pointsy : lists
                X, Y coordinates, <steps> points are appended

        '''
        # get the last point
        x0, y0 = self.pointsx[-1], self.pointsy[-1]
        for si in range(steps):
            x1, y1 = self.get_next_point(u, v, x0, y0)
            if (x1 is not None) and (y1 is not None):
                self.pointsx.append(x1)
                self.pointsy.append(y1)
                x0, y0 = x1, y1
            else:
                self.status = NOGROW

    def move(self, u, v, steps, maxMoves=None):
        ''' Move the streamlet forward

        Add points to the head of streamlet (grow)
        and remove points from the tail

        Parameters
        ----------
            u, v : numpy arrays
                fields of eastward and northward displacement
            steps : int
                number of points to add and remove in one go
            maxMoves : int
                maximum number of allowed moves. If streamlet made more moves
                that <maxMoves> it will not grow but only shrink
        Modifies
        --------
            self.pointsx, self.pointsy : lists
                X, Y coordinates, <steps> points are appended and removed
            self.moves : int
                number of moves so far

        '''
        if maxMoves is None or self.moves <= maxMoves:
            self.grow(u, v, steps)
        for i in range(steps):
            self.pointsx.pop(0)
            self.pointsy.pop(0)
        self.moves += 1
        if len(self) < 2:
            self.status = EMPTY

    def plot(self, **kwargs):
        ''' Plot the streamlet on given canvas

        Parameters
        ----------
            style : str
                style to plot the line (e.g. '-k' or '.-r')
            **kwargs : dict
                parameters for pyplot.plot()
        Modifies
        --------
            self.plot : pyplot.plot
                container for the plot

        '''
        if not self.doplot:
            return

        if self.status != EMPTY:
            self.line = plt.plot(self.pointsx, self.pointsy, self.style, **kwargs)[0]

    @classmethod
    def update_plot(cls, streamlet):
        ''' Update the plot

        Modifies
        --------
            self.plot : pyplot.plot
                container for the plot

        '''
        if not streamlet.doplot:
            return

        streamlet.line.set_xdata(streamlet.pointsx)
        streamlet.line.set_ydata(streamlet.pointsy)

    def __str__(self):
        ''' Print '''
        return '<Streamlet(%5.2f,%5.2f),(%5.2f,%5.2f)%d>' % (self.pointsx[0], self.pointsy[0],
                                      self.pointsx[-1], self.pointsy[-1],
                                      len(self))

    def __len__(self):
        ''' Length of streamlet is number of points '''
        return len(self.pointsx)

class StreamletSet(object):
    ''' Collection of Streamlet objects '''
    lx = 0
    rx = 1
    ly = 0
    uy = 1
    pool = None

    def __init__(self, X=None, Y=None, factor=1,
                                       style='-k',
                                       doplot=True,
                                       fig=None,
                                       **kwargs):
        ''' Create StreamletSet object and set parameters

        Parameter
        ---------
            X, Y : numpy arrays
                regular grids with X, Y coordinates of the pixels
            factor : float
                increase U/V strength by <factor>. Low factor gives smoother
                Streamlets, high factor gives fast growes/moves
            style : str
                parameter for pyplot.plot, e.g. '-k'
            doplot : bool
                Plot ?
            fig : pyplot.Figure
                canvas for plotting
            **kwargs : dict
                parameters for pyplot.plot()
            '''

        # set parameters for growing and plotting
        self.factor = factor
        self.style = style
        self.doplot = doplot
        self.fig = fig
        self.kwargs = kwargs

        # set coordinates of boundaries
        if X is not None and Y is not None:
            self.lx = X[-1, 0] # left X
            self.ly = Y[-1, 0] # lower Y
            self.rx = X[0, -1] # right X
            self.uy = Y[0, -1] # upper Y

        # create empty pool of streamlets
        if self.pool is None:
            self.pool = []

    def add_random(self, u, v, number=100):
        ''' Create Streamlets at random positions

        Parameters
        ----------
            u, v : numpy arrays
                fields of eastward and northward displacement
            number : int
                how many Streamlets to create
        Modifies
        --------
            self.pool : list
                add <number> Streamlet objects initiated in random positions

        '''
        n = 0
        while n < number:
            # create Streamlet
            sl = Streamlet(u, v,
                            lx=self.lx,
                            rx=self.rx,
                            ly=self.ly,
                            uy=self.uy,
                            factor=self.factor,
                            style=self.style,
                            doplot=self.doplot,
                            **self.kwargs)

            # add Streamlet to pool if number of points is 2
            if len(sl) > 1:
                self.pool.append(sl)
                n += 1

    def __str__(self):
        ''' Print '''
        return '<StreamletSet %s>' % str(len(self.pool))

    def __len__(self):
        ''' Get number of streamlets '''
        return len(self.pool)

    def __getitem__(self, key):
        ''' Get streamlet from the pool

        Parameters
        ----------
            key : int
                index of Streamlet in poll

        '''
        return self.pool[key]

    def grow(self, u, v, steps):
        ''' Grow all Streamlets

        Parameters
        ----------
            u, v : numpy arrays
                fields of eastward and northward displacement
            steps : int
                number of points to add in one go
        Modifies
        --------
            self.pool : list
                Streamlets in the self.pool are grown
        See also Streamlet.grow()
        '''
        for sl in self.pool:
            sl.grow(u, v, steps)

        # remove Streamlets which don't grow
        self.cleanup(badStatus=NOGROW)

    def move(self, u, v, steps, maxMoves=None):
        ''' Move all Streamlets

        Parameters
        ----------
            u, v : numpy arrays
                fields of eastward and northward displacement
            steps : int
                number of points to add and remove in one go
        Modifies
        --------
            self.pool : list
                Streamlets in the self.pool are moved
        See also Streamlet.move()

        '''
        for sl in self.pool:
            sl.move(u, v, steps, maxMoves)

        # remove empty Streamlets
        self.cleanup(badStatus=EMPTY)

    def plot(self):
        ''' Plot all streamlets

        Modifies
        --------
            self.pool : list
                all Streamlets in self.pool are plotted
        See also Streamlet.plot()

        '''
        map(Streamlet.plot, self.pool)

    def update_plot(self):
        ''' Update plot of all Streamlets

        Modifies
        --------
            self.pool : list
                all plots of Streamlets are updated
        See also Streamlet.update_plot()

        '''
        map(Streamlet.update_plot, self.pool)

    def cleanup(self, badStatus=EMPTY):
        ''' Remove empty streamlets

        Modifies
        --------
            self.pool : list
                empty Streamlets are removed

        '''
        # find good streamlets into goodLines
        goodLines = []
        for sl in self.pool:
            if sl.status < badStatus:
                goodLines.append(sl)

        # and replace self.pool with goodLines
        self.pool = list(goodLines)

    def grow_and_plot(self, u, v, frame, filename, steps=1, grows=1, **kwargs):
        ''' Generate several images with gradually growing Streamlets

        Parameters
        ----------
            u, v : numpy arrays
                fields of eastward and northward displacement
            frame : int
                number of the frame, is used in image naming
            filename : str
                template for file name, should contain only one %d for <frame>
            steps : int
                number of points to add and remove in one go
            grows : int
                number of calls to self.grow()
                how many frames to generate
            **kwargs : dict
                parameters for pyplot.savefig

        Returns
        -------
            frame : int
                increased frame
        Modifies
        --------
            self.pool : list
                all Streamlets are grown and and plots are updated
            <filename> % <frame> is generated

        '''
        # set canvas to draw
        canvas = plt
        if self.fig is not None:
            canvas = self.fig

        for gi in range(grows):
            print '%d GROW: %d' % (frame, len(self)),
            self.grow(u, v, steps)
            self.update_plot()
            canvas.savefig(filename % frame, **kwargs)
            frame += 1
            print len(self)

        return frame

    def move_and_plot(self, u, v, frame, filename, steps=1,
                                                   moves=1,
                                                   maxMoves=None,
                                                   **kwargs):
        ''' Generate several images with gradually moving Streamlets

        Parameters
        ----------
            u, v : numpy arrays
                fields of eastward and northward displacement
            frame : int
                number of the frame, is used in image naming
            filename : str
                template for file name, should contain only one %d for <frame>
            steps : int
                number of points to add and remove in one go
            moves : int
                number of calls to self.move()
                how many frames to generate
            maxMoves : int
                maximum number of allowed moves. If streamlet made more moves
                that <maxMoves> it will not grow but only shrink
            **kwargs : dict
                parameters for pyplot.savefig

        Returns
        -------
            frame : int
                increased frame
        Modifies
        --------
            self.pool : list
                all Streamlets are grown and and plots are updated
            <filename> % <frame> is generated

        '''
        # set canvas to draw
        canvas = plt
        if self.fig is not None:
            canvas = self.fig

        for mi in range(moves):
            print '%d MOVE: %d' % (frame, len(self)),
            self.move(u, v, steps, maxMoves)
            self.update_plot()
            plt.savefig(filename % frame, **kwargs)
            frame += 1
            print len(self)

        return frame

Python library for drawing nicely floating streamlets
=======
Install
=======
First, install required libraries. The easies way is to use a Python distribution. E.g. `Anaconda <https://store.continuum.io/cshop/anaconda/>`_

::
    
    # download the minimal anaconda distribution
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    
    # make the installer executable
    chmod +x miniconda.sh
    
    # run the installer and let it install miniconda
    # into your home directory
    ./miniconda.sh -b -p $HOME/miniconda
    
    # set the path to contain miniconda executables first
    # add this line in the end of your .bashrc
    # to make the miniconda work next time
    export PATH=$HOME/miniconda/bin:$PATH
    
    # update the anaconda distro
    # 'conda' is the command to manipulate with the distro
    conda update --yes conda
    
    # install the required packages into
    # $HOME/miniconda/lib/python2.7/site-packages
    # in addition that will install all the requirements
    conda install --yes ipython numpy scipy pillow basemap matplotlib

Second, download the code into folder with your python programs
::
    cd /home/username/pythonporgs/
    git clone https://github.com/nansencenter/python-streamlet.git

And finally add the python-streamlet dir to you $PYTHONPATH
::
    echo 'export PYTHONPATH=$PYTHONPATH:/home/username/pythonporgs/python-streamlet' >> ~/.bashrc

:note:
    The package will soon be added to PyPI and installation will be as easy as ```pip install streamlet```

=====
Usage
=====

In you code you should have several things available before using streamlet:
 * X, Y - numpy 2D arrays with regular fields of X and Y coordinates. That can be e.g. lon/lat fields in cylindrical projection (EPSG:4326);
 * U, V - numpy 2D array with fields of X and Y displacement. That can be e.g. eastward and northward components of ocean current or atmospheric winds;
 * matplotlib.pyplot axes initiated e.g. with pyplot.pcolormesh(array_with_background_color). That can be e.g. sea surface temperature

In the first simple exmaple we will seed 100 streamlets containing only two points, let them grow a little bit until they are 5 steps long and will move them 10 steps forward. Length of each particular step depends on the strength of underlying U/V components.

::
    # import API from python-streamlet
    from streamlet import StreamletSet, Streamlet

    # get X, Y from some source
    # X, Y = some_source()
    
    # get U, V from another source
    # U, V = anouther_source()
    
    # plot background color using data from some array
    # quad = plt.pcolormesh(some_array)
    # plt.xlim(X[-1, 0], X[0, -1]) # take coordinates from lower left ([-1, 0]) and ...
    # plt.ylim(Y[-1, 0], Y[0, -1]) # ... upper right ([0, -1]) corners
    
    # generate an empty set of streamlets. It defines:
    #   the grid for drawing streamlets
    #   style of lines
    #   speed of arrows growing
    sls = StreamletSet(X, Y, style='k-', linewidth=0.5, factor=1)
    
    # create 100 random streamlets
    sls.add_random(U, V, 100)

    # grow streamlets and generate 5 frames of the future animation in PNG files
    frame = 0 # frame counter
    grows = 5 # grow streamlet 5 times
    steps = 1 # each time, increment streamlet by only 1 step
    # write frame into the PNG file ('%05d' is replaced with frame number)
    filename = 'example01_%05d.png'
    frame = sls.grow_and_plot(U, V, frame, filename, steps, grows)

    # move streamlets forward and generate 10 frames
    moves = 10 # move streamlets 10 times
    frame = sls.move_and_plot(U, V, frame, filename, steps, moves)

Now convert your PNG frames into a animated GIF using ```ImageMagics```,

::
    convert -delay 5 example01_*.png example01.gif

Or into a movie using ```avconv```. Do it outside Python.

::
    avconv -y -r 24 -i example01_%05d.png -b 3000k -r 24 example01.avi


In the second exmaple we repeat seeding, growing and moving of streamlets 10 times to generate a lengthy animation. 

::
    # dont forget to provide the below
    # X, Y = some_source()
    # U, V = anouther_source()
    # quad = plt.pcolormesh(some_array)
    # plt.xlim(X[-1, 0], X[0, -1]) # take coordinates from lower left ([-1, 0]) and ...
    # plt.ylim(Y[-1, 0], Y[0, -1]) # ... upper right ([0, -1]) corners
    
    sls = StreamletSet(X, Y, style='k-', linewidth=0.5, factor=1)
    frame = 0
    filename = 'example02_%05d.png'
    for i in range(10):
        sls.add_random(U, V, 100)
        frame = sls.grow_and_plot(U, V, frame, filename, steps, grows)
        frame = sls.move_and_plot(U, V, frame, filename, steps, moves)

If you have sequence of background value arrays there is trick to make proper animation. Behind the scenes, Streamlet() does pyplot.plot() only once, when you create a new streamlet. Later, when you grow, or move it updates the xdata and ydata of the plotted line and saves animation without recreating the canvas. Therefore you should not do pcolormesh() everytime you want to update backgound, but rather update the pregenerated pcolormesh with new values as in the example below.

::
    # dont forget to provide X,Y,U,V
    # X, Y = some_source()
    # U, V = anouther_source()

    # NB! Here we set the quad to be None!
    # quad = None

    sls = StreamletSet(X, Y, style='k-', linewidth=0.5, factor=1)
    frame = 0
    filename = 'example03_%05d.png'
    for i in range(10):

        # get new array to show in the backgound
        some_array = get_new_backgound_array(i)

        if quad is None:
            # we call  pcolormesh only the first time in the loop
            quad = plt.pcolormesh(some_array)
            # plt.xlim(X[-1, 0], X[0, -1]) # take coordinates from lower left ([-1, 0]) and ...
            # plt.ylim(Y[-1, 0], Y[0, -1]) # ... upper right ([0, -1]) corners
        else:
            # other times we only update it with values from the array
            quad.set_array(some_array[1:, 1:].ravel())
        
        sls.add_random(U, V, 100)
        frame = sls.grow_and_plot(U, V, frame, filename, steps, grows)
        frame = sls.move_and_plot(U, V, frame, filename, steps, moves)

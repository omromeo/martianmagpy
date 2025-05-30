"""
NAME:
    martianmagpy
DATE:
    09/07/2023
AUTHOR:
    Orlando Romeo
PURPOSE:
    This module includes functions to recreate the simulation results in Lillis et al 2010 
    on the coherence scales of the Martian Magnetic Field for Insight
FUNCTIONS:
    mag_cmap  - Custom Color Map
    mag_grid  - Initialize 3D Magnteization Grid
    powlaw    - Compute Power Law
    gaussian  - Compute gaussian function
    coherence - Change coherence scale of magnetization grid
    demag     - Demagnetize impact crater in grid
    mag_alt   - Find magnetic field at given altitude using Blakely model
    mag_ravg  - Find radial average of B from center of grid for given altitude
    alt_curve - Plot altitude profile from averaged magnetic field for given location
"""
# %%=============== PYTHON MODULES ============================================
from scipy import stats
from scipy.fft import ifftn, fftn
import numpy as np                                                             
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as clr
from scipy import ndimage
import os
plt.rcParams["font.family"] = "Times New Roman" 
plt.rcParams["font.size"] = 30
# %% Functions
def mag_cmap(color='',reverse=0):
    if color == 'blue':
        basic_cols=['#000000','#178EDE','#93F1F3']  #,'#252941','#3A1B1F'
    elif color == 'orange':
        basic_cols=['#000000','#AE5120','#F4DE44']  #,'#252941','#3A1B1F'
    elif color == 'purple':
        basic_cols=['#000000','#7A16DE','#D383F2']  #,'#252941','#3A1B1F'
    elif color == 'white':
        basic_cols=[ '#0EAFC1','#11AAE7','#34E3FF','#C9F0FF','#FFFFFF','#FFE4C9','#fda600','#e77d11','#c1440e' ]
    else:
        basic_cols=['#93F1F3','#178EDE','#000000','#AE5120','#F4DE44']  #,'#252941','#3A1B1F'
        basic_cols=['#C9F0FF','#34E3FF','#11AAE7','#0EAFC1','#044046','#000000',
             '#451804','#c1440e','#e77d11','#fda600','#FFE4C9']
    if reverse != 0:
        basic_cols.reverse()
    mag_cmap=clr.LinearSegmentedColormap.from_list('mag_cmap', basic_cols)
    return mag_cmap
# ================= INITIALIZE MAG GRID =======================================
def mag_grid(Nx,Ny,Nz,dx,dy,dz,M_RMS=10,randseed=10,hist_flg=0,p2d_flg=0,Zplt=20,p3d_flg=0,km=1):
    """
    Function to Initialize 3D Magnteization Grid:

    Parameters
    ----------
    Nx : Int
        Number of points in x direction of grid.
    Ny : Int
        Number of points in y direction of grid..
    Nz : Int
        Number of points in z direction of grid.
    dx : Scalar (meters)
        Spacing in x direction of grid.
    dy : Scalar (meters)
        Spacing in y direction of grid.
    dz : Scalar (meters)
        Spacing in z direction of grid.
    M_RMS : Scalar, optional  (meters)
        Distribution Width for Randomization. The default is 10.
    randseed : Scalar, optional
        Random Seed. The default is 10.
    hist_flg : Int, optional
        If set to 1, will plot histogram of random mag values. The default is 0.
    p2d_flg : Int, optional
        If set to 1, will plot mag values in xy plane. The default is 0.
    Zplt : Int, optional
        Z value for xy plot. The default is 20.
    p3d_flg : Int, optional
        If set to 1, will plot 3d grid. The default is 0.
    km : Int, optional
        If set to 1 will use km for plots instead of meters

    Returns
    -------
    mag3d : array_like
        Magnetized 3d Grid
    """
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    # MAG Values Gaussian Distribution Parameters
    W  = M_RMS    # Gaussian Distribution Width (A/m)
    C  = 0        # Gaussian Distribution Center (A/m) 
    # Set Random Distribution
    np.random.seed(randseed)
    mag3d = np.random.normal(loc=C,scale=W,size=(Nx,Ny,Nz))
    # Histogram of Random Magnetization Vectors
    if hist_flg == 1:
        fig = plt.figure(figsize=(16, 12))
        ax  = fig.add_subplot(111)
        plt.hist(x=mag3d.flatten(), bins='auto', color='r',alpha=0.7)
        plt.xlabel('Magnetization Values (A/m)')
        plt.ylabel('Counts')
        plt.title('Histogram of Magnetization Values on 3D Grid')
        ax.set_axisbelow(True)
        ax.grid()
        plt.show()
    # Magnetization Grid in XY Plane    
    if p2d_flg == 1:
        # Bins for Spatial Plotting
        Xbin = np.linspace(-1*(Nx/2.)*dx,(Nx/2.)*dx,Nx+1)/unit_conv
        Ybin = np.linspace(-1*(Ny/2.)*dy,(Ny/2.)*dy,Ny+1)/unit_conv
        Z = np.linspace(0,1-Nz,Nz)*dz/unit_conv
        vm = np.max(np.abs(mag3d[:,:,Zplt]))
        fig=plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        m=ax.pcolormesh(Xbin,Ybin,np.transpose(mag3d[:,:,Zplt]),cmap=mag_cmap(),vmin=-vm,vmax=vm)
        ax.set_title('Magnetization Grid with Z = '+str(Z[Zplt])+' km')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        cbar = fig.colorbar(m)
        cbar.set_label('Magnetization (A/m)')
        plt.show()
    # Magnetization Grid in 3D Box    
    if p3d_flg == 1:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.gca(projection='3d')
        # Determine range of maximum axis
        maxax = np.max(np.array([Nx*dx,Ny*dy,Nz*dz]))
        ax.set_xlim([0,maxax])
        ax.set_ylim([0,maxax])
        ax.set_zlim([0,maxax/5]) # Iffset z axis to see zoomed version
        # Create X and Y Grids
        X = np.arange(0, Nx*dx, dx)
        Y = np.arange(0, Ny*dy, dy)
        X, Y = np.meshgrid(X, Y)
        # Create Plot of Top surface
        cmap = mag_cmap()
        maxm = np.nanmax(np.abs(mag3d))*.5
        norm = clr.Normalize(vmin=-maxm, vmax=maxm)
        nz = mag3d[:,:,0]
        ax.plot_surface(X, Y, X*0+(Nz*dz)-dz,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Plot Front Surface
        X =  np.arange(0, Nx*dx, dx)
        Z =  np.arange(0, Nz*dz, dz)
        X, Z = np.meshgrid(X, Z, indexing='xy')
        nz = mag3d[0,:,::-1].T
        ax.plot_surface(X, X*0, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable2.set_array([])
        # Plot Right Surface
        Y =  np.arange(0, Ny*dy, dy)
        Z =  np.arange(0, Nz*dz, dz)
        Y, Z = np.meshgrid(Y, Z, indexing='xy')
        nz = mag3d[:,-1,::-1].T
        ax.plot_surface(Y*0+(Nx*dx)-dx, Y, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Turn axis off, with no colorbar
        plt.axis('off')
        fig.colorbar(mappable, shrink=1, aspect=10,orientation='vertical',pad=0,label='Magnetization (A/m)')
        fig.tight_layout()
        plt.show()
    return mag3d
# ================= INITIALIZE MAG GRID 2 =====================================
def insight_geogrid(region,Nz,dz,M=10,p2d_flg=0,Zplt=20,p3d_flg=0,km=1,gridheight=3000):
    """
    Function to Initialize 3D Magnteization Grid near InSight based on Geology:

    Parameters
    ----------
    regions: List
        Dictionary of regions to magnetize, where key is region type and value
        is magnetization ratio between given region and eHt (where InSight is)
    Nz : Int
        Number of points in z direction of grid.
    dz : Scalar (meters)
        Spacing in z direction of grid.
    M : Scalar, optional
        Magnetization strength near Insight in A/m
    randseed : Scalar, optional
        Random Seed. The default is 10.
    hist_flg : Int, optional
        If set to 1, will plot histogram of random mag values. The default is 0.
    p2d_flg : Int, optional
        If set to 1, will plot mag values in xy plane. The default is 0.
    Zplt : Int, optional
        Z value for xy plot. The default is 20.
    p3d_flg : Int, optional
        If set to 1, will plot 3d grid. The default is 0.
    km : Int, optional
        If set to 1 will use km for plots instead of meters
    gridheight:
        Height of where grid surface beings (+ value means above 0 km altitude)
        
    Returns
    -------
    mag3d : array_like
        Magnetized 3d Grid
    """
    # Load the grid values from the predefined .npy file
    if os.name != 'nt':
        hdir = "/home/oromeo/Documents/Research/Insight_Mars/"  # Set save file directory
    else:
        hdir = "C:/Users/Orlando/Documents/2-Research/4-UCB/EPS/Insight-MarsProject/Workspace/"  # Set save file directory
    data = np.load(hdir+"Data/InSight_Grid_Regions.npz")
    rgrid = data['grid']
    region_list = dict(data['region'])
    
    ddata = np.load(hdir+"Data/InSight_Grid_Depths.npz")
    dgrid = ddata['dgrid']*1e3
    Nx = 512
    dx = 2e3
    Ny=Nx
    dy=dx
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    # Initial mag grid
    maggrid = np.zeros((Nx,Nx,Nz))
    # Iterate over each element in the dictionary
    for key, value in region.items():
        if value != 0.0:
            ind = (np.where((rgrid == int(region_list[key]))))
            #Check for impacts
            if key == 'AHi':
                # Extract the row and column indices
                rows = ind[0]
                cols = ind[1]
                # Create a mask for the condition
                mask = (rows <= 350) | (cols <= 150)
                # Apply the mask to filter the rows and columns
                filtered_rows = rows[mask]
                filtered_cols = cols[mask]
                # Create the new filtered ind tuple
                ind = (filtered_rows, filtered_cols)
            if np.any(ind):
                maggrid[ind[0],ind[1],:] = M*float(region[key])
    # Average over region borders using image filter
    mag3d = np.copy(maggrid)
    for i in range(100):
        mag3d = ndimage.uniform_filter(mag3d, size=5, mode='nearest')
    # Based on depth, remove magnetization
    z = -1.0*np.arange(0+dz/2, dz*Nz+dz/2, dz) + gridheight#np.ceil(np.max(dgrid))
    Z = z[np.newaxis, np.newaxis, :] + np.zeros((Nx, Nx, 1))
    # Use broadcasting and logical indexing to set Z to sNaN where Z > d
    Z[Z > dgrid[:, :, np.newaxis]] = np.nan
    mag3d[np.isnan(Z)] = 0
    #region_list = {'AHv':1, 'AHi':2, 'lAv':3, 'AHtu':4, 'Htu':5, 'mNh':6, 'HNt':7,'eHt':8,'lHt':9, 'lNh':10}
    #region = {'AHv':0.0, 'AHi':0.0, 'lAv':0.0, 'AHtu':0.0, 'Htu':0.0, 'mNh':2.0, 'HNt':2.0,'eHt':1.0,'lHt':0.0}
    #region = {'mNh':2.0, 'HNt':-2.0, 'eHt':1.0, 'lHt':0.0}
    # Magnetization Grid in XY Plane    
    if p2d_flg == 1:
        # Bins for Spatial Plotting
        Xbin = np.linspace(-1*(Nx/2.)*dx,(Nx/2.)*dx,Nx+1)/unit_conv
        Ybin = np.linspace(-1*(Ny/2.)*dy,(Ny/2.)*dy,Ny+1)/unit_conv
        Z = np.linspace(0,1-Nz,Nz)*dz/unit_conv
        vm = np.max(np.abs(mag3d[:,:,Zplt]))
        fig=plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        m=ax.pcolormesh(Xbin,Ybin,np.transpose(mag3d[:,:,Zplt]),cmap='seismic',vmin=-vm,vmax=vm)
        ax.set_title('Magnetization Grid with Z = '+str(Z[Zplt])+' km')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        cbar = fig.colorbar(m)
        cbar.set_label('Magnetization (A/m)')
        plt.show()
    # Magnetization Grid in 3D Box    
    if p3d_flg == 1:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.gca(projection='3d')
        # Determine range of maximum axis
        maxax = np.max(np.array([Nx*dx,Ny*dy,Nz*dz]))
        ax.set_xlim([0,maxax])
        ax.set_ylim([0,maxax])
        ax.set_zlim([0,maxax/20]) # Offset z axis to see zoomed version
        # Create X and Y Grids
        X = np.arange(0, Nx*dx, dx)
        Y = np.arange(0, Ny*dy, dy)
        X, Y = np.meshgrid(X, Y)
        # Create Plot of Top surface
        cmap = mag_cmap()
        maxm = np.nanmax(np.abs(mag3d))*.5
        norm = clr.Normalize(vmin=-maxm, vmax=maxm)
        nz = mag3d[:,:,9]
        ax.plot_surface(X, Y, X*0+(Nz*dz)-dz,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Plot Front Surface
        X =  np.arange(0, Nx*dx, dx)
        Z =  np.arange(0, Nz*dz, dz)
        X, Z = np.meshgrid(X, Z, indexing='xy')
        nz = mag3d[0,:,::-1].T
        ax.plot_surface(X, X*0, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable2.set_array([])
        # Plot Right Surface
        Y =  np.arange(0, Ny*dy, dy)
        Z =  np.arange(0, Nz*dz, dz)
        Y, Z = np.meshgrid(Y, Z, indexing='xy')
        nz = mag3d[:,-1,::-1].T
        ax.plot_surface(Y*0+(Nx*dx)-dx, Y, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Turn axis off, with no colorbar
        plt.axis('off')
        fig.colorbar(mappable, shrink=1, aspect=10,orientation='vertical',pad=0,label='Magnetization (A/m)')
        fig.tight_layout()
        plt.show()
    return mag3d
# ================= INITIALIZE MAG GRID 2 =====================================
def zhurong_geogrid(region,Nz,dz,M=10,p2d_flg=0,Zplt=20,p3d_flg=0,km=1,gridheight=1000):
    """
    Function to Initialize 3D Magnteization Grid near InSight based on Geology:

    Parameters
    ----------
    regions: List
        Dictionary of regions to magnetize, where key is region type and value
        is magnetization ratio between given region and eHt (where InSight is)
    Nz : Int
        Number of points in z direction of grid.
    dz : Scalar (meters)
        Spacing in z direction of grid.
    M : Scalar, optional
        Magnetization strength near Insight in A/m
    randseed : Scalar, optional
        Random Seed. The default is 10.
    hist_flg : Int, optional
        If set to 1, will plot histogram of random mag values. The default is 0.
    p2d_flg : Int, optional
        If set to 1, will plot mag values in xy plane. The default is 0.
    Zplt : Int, optional
        Z value for xy plot. The default is 20.
    p3d_flg : Int, optional
        If set to 1, will plot 3d grid. The default is 0.
    km : Int, optional
        If set to 1 will use km for plots instead of meters
    gridheight:
        Height of where grid surface beings (+ value means above 0 km altitude)
        
    Returns
    -------
    mag3d : array_like
        Magnetized 3d Grid
    """
    # Load the grid values from the predefined .npy file
    if os.name != 'nt':
        hdir = "/home/oromeo/Documents/Research/Insight_Mars/"  # Set save file directory
    else:
        hdir = "C:/Users/Orlando/Documents/2-Research/4-UCB/EPS/Insight-MarsProject/Workspace/"  # Set save file directory
    data = np.load(hdir+"Data/Zhurong_Grid_Regions.npz")
    rgrid = data['grid']
    region_list = dict(data['region'])
    
    ddata = np.load(hdir+"Data/Zhurong_Grid_Depths.npz")
    dgrid = ddata['dgrid']*1e3
    Nx = 512
    dx = 2e3
    Ny=Nx
    dy=dx
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    # Initial mag grid
    maggrid = np.zeros((Nx,Nx,Nz))
    # Iterate over each element in the dictionary
    for key, value in region.items():
        if value != 0.0:
            ind = (np.where((rgrid == int(region_list[key]))))
            if np.any(ind):
                maggrid[ind[0],ind[1],:] = M*float(region[key])
    # Average over region borders using image filter
    mag3d = np.copy(maggrid)
    for i in range(100):
        mag3d = ndimage.uniform_filter(mag3d, size=5, mode='nearest')
    # Based on depth, remove magnetization
    z = -1.0*np.arange(0+dz/2, dz*Nz+dz/2, dz) + gridheight#np.ceil(np.max(dgrid))
    Z = z[np.newaxis, np.newaxis, :] + np.zeros((Nx, Nx, 1))
    # Use broadcasting and logical indexing to set Z to sNaN where Z > d
    Z[Z > dgrid[:, :, np.newaxis]] = np.nan
    mag3d[np.isnan(Z)] = 0
    # Magnetization Grid in XY Plane    
    if p2d_flg == 1:
        # Bins for Spatial Plotting
        Xbin = np.linspace(-1*(Nx/2.)*dx,(Nx/2.)*dx,Nx+1)/unit_conv
        Ybin = np.linspace(-1*(Ny/2.)*dy,(Ny/2.)*dy,Ny+1)/unit_conv
        Z = np.linspace(0,1-Nz,Nz)*dz/unit_conv
        vm = np.max(np.abs(mag3d[:,:,Zplt]))
        fig=plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        m=ax.pcolormesh(Xbin,Ybin,np.transpose(mag3d[:,:,Zplt]),cmap='seismic',vmin=-vm,vmax=vm)
        ax.set_title('Magnetization Grid with Z = '+str(Z[Zplt])+' km')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        cbar = fig.colorbar(m)
        cbar.set_label('Magnetization (A/m)')
        plt.show()
    # Magnetization Grid in 3D Box    
    if p3d_flg == 1:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        #ax = fig.gca(projection='3d')
        # Determine range of maximum axis
        maxax = np.max(np.array([Nx*dx,Ny*dy,Nz*dz]))
        ax.set_xlim([0,maxax])
        ax.set_ylim([0,maxax])
        ax.set_zlim([0,maxax/20]) # Offset z axis to see zoomed version
        # Create X and Y Grids
        X = np.arange(0, Nx*dx, dx)
        Y = np.arange(0, Ny*dy, dy)
        X, Y = np.meshgrid(X, Y)
        # Create Plot of Top surface
        cmap = mag_cmap()
        maxm = np.nanmax(np.abs(mag3d))*.5
        norm = clr.Normalize(vmin=-maxm, vmax=maxm)
        nz = mag3d[:,:,9]
        ax.plot_surface(X, Y, X*0+(Nz*dz)-dz,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Plot Front Surface
        X =  np.arange(0, Nx*dx, dx)
        Z =  np.arange(0, Nz*dz, dz)
        X, Z = np.meshgrid(X, Z, indexing='xy')
        nz = mag3d[0,:,::-1].T
        ax.plot_surface(X, X*0, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable2.set_array([])
        # Plot Right Surface
        Y =  np.arange(0, Ny*dy, dy)
        Z =  np.arange(0, Nz*dz, dz)
        Y, Z = np.meshgrid(Y, Z, indexing='xy')
        nz = mag3d[:,-1,::-1].T
        ax.plot_surface(Y*0+(Nx*dx)-dx, Y, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Turn axis off, with no colorbar
        plt.axis('off')
        fig.colorbar(mappable, shrink=1, aspect=10,orientation='vertical',pad=0,label='Magnetization (A/m)')
        fig.tight_layout()
        plt.show()
    return mag3d
# ================= APPLY COHERENCE SCALE TO MAG GRID =========================
def powlaw(x,pd,norm):
    """
    Function to compute power law

    Parameters
    ----------
    x : array_like
        input array.
    pd : scalar
        power degree.
    norm : int
        If set to 1, will return normalized function.

    Returns
    -------
    f : array_like
        Power Law Function.
    """
    # Account for zero
    xzero = np.where(x == 0)
    x[xzero] = 1
    f = x**(-1*pd)
    f[xzero] = 0.0
    if norm == 1:
        f = f/np.nanmax(f)
    return f
def gaussian(x,c,w,norm):
    """
    Function to compute gaussian function

    Parameters
    ----------
    x : array_like
        input array.
    c : scalar (can be more than 1)
        Gaussian Center.
    w : scalar (can be more than 1)
        Gaussian Width.
    norm : int
        If set to 1, will return normalized function.

    Returns
    -------
    f : array_like
        Gaussian Function.
    """
    # Initial Loop size
    f = x*0.0
    # Add gaussian functions
    if isinstance(c, (list, tuple, np.ndarray)):
        for i in range(len(c)):
            # NEW METHOD
            fi = np.exp(-0.5*((x-c[i])/w[i])**2) / (w[i]*np.sqrt(2*np.pi))
            #f += fi
            # OLD METHOD
            fi = np.exp(-0.5*((x-c[i])/w[i])**2) / (w[i]*np.sqrt(2*np.pi))
            ## Normalize
            ##f += (fi-np.nanmin(fi))/(np.nanmax(fi)-np.nanmin(fi))
            ## Normalize and take max value
            fi = (fi-np.nanmin(fi))/(np.nanmax(fi)-np.nanmin(fi))
            f = np.max([f,fi],axis=0)
    else:
        f += np.exp(-0.5*((x-c)/w)**2) / (w*np.sqrt(2*np.pi))
    if norm == 1:
        f = f/np.nanmax(f)
    return f
def coherence(mag3d,Nx,Ny,Nz,dx,dy,dz,par,fltr_flg=0,p2d_flg=0,Zplt=20,p3d_flg=0,km=1):
    """
    Function to change coherence scale of magnetization grid.

    Parameters
    ----------
    mag3d : array_like
        3D Matrix of Magnetization grid.
    Nx : Int
        Number of points in x direction of grid.
    Ny : Int
        Number of points in y direction of grid..
    Nz : Int
        Number of points in z direction of grid.
    dx : Scalar
        Spacing in x direction of grid.
    dy : Scalar
        Spacing in y direction of grid.
    dz : Scalar
        Spacing in z direction of grid.
    par : dictionary of:
        Filter Name = 'powlaw'
                      'gaussian'
        pl : Scalar
            Power Degree in lateral direction.
        pv : Scalar
            Power Degree in vertical direction.
        lcc : Scalar
            Gaussian Center in lateral direction.
        vcc : Scalar
            Gaussian Center in vertical direction.
        lcw : Scalar
            Gaussian Width in lateral direction.
        vcw : Scalar
            Gaussian Width in vertical direction.
    fltr_flg : Int, optional
        If set to 1, will plot filter functions. The default is 0.
    p2d_flg : Int, optional
        If set to 1, will plot mag values in xy plane. The default is 0.
    Zplt : Int, optional
        Z value for xy plot. The default is 20.
    p3d_flg : Int, optional
        If set to 1, will plot 3d grid. The default is 0.
    km : Int, optional
        If set to 1 will use km for plots instead of meters    

    Returns
    -------
    fmag3d : array_like
        Magnetization Grid with coherence in horizontal and vertical directions.
    """
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    twopi = 2.0 * np.pi
    # ============= MAG GRID FFT ==============================================
    cmag3d = np.fft.fftn(mag3d,s=(Nx,Ny,Nz))  # Perform FFT
    kx_arr = np.fft.fftfreq( Nx, d=dx )*twopi # Find wavenumber from x
    ky_arr = np.fft.fftfreq( Ny, d=dy )*twopi # Find wavenumber from y
    kz_arr = np.fft.fftfreq( Nz, d=dz )*twopi # Find wavenumber from z
    kx, ky, kz = np.meshgrid(kx_arr,ky_arr,kz_arr,indexing='ij') # Create grid of wavenumbers
    k_lateral  = np.sqrt(kx**2+ky**2) # Combine x and y directions
    k_vertical = np.sqrt(kz**2)       # Keep z direction separate 
    # ============= FFT SPATIAL FILTER ========================================
    # Power Law Filter
    if par["filtername"].lower() == 'powlaw':
        filter_lateral = powlaw(np.abs(k_lateral),par["pl"],norm=1)
        # Check if only one element in z direction
        if len(k_vertical[0,0,:]) == 1:
            filter_vertical=filter_lateral*0+1
        else:
            filter_vertical = powlaw(np.abs(k_vertical),par["pv"],norm=1)
    # Gaussian Filter
    if par["filtername"].lower() == 'gaussian':
        # Create Filter for FFT
        kxc = par["lcc"]*twopi  # Gaussian Distribution Wave Number Center (1/km) in horizontal direction
        kxw = par["lcw"]*twopi  # Gaussian Distribution Wave Number Width (1/km)
        kzc = par["vcc"]*twopi  # Gaussian Distribution Wave Number Center (1/km) in vertical direction
        kzw = par["vcw"]*twopi  # Gaussian Distribution Wave Number Width (1/km)
        filter_lateral = gaussian(np.abs(k_lateral),kxc,kxw,norm=1)
        filter_vertical = gaussian(np.abs(k_vertical),kzc,kzw,norm=1)
    # Calculate total filter for both directions
    filter_total = filter_lateral*filter_vertical
    filter_norm  = filter_total/np.nanmax(filter_total)
    # Apply Filter
    fmag3d = cmag3d*filter_norm
    # FFT Filter Plot   
    if fltr_flg == 1:
        # Create Plot
        fig=plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111)
        # Create Data
        pk_lateral = np.logspace(-3,1, num=500)/unit_conv
        pk_vertical = np.logspace(-3,1, num=500)/unit_conv
        # Power Law Filter
        if par["filtername"].lower() == 'powlaw':
            pfilter_lateral = powlaw(np.abs(pk_lateral),par["pl"],norm=1)
            pfilter_vertical = powlaw(np.abs(pk_vertical),par["pv"],norm=1)
            ax.set_yscale('log')
            titlename = 'Power Law Coherence Filter'
        # Gaussian Filter
        if par["filtername"].lower() == 'gaussian':
            pfilter_lateral = gaussian(np.abs(pk_lateral),kxc,kxw,norm=1)
            pfilter_vertical = gaussian(np.abs(pk_vertical),kzc,kzw,norm=1)
            titlename = 'Gaussian Coherence Filter'
        # Choose a colormap
        cmap = mag_cmap('blue')
        # Get the middle value from the colormap
        #cclr = cmap(0.5)
        ax.plot(pk_lateral*unit_conv,pfilter_lateral,'k-',linewidth=6.0,label='Lateral')
        ax.plot(pk_vertical*unit_conv,pfilter_vertical,'k--',linewidth=6.0,label='Vertical')
        ax.set_xscale('log')
        ax.set_title(titlename)
        ax.set_ylabel('Normalized Filter')
        ax.set_xlabel('Wavenumber (km$^{-1}$)', labelpad=-5)
        ax.set_xlim([.001,10])
        ax.set_ylim([.001,1])
        ax.legend(fontsize=24,loc='upper right')
        plt.grid()
        fig.tight_layout()
        fig.subplots_adjust(left=0.11, right=0.95, top=.93, bottom=0.13)
        plt.show()
    # ============= MAG GRID INVERSE FFT ======================================
    fmag3d = np.real(np.fft.ifftn((fmag3d)))  # Perform Inverse FFT (might need np.real_if_close())
    # Adjust for normalization
    normscale = np.nanmax(mag3d)/np.nanmax(fmag3d)
    fmag3d = fmag3d*normscale
    # Plot 2D Grid in xy plane
    if p2d_flg == 1:
        # Bins for Spatial Plotting
        Xbin = np.linspace(-1*(Nx/2.)*dx,(Nx/2.)*dx,Nx+1)/unit_conv
        Ybin = np.linspace(-1*(Ny/2.)*dy,(Ny/2.)*dy,Ny+1)/unit_conv
        Z = np.linspace(0,1-Nz,Nz)*dz/unit_conv
        fig=plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        vm = np.max(np.abs(fmag3d[:,:,Zplt]))
        m=ax.pcolormesh(Xbin,Ybin,np.transpose(fmag3d[:,:,Zplt]),cmap=mag_cmap(),vmin=-vm,vmax=vm)
        ax.set_title('Magnetization Grid with Z = '+str(Z[Zplt])+' km')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        cbar = fig.colorbar(m)
        cbar.set_label('Magnetization (A/m)')
        plt.show()
    # Magnetization Grid in 3D Box    
    if p3d_flg == 1:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        # Determine range of maximum axis
        maxax = np.max(np.array([Nx*dx,Ny*dy,Nz*dz]))
        ax.set_xlim([0,maxax])
        ax.set_ylim([0,maxax])
        ax.set_zlim([0,maxax/10]) # Iffset z axis to see zoomed version
        # Create X and Y Grids
        X = np.arange(0, Nx*dx, dx)
        Y = np.arange(0, Ny*dy, dy)
        X, Y = np.meshgrid(X, Y)
        # Create Plot of Top surface
        cmap = mag_cmap()
        maxm = np.nanmax(np.abs(fmag3d))*.5
        norm = clr.Normalize(vmin=-maxm, vmax=maxm)
        nz = fmag3d[:,:,0]
        ax.plot_surface(X, Y, X*0+(Nz*dz)-dz,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Plot Front Surface
        X =  np.arange(0, Nx*dx, dx)
        Z =  np.arange(0, Nz*dz, dz)
        X, Z = np.meshgrid(X, Z, indexing='xy')
        nz = fmag3d[0,:,::-1].T
        ax.plot_surface(X, X*0, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable2.set_array([])
        # Plot Right Surface
        Y =  np.arange(0, Ny*dy, dy)
        Z =  np.arange(0, Nz*dz, dz)
        Y, Z = np.meshgrid(Y, Z, indexing='xy')
        nz = fmag3d[:,-1,::-1].T
        ax.plot_surface(Y*0+(Nx*dx)-dx, Y, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Turn axis off, with no colorbar
        plt.axis('off')
        fig.colorbar(mappable, shrink=1, aspect=10,orientation='vertical',pad=0,label='Magnetization (A/m)')
        fig.tight_layout()
        plt.show()
    return fmag3d
# ================= IMPACT DEMAGNETIZATION GRID ===============================
def demag(fmag3d,Nx,Ny,Nz,dx,dy,dz,r1=125,r2=175,p2d_flg=0,Zplt=10,p3d_flg=0,km=1):
    """
    Function to create demagnetized impact crater in grid.

    Parameters
    ----------
    fmag3d : array_like
        3D Matrix of Coherent Magnetization grid.
    Nx : Int
        Number of points in x direction of grid.
    Ny : Int
        Number of points in y direction of grid..
    Nz : Int
        Number of points in z direction of grid.
    dx : Scalar
        Spacing in x direction of grid.
    dy : Scalar
        Spacing in y direction of grid.
    dz : Scalar
        Spacing in z direction of grid.
    r1 : Scalar, optional
        Inner radius for impact crater. The default is 125.
    r2 : Scalar, optional
        Outer radius for impact crater for linear increase. The default is 175.
    p2d_flg : Int, optional
        If set to 1, will plot mag values in xy plane. The default is 0.
    Zplt : Int, optional
        Z value for xy plot. The default is 20.
    p3d_flg : Int, optional
        If set to 1, will plot 3d grid. The default is 0.
    km : Int, optional
        If set to 1 will use km for plots instead of meters

    Returns
    -------
    demag3d : array_like
        Demagnetized grid from impact crater.
    """
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    # Spatial Coordinates
    X = np.linspace(-1*(Nx/2),(Nx/2)-1,Nx)*dx
    Y = np.linspace(-1*(Ny/2),(Ny/2)-1,Ny)*dy
    Z = np.linspace(0,1-Nz,Nz)*dz
    # Find Crater Demag Grid
    X2,Y2     = np.meshgrid(X,Y,indexing='ij')    # Create 2D X and Y arrays
    XY        = (X2**2 + Y2**2)**.5               # Find Radial Distance From Center
    imp_demag = np.ones((Nx,Ny))                  # Initialize impact demagnetization grid
    imp_demag[XY < r1] = 0.0                      # Set MAG Values to zero within set radius
    # Set MAG Values to linearly increase from set radius
    imp_demag[(XY >=r1) & (XY <=r2)] = (XY[(XY >=r1) & (XY <=r2)]-r1)*(1/(r2-r1))
    # Combine demag and mag grids to produce final impact mag grid
    demag3d     = fmag3d*np.broadcast_to(imp_demag.transpose(),(Nz,Ny,Nx)).transpose()
    # Magnetization Grid in XY Plane    
    if p2d_flg == 1:
        # Bins for Spatial Plotting
        Xbin = np.linspace(-1*(Nx/2.)*dx,(Nx/2.)*dx,Nx+1)/unit_conv
        Ybin = np.linspace(-1*(Ny/2.)*dy,(Ny/2.)*dy,Ny+1)/unit_conv
        fig=plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        vm = np.max(np.abs(demag3d[:,:,Zplt]))
        m=ax.pcolormesh(Xbin,Ybin,np.transpose(demag3d[:,:,Zplt]),cmap=mag_cmap(),vmin=-vm,vmax=vm)
        ax.set_title('Demagnetization Grid with Z = '+str(Z[Zplt]/unit_conv)+' km')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        cbar = fig.colorbar(m)
        cbar.set_label('Magnetization (A/m)')
        fig.tight_layout()
        plt.show()
    # Magnetization Grid in 3D Box    
    if p3d_flg == 1:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        # Determine range of maximum axis
        maxax = np.max(np.array([Nx*dx,Ny*dy,Nz*dz]))
        ax.set_xlim([0,maxax])
        ax.set_ylim([0,maxax])
        ax.set_zlim([0,maxax/5]) # Iffset z axis to see zoomed version
        # Create X and Y Grids
        X = np.arange(0, Nx*dx, dx)
        Y = np.arange(0, Ny*dy, dy)
        X, Y = np.meshgrid(X, Y)
        # Create Plot of Top surface
        cmap = mag_cmap()
        maxm = np.nanmax(np.abs(demag3d))*.5
        norm = clr.Normalize(vmin=-maxm, vmax=maxm)
        nz = demag3d[:,:,0]
        ax.plot_surface(X, Y, X*0+(Nz*dz)-dz,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Plot Front Surface
        X =  np.arange(0, Nx*dx, dx)
        Z =  np.arange(0, Nz*dz, dz)
        X, Z = np.meshgrid(X, Z, indexing='xy')
        nz = demag3d[0,:,::-1].T
        ax.plot_surface(X, X*0, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable2.set_array([])
        # Plot Right Surface
        Y =  np.arange(0, Ny*dy, dy)
        Z =  np.arange(0, Nz*dz, dz)
        Y, Z = np.meshgrid(Y, Z, indexing='xy')
        nz = demag3d[:,-1,::-1].T
        ax.plot_surface(Y*0+(Nx*dx)-dx, Y, Z,rstride=2,cstride=2,facecolors=cmap(norm(nz)),
                         linewidth=0, antialiased=False)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        # Turn axis off, with no colorbar
        plt.axis('off')
        fig.colorbar(mappable, shrink=1, aspect=10,orientation='vertical',pad=0,label='Magnetization (A/m)')
        fig.tight_layout()
        plt.show()
    return demag3d
# ==== FIND B AT ALTITUDE Z0 USING ARRAY OPERATIONS FOR MULTIPLE ALTITUDES ====      
def mag_alt(demag3d,alt_arr,Nx,Ny,Nz,dx,dy,dz,mvec=np.array([0,1,1]),p2d_flg=0,Zplt=0,km=1,verbose=1,center=0):
    """
    Function to find magnetic field at given altitude using Blakely model.
    Implements array operations for faster computations.
    4D - (X,Y,Z,Altitudes)
    
    Parameters
    ----------
    demag3d : array_like
        3D Matrix of Coherent Magnetization grid.
    alt_arr : array_like
        Array of altitudes (meters)
    Nx : Int
        Number of points in x direction of grid.
    Ny : Int
        Number of points in y direction of grid..
    Nz : Int
        Number of points in z direction of grid.
    dx : Scalar
        Spacing in x direction of grid.
    dy : Scalar
        Spacing in y direction of grid.
    dz : Scalar
        Spacing in z direction of grid.
    mvec: Array (3)
        3 element array of unit magnetization vector
    p2d_flg : Int, optional
        If set to 1, will plot mag values in xy plane. The default is 0.
    Zplt : Int, optional
        Z value for xy plot. The default is 0.
    km : Int, optional
        If set to 1 will use km for plots instead of meters
    verbose: Int, optional
        If set to 1, will output current step of code
    center: Int, optional
        If set, will only output B at center locations for given altitudes

    Returns
    -------
    B : array_like (3D)
        Magnetic Field (3D) matrix for x and y locations at given altitudes
    """    
    ###########################################################################
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    twopi = 2.0*np.pi
    # Constants
    Cm = 1e-7 # Henry/meter
    t2nt = 1e9 # Tesla to nanotesla
    # Unit Magnetization Vector Direction
    mvec_unit = mvec/np.sqrt(np.sum(mvec**2.0))
    # Create Wavenumber grid
    kx_arr    = np.fft.fftfreq(Nx,d=dx)*twopi # Find wavenumber from x
    ky_arr    = np.fft.fftfreq(Ny,d=dy)*twopi # Find wavenumber from y
    kx, ky    = np.meshgrid(kx_arr,ky_arr,indexing='ij') # Create grid of wavenumbers
    k_lateral = np.sqrt(kx**2+ky**2) # Combine x and y directions
    ###########################################################################
    # Find theta variable m (real part + complex part)
    kind = k_lateral!=0
    thetam = mvec_unit[2] + np.divide((mvec_unit[0]*kx+mvec_unit[1]*ky), k_lateral, out=np.zeros_like(k_lateral, dtype=np.float32), where=kind)*1j
    # Find theta components
    thetabx = np.divide(kx, k_lateral, out=np.zeros_like(k_lateral, dtype=np.float32), where=kind)*1j
    thetaby = np.divide(ky, k_lateral, out=np.zeros_like(k_lateral, dtype=np.float32), where=kind)*1j
    thetabz = np.ones((Nx, Ny), dtype=np.complex64)
    # find nan cases due to dividing by zero (k_lateral = 0)
    thetabz[~kind] = 0
    thetam[~kind]  = 0
    # Add new dimensions for each altitude and convert to complex number
    alt_len = len(alt_arr)
    cdemag4d = np.broadcast_to((demag3d.astype(np.complex64)).transpose(),(alt_len,Nz,Ny,Nx)).transpose()
    # Convert to lower precision
    cdemag4d = cdemag4d.astype(np.complex64)
    # Perform FFT Along Horizontal Directions
    if verbose == 1:
        print('Performing FFT')
    cmagtrans = fftn(cdemag4d,s=(Nx,Ny),axes=(0,1))
    # Ensure arrays are 4D
    k_lateral4d = np.broadcast_to(k_lateral.transpose(),(alt_len,Nz,Ny,Nx)).transpose()
    thetam4d    = np.broadcast_to(thetam.transpose(),(alt_len,Nz,Ny,Nx)).transpose()
    # Depth from surface at altitude z0 to grid layer
    z1_arr = alt_arr[:,None]+np.arange(0,Nz*dz,dz)
    z1     = np.broadcast_to((z1_arr).transpose(),(Nx,Ny,Nz,alt_len))
    z2     = np.broadcast_to((z1_arr+dz).transpose(),(Nx,Ny,Nz,alt_len))
    # Calculate total field anomaly (Blakely 11.51)
    if verbose == 1:
        print('Computing FIELD')
    btrans = 2.0*np.pi*Cm*t2nt*cmagtrans*thetam4d*( np.exp(-1*k_lateral4d*z1)-np.exp(-1*k_lateral4d*z2))    
    bxtrans = np.broadcast_to(thetabx.transpose(),(alt_len,Nz,Ny,Nx)).transpose()*btrans
    bytrans = np.broadcast_to(thetaby.transpose(),(alt_len,Nz,Ny,Nx)).transpose()*btrans
    bztrans = np.broadcast_to(thetabz.transpose(),(alt_len,Nz,Ny,Nx)).transpose()*btrans
    # Perform IFFT
    if verbose == 1:
        print('Performing IFFT')
    # Convert to lower precision
    bxtrans = bxtrans.astype(np.complex64)
    bytrans = bytrans.astype(np.complex64)
    bztrans = bztrans.astype(np.complex64)
    # Perform IFFT using scipy.fft.ifftn
    bxi = ifftn(bxtrans, axes=(0, 1))
    byi = ifftn(bytrans, axes=(0, 1))
    bzi = ifftn(bztrans, axes=(0, 1))
    #bxi = np.fft.ifftn(bxtrans,axes=(0,1))
    #byi = np.fft.ifftn(bytrans,axes=(0,1))
    #bzi = np.fft.ifftn(bztrans,axes=(0,1))
    # Take Real Part
    if verbose == 1:
        print('Summing Vector Field')
    # Check if only center value of B
    if center == 1:
        bx = np.sum(bxi[int(Nx/2),int(Ny/2),:,:].real,axis=0)
        by = np.sum(byi[int(Nx/2),int(Ny/2),:,:].real,axis=0)
        bz = np.sum(bzi[int(Nx/2),int(Ny/2),:,:].real,axis=0)
    else:
        bx = np.sum(bxi.real,axis=2)
        by = np.sum(byi.real,axis=2)
        bz = np.sum(bzi.real,axis=2)
    # Sum B Vector Field
    B = np.sqrt(bx**2+by**2+bz**2)
    # Plot Figure    
    if p2d_flg == 1:
        # Bins for Spatial Plotting
        Xbin = np.linspace(-1*(Nx/2.)*dx,(Nx/2.)*dx,Nx+1)/unit_conv
        Ybin = np.linspace(-1*(Ny/2.)*dy,(Ny/2.)*dy,Ny+1)/unit_conv
        fig=plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        #vm = np.max(np.abs(demag3d[:,:,Zplt]))
        m=ax.pcolormesh(Xbin,Ybin,B[:,:,Zplt],cmap='nipy_spectral',vmin=0,vmax=np.nanmax(B[:,:,Zplt]))
        ax.set_title('Magnetic Field Magnitude Grid at Z = '+str(alt_arr[Zplt]/unit_conv)+' km')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        cbar = fig.colorbar(m)
        #cbar.set_ticks([0,35,70,105,140])
        cbar.set_label('B (nT)')
        plt.show()    
        #fig.savefig('Plots/'+str(np.abs(z0))+'Alt_Magneticfield.png')
    return B
# ================= FIND RADIAL AVERAGE B =====================================      
def mag_ravg(B,alt,Nx,Ny,Nz,dx,dy,dz,snum,p2d_flg=1,km=1):
    """
    Function to find radial average of B from center of grid for given altitude
    and set number of simulations.
    
    Parameters
    ----------
    B : array_like
        Magnetic Field at given altitude -> 4D, [x,y,alt,simulation number]
    alt : array_like
        Array of altitudes (meters)
    Nx : Int
        Number of points in x direction of grid.
    Ny : Int
        Number of points in y direction of grid..
    Nz : Int
        Number of points in z direction of grid.
    dx : Scalar
        Spacing in x direction of grid.
    dy : Scalar
        Spacing in y direction of grid.
    dz : Scalar
        Spacing in z direction of grid.
    snum: Int
        Number of simulations processed
    p2d_flg : Int, optional
        If set to 1, will plot mag values in xy plane. The default is 0.
    km : Int, optional
        If set to 1 will use km for plots instead of meters

    Returns
    -------
    Bavg : array_like (3D)
        Averaged Magnetic Field (3D) matrix for x and y locations at given altitudes
    Bstd : array_like (3D)
        STD DEV Magnetic Field (3D) matrix for x and y locations at given altitudes
    """    
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    alt_arr = alt/unit_conv
    # Average over number of sims
    Bavg  = np.nanmean(B,axis=3)
    Bstd  = np.nanstd(B,axis=3)
    if p2d_flg == 1:
        # Find Averaged Magnetic Field Values
        X = np.linspace(-1*(Nx/2),(Nx/2)-1,Nx)*dx/unit_conv
        Y = np.linspace(-1*(Ny/2),(Ny/2)-1,Ny)*dy/unit_conv
        # Find Radial Distance from Crater center
        X2,Y2     = np.meshgrid(X,Y,indexing='ij') 
        R = np.sqrt(X2**2+Y2**2)
        delr = dx/unit_conv
        Rbin = np.arange(0,int((np.max(np.abs(X))+np.max(np.abs(Y)))/2),delr)
        R_cntr = Rbin[0:-1]+delr/2.0
        # Plot
        fig,ax=plt.subplots(1,2,figsize=(18, 9))
        # Iterate over each altitude
        for ai in range(0,len(Bavg[0,0,:])):
            Bai = Bavg[:,:,ai]
            # Compute Averages and STDs
            Baim    = stats.binned_statistic(R.flatten(), Bai.flatten(), statistic='mean', bins=Rbin)
            Bais    = stats.binned_statistic(R.flatten(), Bai.flatten(), statistic='std', bins=Rbin)
            Bai_avg = Baim.statistic
            Bai_std = Bais.statistic
            line1, = ax[0].plot(R_cntr,Bai_avg,'-',linewidth=5,label=str(alt_arr[ai])+' km')
            ax[0].errorbar(R_cntr, Bai_avg, Bai_std, linestyle='None', marker='^',color=line1.get_color())
            ax[0].fill_between(R_cntr, Bai_avg-Bai_std, 
                               Bai_avg+Bai_std, color=line1.get_color(), alpha=.2)
            line2, = ax[1].plot(R_cntr,Bai_avg/np.nanmax(Bai_avg),'-',linewidth=5,label=str(alt_arr[ai])+' km')
            ax[1].errorbar(R_cntr, Bai_avg/np.nanmax(Bai_avg), Bai_std/np.nanmax(Bai_avg), linestyle='None', marker='.',color=line1.get_color())
            ax[1].fill_between(R_cntr, Bai_avg/np.nanmax(Bai_avg)-Bai_std/np.nanmax(Bai_avg), 
                               Bai_avg/np.nanmax(Bai_avg)+Bai_std/np.nanmax(Bai_avg), color=line1.get_color(), alpha=.2)
        ax[0].set_yscale('log')
        ax[0].set_title('Radial <|B|> over '+str(snum)+' Sims')
        ax[0].set_ylabel('<|B|> (nT)')
        ax[0].set_xlabel('R (km)')
        ax[0].set_xlim([0,500])
        ax[0].set_ylim([0.1,1000])
        ax[0].grid(True)
        ax[0].legend(fontsize=24)
        ax[1].set_title('Radial Norm <|B|> over '+str(snum)+' Sims')
        ax[1].set_ylabel('Norm <|B|>')
        ax[1].set_xlabel('R (km)')
        ax[1].set_xlim([0,500])
        ax[1].set_ylim([0,1.2])
        ax[1].grid(True)
        ax[1].legend(fontsize=24)
        fig.tight_layout()
        plt.show()
    
    return Bavg,Bstd  
# ================= FIND Altitude Profile from Averaged B =====================      
def alt_curve(Bavg,Bstd,alt_arr,Nx,Ny,Nz,dx,dy,dz,snum,km=1):       
    """
    Function to plot altitude profile from averaged magnetic field for given
    location.
    
    Parameters
    ----------
    Bavg : array_like (3D)
        Averaged Magnetic Field (3D) matrix for x and y locations at given altitudes
    Bstd : array_like (3D)
        STD DEV Magnetic Field (3D) matrix for x and y locations at given altitudes
    alt_arr : array_like
        Array of altitudes (meters)
    Nx : Int
        Number of points in x direction of grid.
    Ny : Int
        Number of points in y direction of grid..
    Nz : Int
        Number of points in z direction of grid.
    dx : Scalar
        Spacing in x direction of grid.
    dy : Scalar
        Spacing in y direction of grid.
    dz : Scalar
        Spacing in z direction of grid.
    snum: Int
        Number of simulations processed
    km : Int, optional
        If set to 1 will use km for plots instead of meters

    Returns
    -------
    Bavg_loc : array_like (3D)
        Averaged Magnetic Field (3D) matrix for given locations at given altitudes
    Bstd_loc : array_like (3D)
        STD DEV Magnetic Field (3D) matrix for given locations at given altitudes
    """    
    # Check whether to use km or meters
    unit_conv = 1
    if km == 1:
        unit_conv =1e3
    alt_arr = alt_arr/unit_conv
    # Obtain Center B
    Bavg_loc = Bavg[int(Nx/2),int(Ny/2),:]
    Bstd_loc = Bstd[int(Nx/2),int(Ny/2),:]
    
    
    # Create Figure
    fig=plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.errorbar(Bavg_loc,alt_arr,xerr=Bavg_loc*.15,fmt='k-',linewidth=6,elinewidth=3,zorder=2,label='Model')
    
    ax.fill_betweenx(alt_arr, Bavg_loc-Bstd_loc, Bavg_loc+Bstd_loc,color='gray',zorder=1,alpha=0.5)
    ax.set_xlabel('Average |B| (nT)',fontsize=40)
    ax.set_ylabel('Altitude (km)',fontsize=40)
    #ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1e0,1e3])
    ax.set_ylim([1,500])
    #p185 = ax.plot(46.2,185,'rD',markersize=35,zorder=3,label='MGS B$\mathregular{_{185}}$')
    #ax.plot(24.6,233,'ro',markersize=35,zorder=3)
    #p400 = ax.plot(6.36,400,'co',markersize=35,zorder=3,label='MGS B$\mathregular{_{400}}$')
    #ax.quiver(2013,75, 0,-.33,scale=1,color='b',label='Insight B$\mathregular{_{0}}$',headwidth=4,headlength=6,headaxislength=6,width=.0115)
    #ax.plot([2013,2013],[.001,1e5],'b--',lw=10)
    ax.grid(True)
    ax.legend(fontsize=40)
    ax.set_title('Average Altitude Profile Over '+str(snum)+' Simulations')
    plt.show()
    # Return specific B values at given location
    return Bavg_loc,Bstd_loc
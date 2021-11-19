"""
Homework: Self-organizing maps
Course  : Data Mining II (636-0019-00L)

Auxiliary functions to help in the implementation of an online version
of the self-organizing map (SOM) algorithm.
"""
# Author: Dean Bodenham, May 2016
# Modified by: Damian Roqueiro, May 2017

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Circle

"""
A function to create the S curve
"""
def makeSCurve():
    n_points = 1000
    noise = 0.2
    X, color = datasets.samples_generator.make_s_curve(n_points, noise=noise, random_state=0)
    Y = np.array([X[:,0], X[:,2]])
    Y = Y.T
    # Stretch in all directions
    Y = Y * 2
    
    # Now add some background noise
    xMin = np.min(Y[:,0])
    xMax = np.max(Y[:,0])
    yMin = np.min(Y[:,1])
    yMax = np.max(Y[:,1])
    
    n_bg = n_points/5
    Ybg = np.zeros(shape=(n_bg,2))
    Ybg[:,0] = np.random.uniform(low=xMin, high=xMax, size=n_bg)
    Ybg[:,1] = np.random.uniform(low=yMin, high=yMax, size=n_bg)
    
    Y = np.concatenate((Y, Ybg))
    return Y


"""
Plot the data and SOM for the S-curve
  data: 2 dimensional dataset (first two dimensions are plotted)
  buttons: N x 2 array of N buttons in 2D
  fileName: full path to the output file (figure) saved as .pdf or .png
"""
def plotDataAndSOM(data, buttons, fileName):
    fig = plt.figure(figsize=(8, 8))
    # Plot the data in grey
    plt.scatter(data[:,0], data[:,1], c='grey')
    # Plot the buttons in large red dots
    plt.plot(buttons[:,0], buttons[:,1], 'ro', markersize=10)
    # Label axes and figure
    plt.title('S curve dataset, with buttons in red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(fileName)


# Important note:
# 
# Most of the functions below are currently just headers. Provide a function
# body for each of them. 
#
# In case you want to create your own functions with their own interfaces, adjust
# the rest of the code appropriately.

"""
Create a grid of points, dim p x q, and save grid in a (p*q, 2) array
  first column: x-coordinate
  second column: y-coordinate
"""

def createGrid(p, q):
    index=0
    grid=np.zeros(shape=(p*q, 2))
    for i in range(p):
        for j in range(q):
            index = i*q + j
            grid[index, 0] = i
            grid[index, 1] = j
    return grid


"""
A function to plot the crabs results
It applies a SOM previously computed (parameters grid and buttons) to a given
dataset (parameters data)

Parameters
 X : is the original data that was used to compute the SOM.
     Rows are samples and columns are features.
 idInfo : contains the information (sp and sex for the crab dataset) about
          each data point in X.
          The rows in idInfo match one-to-one to rows in X.
 grid, buttons : obtained from computing the SOM on X.
 fileName : full path to the output file (figure) saved as .pdf or .png
"""
def plotSOMCrabs(X, idInfo, grid, buttons, fileName):
    # Use the following colors for samples of each pair [species, sex]
    # Blue male:     dark blue #0038ff
    # Blue female:   cyan      #00eefd
    # Orange male:   orange    #ffa22f
    # Orange female: yellow    #e9e824

    # TODO replace statement below with function body
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i in range(grid.shape[0]):
        ax = fig.add_subplot(111)
        cir1 = Circle(xy = (grid[i]), radius=0.25, alpha=1, color = 'black', fill=False) 
        ax.add_patch(cir1)
        plt.axis('scaled')
        plt.axis('equal') 
    
    position = X[:,0:2]
    for i in range(X.shape[0]):
        index = int(idInfo.loc[i,'label'])
        position[i] = grid[index] + np.hstack((np.random.uniform(-0.20, 0.20),np.random.uniform(-0.20, 0.20)))
        if idInfo.loc[i,'sp'] == 'B':
            if idInfo.loc[i,'sex'] == 'M':
                plt.scatter(position[i][0], position[i][1], c='#0038ff')
            if idInfo.loc[i,'sex'] == 'F':
                plt.scatter(position[i][0], position[i][1], c='#00eefd')
        if idInfo.loc[i,'sp'] == 'O':
            if idInfo.loc[i,'sex'] == 'M':
                plt.scatter(position[i][0], position[i][1], c='#ffa22f')
            if idInfo.loc[i,'sex'] == 'F':
                plt.scatter(position[i][0], position[i][1], c='#e9e824')
    plt.savefig(fileName)


"""
Function for computing distance in grid space.
Use Euclidean distance.
"""

def getGridDist(z0, z1):
    # TODO replace statement below with function body
    Gdistance = np.linalg.norm(z0 - z1)
    return Gdisatance


"""
Function for computing distance in feature space.
Use Euclidean distance.
"""
def getFeatureDist(z0, z1):
    # TODO replace statement below with function body
    Fdistance = np.linalg.norm(z0 - z1)
    return Fdistance

"""
Create distance matrix between points numbered 1,2,...,K=p*q from grid
"""
def createGridDistMatrix(grid):
    # TODO replace statement below with function body
    E = sp.spatial.distance.pdist(grid,'euclidean')
    buttonDist = sp.spatial.distance.squareform(E)
    return buttonDist


"""
Create array for epsilon. Values in the array decrease to 1.
"""
def createEpsilonArray(epsilon_max, N):
    # TODO replace statement below with function body
    Earray = np.linspace(epsilon_max,1,N)
    return Earray


"""
Create array for alpha. Values in the array decrease to 0.
"""
def createAlphaArray(alpha_max, N):
    # TODO replace statement below with function body
    Aarray = np.linspace(alpha_max,0,N)
    return Aarray


"""
X is whole data set, K is number of buttons to choose
"""
def initButtons(X, K):
    # TODO replace statement below with function body
    buttons = X[np.random.randint(X.shape[0], size= K),:]
    return buttons


"""
x is one data point, buttons is the grid in FEATURE SPACE
"""
def findNearestButtonIndex(x, buttons):
    # TODO replace statement below with function body
    D = []
    for i in range(len(buttons)):
        D = np.append(D, getFeatureDist(x, buttons[i]))
    index = np.where(D == np.min(D))[0][0]
    return index


"""
Find all buttons within a neighborhood of epsilon of index IN GRID SPACE 
(return a boolean vector)
"""
def findButtonsInNhd(index, epsilon, buttonDist):
    # TODO replace statement below with function body
    switch = []
    for i in range(buttonDist.shape[0]):
        if buttonDist[index][i] <= epsilon:
            switch = np.append(switch, 1)
        else:
            switch = np.append(switch, 0)
    return switch

"""
Do gradient descent step, update each button position IN FEATURE SPACE
"""
def updateButtonPosition(button, x, alpha, switch):
    # TODO replace statement below with function body
    x1 = np.tile(x,(button.shape[0],1))
    switch1 = np.tile(switch,(button.shape[1],1)).T
    newbuttons = button-alpha*(button-x1)*switch1
    return newbuttons

"""
Compute the squared distance between data points and their nearest button
"""
def computeError(data, buttons):
    # TODO replace statement below with function body
    error = 0
    for i in range(data.shape[0]):
        x = data[i]
        Nbutton = buttons[findNearestButtonIndex(x, buttons)]
        errordistance = getFeatureDist(x,Nbutton)
        error = error + errordistance
    return error

"""
Implementation of the self-organizing map (SOM)

Parameters
 X : data, rows are samples and columns are features
 p, q : dimensions of the grid
 N : number of iterations
 alpha_max : upper limit for learning rate
 epsilon_max : upper limit for radius
 compute_error : boolean flag to determine if the error is computed.
                 The computation of the error is time-consuming and may
                 not be necessary every time the function is called.
                 
Returns
 buttons, grid : the buttons and grid of the newly created SOM
 error : a vector with error values. This vector will contain zeros if 
         compute_error is False

TODO: Complete the missing parts in this function following the pseudocode
      in the homework sheet
"""
def SOM(X, p, q, N, alpha_max, epsilon_max, compute_error=False):
    # 1. Create grid and compute pairwise distances
    grid = createGrid(p, q)
    gridDistMatrix = createGridDistMatrix(grid)
    
    # 2. Randomly select K out of d data points as initial positions
    #    of the buttons
    K = p * q
    d = X.shape[0]
    buttons = initButtons(X, K)
    
    # 3. Create a vector of size N for learning rate alpha
    Aarray = createAlphaArray(alpha_max, N)
    # 4. Create a vector of size N for epsilon 
    Earray = createEpsilonArray(epsilon_max, N)
    # Initialize a vector with N zeros for the error
    # This vector may be returned empty if compute_error is False
    error = np.zeros(N)

    # 5. Iterate N times
    for i in range(N):
        # 6. Initialize/update alpha and epsilon
        alpha = Aarray[i]
        epsilon = Earray[i]
        # 7. Choose a random index t in {1, 2, ..., d}
        t = np.random.randint(d, size= 1)
        x = X[t,:]
        # 8. Find button m_star that is nearest to x_t in F 
        index = findNearestButtonIndex(x, buttons)
        # 9. Find all grid points in epsilon-nhd of m_star in GRID SPACE  
        switch = findButtonsInNhd(index, epsilon, gridDistMatrix)
        # 10. Update position (in FEATURE SPACE) of all buttons m_j
        #     in epsilon-nhd of m_star, including m_star
        buttons = updateButtonPosition(buttons, x, alpha, switch)
        # Compute the error 
        # Note: The computation takes place only if compute_error is True
        #       Replace the statement below with your code
        if compute_error == True:
            error[i] = computeError(X, buttons)

    # 11. Return buttons, grid and error
    return (buttons, grid, error)


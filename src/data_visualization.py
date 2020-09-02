import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

# pca is not good, data variance is difference, 
# try normalization data

def pca_analysis():
    data = np.loadtxt(sys.argv[1])
    means= np.mean(data,0)
    stds = np.std(data,0)
    data = (data - means)/stds
    pca = PCA()
    pca.fit(data)

    print(pca.explained_variance_ratio_)
    print(pca.components_)
    return True
def grid_depth_variance(image_shape=None):
    grid_size = 5
    data = np.loadtxt(sys.argv[1])
    data_min = np.min(data,0)
    data_max = np.max(data,0)
    if image_shape is not None:
        data_min[0:2]=0
        data_max[0:2]=image_shape
    grid_col_n,grid_row_n  = (data_max[0:2] - data_min[0:2])//grid_size
    grid_col_n = int(grid_col_n)+1
    grid_row_n = int(grid_row_n)+1
    depth_grid =[ [[] for i in range(0,grid_col_n)] for j in range(0,grid_row_n)]
    for i in range(0,data.shape[0]):
        grid_col_id = int((data[i,0]-data_min[0])//grid_size)
        grid_row_id = int((data[i,1]-data_min[1])//grid_size)
        depth_grid[grid_row_id][grid_col_id].append(data[i][4])
    depth_variance_grid = np.zeros((grid_row_n,grid_col_n))
    depth_mean_grid = np.zeros((grid_row_n,grid_col_n))
    for i_row in range(0,grid_row_n):
        for i_col in range(0,grid_col_n):
            depth_variance_grid[i_row,i_col] = np.std(depth_grid[i_row][i_col])
            depth_mean_grid[i_row,i_col] = np.mean(depth_grid[i_row][i_col])
    ax1 = plt.subplot(121)
    im1 = ax1.imshow(depth_variance_grid,cmap='cool')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im1, cax=cax)
    ax2 = plt.subplot(122)
    im2 = ax2.imshow(depth_mean_grid,cmap='cool')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im2, cax=cax2)

    plt.show()
    np.savetxt('depth_variance_grid.txt',depth_variance_grid)
    np.savetxt('depth_mean_grid.txt',depth_mean_grid)

def depth_flow_coor_coorelation():
    data = np.loadtxt(sys.argv[1])
    means= np.mean(data,0)
    stds = np.std(data,0)
    data = (data - means)/stds
    pca = PCA()
    pca.fit(data)
    # each row is a new basis
    print(pca.explained_variance_ratio_)
    print(pca.components_)
    return True
       


    
def main():
    grid_depth_variance([1241,376])

if __name__ == '__main__':
    main()

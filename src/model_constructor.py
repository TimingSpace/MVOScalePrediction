import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ModelConstructor:
    def __init__(self,model_parameters):
        self.model_parameters = model_parameters
        self.image_shape = np.array(model_parameters['image_shape'])
        self.grid_size   = model_parameters['grid_size']
        grid_col_n,grid_row_n  = self.image_shape//self.grid_size
        self.grid_col_n = int(grid_col_n)+1
        self.grid_row_n = int(grid_row_n)+1
        self.depth_grid =[ [[] for i in range(0,self.grid_col_n)] for j in range(0,self.grid_row_n)]
        self.model = np.zeros((self.grid_row_n,self.grid_col_n,2))

    def process(self,R,t,feature2d,feature3d,scale):
        if feature2d is None:
            return
        for i in range(0,feature2d.shape[0]):
            grid_col_id = int((feature2d[i,0])//self.grid_size)
            grid_row_id = int((feature2d[i,1])//self.grid_size)
            self.depth_grid[grid_row_id][grid_col_id].append(feature3d[i][2]*scale)

    def mean(self):
        for i_row in range(0,self.grid_row_n):
            for i_col in range(0,self.grid_col_n):
                self.model[i_row,i_col,0] = np.std(self.depth_grid[i_row][i_col])
                self.model[i_row,i_col,1] = np.mean(self.depth_grid[i_row][i_col])
        return self.model

    def model_diff(self):
        return 0
    def visualization(self):
        depth_variance_grid = self.model[:,:,0]
        depth_mean_grid = self.model[:,:,1]
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
    def write(self,name):
        np.savetxt(name,model)

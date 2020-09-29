import matplotlib.pyplot as plt 
import numpy as np
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
def main():
    depth_variance_grid = np.loadtxt(sys.argv[1])
    depth_mean_grid     = np.loadtxt(sys.argv[2])
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

    depth_reliable = depth_variance_grid
    depth_reliable[depth_variance_grid>10]=0
    plt.imshow(depth_reliable)
    plt.show()
if __name__ == '__main__':
    main()

import numpy as np

class ScalePredictor:
    def __init__(self,depth_mean_grid,depth_variance_grid,grid_size):
        self.depth_mean_grid     = depth_mean_grid
        self.depth_std_grid = depth_variance_grid 
        self.grid_size           = grid_size


    def get_depth(self,u,v):
        i_col = int(u//self.grid_size)
        i_row = int(v//self.grid_size)
        depth_mean = self.depth_mean_grid[i_row,i_col]
        depth_std  = self.depth_std_grid[i_row,i_col]
        return depth_mean,depth_std
    
    def scale_fusion(self,scales,stds):
        variance = stds*stds
        scale   = np.sum(scales/variance)/np.sum(1/variance)
        return scale

    def scale_predict(self,feature_uv,feature_depth):
        assert feature_uv.shape[0] == feature_depth.shape[0]
        scales = np.zeros(feature_uv.shape[0])
        stds = np.zeros(feature_uv.shape[0])
        for feature_id  in range(feature_uv.shape[0]):
            u,v = feature_uv[feature_id]
            depth_abs,depth_std = self.get_depth(u,v)
            scales[feature_id]  = depth_abs/feature_depth[feature_id]
            stds[feature_id]    = depth_std
        scale =  self.scale_fusion(scales,stds)
        return scale


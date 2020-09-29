import numpy as np

class ScalePredictor:
    def __init__(self,depth_mean_grid,depth_variance_grid,grid_size,threshold=20):
        self.depth_mean_grid     = depth_mean_grid
        self.depth_std_grid = depth_variance_grid 
        self.grid_size           = grid_size
        self.variance_threshold  = threshold


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
        scales = []
        stds =   []
        for feature_id  in range(feature_uv.shape[0]):
            u,v = feature_uv[feature_id]
            depth_abs,depth_std = self.get_depth(u,v)
            scale_t = depth_abs/feature_depth[feature_id]
            if depth_std<self.variance_threshold:
                scales.append(scale_t)
                stds.append(depth_std)
        scales = np.array(scales)
        stds   = np.array(stds)
        scale =  self.scale_fusion(scales,stds)
        return scale


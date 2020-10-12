import numpy as np

class StructureEncoder:
    def __init__(self,structure_parameters): 
        self.structure_parameters = structure_parameters
        self.image_shape = np.array(structure_parameters['image_shape'])
        self.grid_size   = structure_parameters['grid_size']
        grid_col_n,grid_row_n  = self.image_shape//self.grid_size
        self.grid_col_n = int(grid_col_n)+1
        self.grid_row_n = int(grid_row_n)+1
        self.depth_grid =[ [[] for i in range(0,self.grid_col_n)] for j in range(0,self.grid_row_n)]
        self.code = None
    def encoding(self,feature2d,feature3d):
        if feature2d is None:
            return
        for i in range(0,feature2d.shape[0]):
            grid_col_id = int((feature2d[i,0])//self.grid_size)
            grid_row_id = int((feature2d[i,1])//self.grid_size)
            self.depth_grid[grid_row_id][grid_col_id].append(feature3d[i][2])
        model = np.zeros((self.grid_row_n,self.grid_col_n,2))
        for i_row in range(0,self.grid_row_n):
            for i_col in range(0,self.grid_col_n):
                model[i_row,i_col,0] = len(self.depth_grid[i_row][i_col])
                if model[i_row,i_col,0] > 0:
                    model[i_row,i_col,1] = np.mean(self.depth_grid[i_row][i_col])
                else:
                    model[i_row,i_col,1] = 0
        self.code = model
        code = StructureCode(model)
        return code
    def __sub__(self,other):
        assert self.code is not None and other.code is not  None
        len_diff = np.abs(self.code[:,:,0] - other.code[:,:,0])
        mean_diff= np.abs(self.code[:,:,1] - other.code[:,:,1])
        diff = np.sum(len_diff) + np.sum(self.code[:,:,0]*other.code[:,:,0]*mean_diff)
        return diff

class StructureCode:
    def __init__(self,code=None):
        self.code = code
    def __sub__(self,other):
        assert self.code is not None and other.code is not  None
        len_diff = np.abs(self.code[:,:,0] - other.code[:,:,0])
        mean_diff= np.abs(self.code[:,:,1] - other.code[:,:,1])
        diff = np.sum(len_diff) + np.sum(self.code[:,:,0]*other.code[:,:,0]*mean_diff)
        return diff


if __name__ == '__main__':
    stru_params = {'image_shape':[1241,376],'grid_size':5}
    structure_encoder = StructureEncoder(stru_params)

    code1 = structure_encoder.encoding(np.array([[1,2],[2,3]]),np.array([[1,2,3],[2,3,4]]))
    code2 = structure_encoder.encoding(np.array([[1,2],[2,2]]),np.array([[1,2,3],[2,2,4]]))
    res = structure_encoder - structure_encoder
    res = code1-code2
    print(res)
    

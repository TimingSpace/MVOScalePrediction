import numpy as np
def cat(datas,dim):
    data_n = len(datas)
    data_shape = np.array(datas[0])
    data_dims_n  = len(data_shape)
    assert dim>=0 and dim < data_dims_n
    fixed_dims = list(range(0,dim))+list(range(dim+1,data_dims_n))
    cat_dims   = [0,data_shape[dim]]
    for i in range(1,data_n):
        data_shape_l = np.array(datas[i])
        data_dims_n_l  = len(data_shape_l)
        assert data_dims_n_l == data_dims_n
        assert data_shape_l[fixed_dims] == data_shape[fixed_dims]
        assert datas[i].dtype == datas[0].dtype
        cat_dims.append(data_shape_l[dim])

    new_shape = data_shape.copy()
    new_shape[dim] = np.sum(cat_dims)
    new_data = np.zeros(new_shape,dtype = datas[0].dtype)
    for i range(0,data_n):
        new_data[]


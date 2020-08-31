import numpy as np

def array2str(data):
    data_s = ' '.join(map(str, data))
    return data_s

def mat2str(data):
    res = ''
    for i in range(data.shape[0]):
        data_i  = data[i,:]
        data_i_s= ' '.join(map(str, data_i))
        res += data_i_s+'\n'
    return res
 
if __name__ == '__main__':
    a = np.ones((3,5))
    s = mat2str(a)
    print(s)

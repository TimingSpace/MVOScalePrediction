import numpy as np
import sys
import cv2
from  visual_odometry import VisualOdometry
from model_constructor import ModelConstructor
import visualization_utils as vu
class ScalePredictor:
    def __init__(self,scale_parameters,model):
        self.model = model
        self.scale_parameters = scale_parameters
    def process(self,feature2d,feature3d):
        scale = 1
        return scale


def model_construction(image_list,scale_gt,vo,mc):
    model = None
    vis_flag = False
    for frame_id in range(0,len(image_list[:-1])):
        image_name_cur = image_list[frame_id]
        frame    = cv2.imread(image_name_cur)
        print(image_name_cur)
        assert frame is not None
        R,t,feature3d,feature2d = vo.process(frame)
        mc.process(R,t,feature2d,feature3d,scale_gt[frame_id])
        if vis_flag:
            frame_copy = frame.copy()
            vu.draw_feature(frame_copy,feature2d)
            vu.draw_feature_depth(frame_copy,feature2d,feature3d)
            cv2.imshow('image',frame_copy)
            cv2.waitKey()

    mc.mean()
    return mc.model

def main():
    flag_train = True
    image_list_file = open(sys.argv[1])
    image_list_file.readline() # skip the  first line
    image_list = image_list_file.read().split('\n')
    scale_gt   = np.loadtxt(sys.argv[2])
    vo_params={'fx':716,'fy':716,'cx':607,'cy':189,'bucket_size':30,'dense':2,'feature_threshold':0.000007}
    vo = VisualOdometry(vo_params)
    model_params = {'image_shape':[1241,376],'grid_size':5}
    mc = ModelConstructor(model_params)
    if flag_train:
        environment_model = model_construction(image_list,scale_gt,vo,mc)
    else:
        environment_model = model_load()
    mc.visualization()
    scale_params={}
    sc = ScalePredictor(scale_params,environment_model)

    

if __name__ == '__main__':
    main()

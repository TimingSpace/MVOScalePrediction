import numpy as np
import sys
import cv2
from  visual_odometry  import VisualOdometry
from model_constructor import ModelConstructor
from scale_predictor   import ScalePredictor
import visualization_utils as vu
import utils.draw as draw

def model_construction(image_list,scale_gt,vo,mc):
    model = None
    vis_flag = False
    for frame_id in range(0,len(image_list[:-1])):
        image_name_cur = image_list[frame_id]
        frame    = cv2.imread(image_name_cur)
        print(image_name_cur)
        assert frame is not None
        flag,R,t,feature3d,feature2d = vo.process(frame)
        if flag == 'stop':
            continue
        mc.process(R,t,feature2d,feature3d,scale_gt[frame_id-1])
        if vis_flag:
            frame_copy = frame.copy()
            vu.draw_feature(frame_copy,feature2d)
            vu.draw_feature_depth(frame_copy,feature2d,feature3d)
            cv2.imshow('image',frame_copy)
            cv2.waitKey()

    mc.mean()
    return mc.model

def vo_with_scale(image_list,vo,sp):
    path = []
    pose = np.eye(4)
    path.append(pose)
    for frame_id in range(0,len(image_list[:-1])):
        image_name_cur = image_list[frame_id]
        frame    = cv2.imread(image_name_cur)
        print(image_name_cur)
        assert frame is not None,'got invalid image'
        flag,R,t,feature3d,feature2d = vo.process(frame)
        if flag == 'init':
            continue
        elif flag == 'stop':
            path.append(path[-1])
            continue
        scale = sp.scale_predict(feature2d,feature3d[:,2])
        assert scale >= 0, 'got invalid scale'
        motion = np.eye(4)
        motion[0:3,0:3] = R
        motion[0:3,3]   = (t*scale).reshape(-1)
        path.append(path[-1]@motion)

    path = np.array(path)
    return path.reshape(-1,16)[:,0:12]

def main():
    flag_train = False
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
        mc.write('environment_model')
    else:
        mc.read('environment_model')
        environment_model = mc.model

    #mc.visualization()
    scale_params={'threshold':20}
    sc = ScalePredictor(mc,scale_params)
    vo.reset()
    path = vo_with_scale(image_list,vo,sc)
    np.savetxt('path.txt',path)
    draw.draw_path(path)


    

if __name__ == '__main__':
    main()

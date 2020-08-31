import cv2
import sys
import numpy as np
import utils.draw as draw
import utils.format as form
lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

class FeatureDetector:
    def __init__(self,threshold = 0.000007,bucket_size=30,density=2):
        self.akaze = cv2.AKAZE_create(threshold=threshold)
        self.bucket_size = bucket_size
        self.density = density
    def detect(self,image):
        print('bucked feature')
        kp_akaze = self.akaze.detect(image,None) 
        px_cur = np.array([x.pt for x in kp_akaze], dtype=np.float32)
        return self.bucket(px_cur,self.bucket_size,self.density)

    def bucket(self,features,bucket_size=30,density=2):
        u_max,v_max = np.max(features,0)
        u_min,v_min = np.min(features,0)
        print(u_min,v_min,u_max,v_max)
        bucket_x = 1+(u_max )//bucket_size
        bucket_y = 1+(v_max )//bucket_size
        print(bucket_y)
        bucket = []
        for i in range(int(bucket_y)):
            buc = []
            for j in range(int(bucket_x)):
                buc.append([])
            bucket.append(buc)
        print(len(bucket))
        i_feature = 0
        for feature in features:
            u = int(feature[0])//bucket_size
            v = int(feature[1])//bucket_size
            bucket[v][u].append(i_feature)
            i_feature+=1

        #print(bucket)
        new_feature=[]
        for i in range(int(bucket_y)):
            for j in range(int(bucket_x)):
                feature_id = bucket[i][j]
                np.random.shuffle(feature_id)
                for k in range(min(density,len(feature_id))):
                    new_feature.append(features[feature_id[k]])
        
        return np.array(new_feature)

def motion_estimarion(feature_cur,feature_last,fx,cx,cy):
    camera_matrix = np.eye(3)
    camera_matrix[0,0] = camera_matrix[1,1] = fx
    camera_matrix[0,2] = cx
    camera_matrix[1,2] = cy


    E, mask = cv2.findEssentialMat(feature_cur, feature_last,cameraMatrix = camera_matrix , method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask,points_3d = cv2.recoverPose(E, feature_cur,\
         feature_last,cameraMatrix=camera_matrix,distanceThresh=100)
    print(t)
    mask_bool = np.array(mask>0).reshape(-1)
    # the 3d coordinate of feature ref
    points_3d_selected = points_3d[:,mask_bool].T
    return points_3d_selected[:,2]/points_3d_selected[:,3],mask_bool

def feature_tracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2

def dense_flow(frame_1,frame_2):
    hsv = np.zeros((frame_1.shape[0],frame_1.shape[1],3),dtype='uint8')
    flow = cv2.calcOpticalFlowFarneback(frame_1,frame_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2',bgr)
def main():
    image_name_file = open(sys.argv[1])
    image_name_file.readline()
    image_names = image_name_file.read().split('\n')
    begin_id = 0
    image_id = 0
    image_last=None
    detector = FeatureDetector()
    image_names = image_names[:-1]
    data_file = open('data.txt','a')
    for image_name in image_names:
        if image_id< begin_id:
            image_id+=1
            continue
        print(image_name)
        image = cv2.imread(image_name)
        image_show = image.copy()
        image_show_black = np.zeros(image.shape,dtype='uint8')
        #features = feature_detection(image,image_show)
        features = detector.detect(image)
        if(image_last is not None):
            feature_cur,feature_last = feature_tracking(image,image_last,features)
            print(feature_cur.shape)
            feature_flow = np.abs(feature_cur -feature_last)
            depth_f,mask = motion_estimarion(feature_cur,feature_last,716,607,189)
            data = np.stack((feature_cur[mask,0],feature_cur[mask,1],feature_flow[mask,0],feature_flow[mask,1],depth_f)).transpose()
            data_s = form.mat2str(data)
            data_file.write(data_s)
            data_file.flush()
            
        image_last = image.copy()
        image_id+=1

if __name__ == '__main__':
    main()

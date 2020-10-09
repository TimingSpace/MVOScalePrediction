
import cv2
import numpy as np
import visualization_utils as vu

class FeatureDetector:
    def __init__(self,threshold = 0.000007,bucket_size=30,density=2):
        self.akaze = cv2.AKAZE_create(threshold=threshold)
        self.bucket_size = bucket_size
        self.density = density
    def detect(self,image):
        kp_akaze = self.akaze.detect(image,None) 
        px_cur = np.array([x.pt for x in kp_akaze], dtype=np.float32)
        return self.bucket(px_cur,self.bucket_size,self.density)

    def bucket(self,features,bucket_size=30,density=2):
        u_max,v_max = np.max(features,0)
        u_min,v_min = np.min(features,0)
        bucket_x = 1+(u_max )//bucket_size
        bucket_y = 1+(v_max )//bucket_size
        bucket = []
        for i in range(int(bucket_y)):
            buc = []
            for j in range(int(bucket_x)):
                buc.append([])
            bucket.append(buc)
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

class VisualOdometry:
    def __init__(self,params):
        self.fx = params['fx']
        self.fy = params['fy']
        self.cx = params['cx']
        self.cy = params['cy']
        self.feature_bucket_size = params['bucket_size']
        self.feature_dense       = params['dense']
        self.feature_threshold   = params['feature_threshold']
        self.feature_detector    = FeatureDetector(self.feature_threshold,self.feature_bucket_size,self.feature_dense)

        self.camera_matrix = np.eye(3)
        self.camera_matrix[0,0] = self.camera_matrix[1,1] = self.fx
        self.camera_matrix[0,2] = self.cx
        self.camera_matrix[1,2] = self.cy
        self.lk_params = dict(winSize  = (21, 21), 
                        #maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.image_last = None
    def feature_tracking(self,image_ref, image_tar, px_ref):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_tar, px_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        return kp1, kp2

    def motion_estimarion(self,feature_cur,feature_last):
        E, mask = cv2.findEssentialMat(feature_cur, feature_last,cameraMatrix = self.camera_matrix , method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask,points_3d = cv2.recoverPose(E, feature_cur,\
             feature_last,cameraMatrix=self.camera_matrix,distanceThresh=100)
        print(t)
        mask_bool = np.array(mask>0).reshape(-1)
        # the 3d coordinate of feature ref
        points_3d_selected = points_3d[:,mask_bool].T
        return R,t,(points_3d_selected[:,0:3].transpose()/points_3d_selected[:,3]).transpose(),mask_bool


    def process(self,image):
        features = self.feature_detector.detect(image)
        if self.image_last is None:
            self.image_last = image.copy()
            return None, None, None,None
        feature_cur,feature_last = self.feature_tracking(image,self.image_last,features)
        R,t,feature3d,mask = self.motion_estimarion(feature_cur,feature_last)
        if False:
            image_show = image.copy()
            vu.draw_line(image_show,feature_cur[mask,:],feature_last[mask,:])
            cv2.imshow('im',image_show)
            cv2.waitKey()
        self.image_last = image.copy()
        return R,t,feature3d,feature_cur[mask,:]

    def reset(self):
        self.image_last = None






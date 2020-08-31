import cv2

def draw_line(img,feature_s,feature_t,color=(255,255,0)):
    
    diff = np.abs(feature_s -feature_t)
    ratio = np.arctan(diff[:,0]/(diff[:,1]+0.01))
    max_r = np.max(ratio)
    min_r = np.min(ratio)
    print(max_r,min_r)
    c = 255*(ratio-min_r)/(max_r-min_r)
    for i in range(feature_s.shape[0]):
        color = (int(c[i]),int(c[i]),int(255-c[i]))
        cv2.line(img,(int(feature_s[i,0]),int(feature_s[i,1])),(int(feature_t[i,0]),int(feature_t[i,1])),color,2)
def draw_feature(img,feature,color=(255,255,0)):
    for i in range(feature.shape[0]):
        cv2.circle(img,(int(feature[i,0]),int(feature[i,1])),3,color,-1)



import cv2
import numpy as np

def draw_triangle(img,features,triangle_ids,color=(255,255,0),fill=False):
    for i in range(0,triangle_ids.shape[0]):
        triangle_id = triangle_ids[i]
        triangle_points =  np.array(features[triangle_id],np.int32)
        pts = triangle_points.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,color)
        if fill:
            cv2.fillPoly(img,[pts],color)

def draw_feature_depth(img,feature2d,feature3d,color=(255,255,0)):
    if feature2d is not None:
        near = np.min(feature3d[:,2])
        far   = np.max(feature3d[:,2])
     
        for i in range(feature2d.shape[0]):
            pos_y_norm = (feature3d[i,2]-near)/(far-near)
            cv2.circle(img,(int(feature2d[i,0]),int(feature2d[i,1])),3,(255*pos_y_norm,0,255-255*pos_y_norm),-1)
            #cv2.circle(img,(int(feature[i,0]),int(feature[i,1])),3,color,-1)


def draw_feature(img,feature,color=(255,255,0)):
    if feature is not None:
        for i in range(feature.shape[0]):
            cv2.circle(img,(int(feature[i,0]),int(feature[i,1])),3,color,-1)
def draw_line(img,feature_s,feature_t,color=(255,255,0)):
    if feature_s is None:
        return
    diff = np.abs(feature_s -feature_t)
    ratio = np.arctan(diff[:,0]/(diff[:,1]+0.01))
    max_r = np.max(ratio)
    min_r = np.min(ratio)
    print(max_r,min_r)
    c = 255*(ratio-min_r)/(max_r-min_r)
    for i in range(feature_s.shape[0]):
        color = (int(c[i]),int(c[i]),int(255-c[i]))
        cv2.line(img,(int(feature_s[i,0]),int(feature_s[i,1])),(int(feature_t[i,0]),int(feature_t[i,1])),color,2)


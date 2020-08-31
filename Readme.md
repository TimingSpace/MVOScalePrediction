# Monocular Visual Odometry Scale Estimation
@author Xiangwei Wang

## General Idea
1. Learn to model the relation between pixel depth and the pixel coordinate (together with optical flow) with ground truth data for specific scene
2. Use the probability model to predict the depth of each feature
3. Use the depth to predict the scale


## Pre-expeiments
1. depth modelling


## Procedure
### Training 
#### Input
1. image sequences
2. ground truth motion scale(only scale is enough)

#### Training
1. Feature estimation and matching to formulate the (u,v,du,dv) and (u',v',-du,-dv)
2. associate that with the absolute scale (u,v,du,dv,s) and (u',v',-du,-dv,-s)
3. Use the data to model p(s|u,v,du,dv) by discrete model or ? 


### Prediction


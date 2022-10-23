# cxzxs
Deep-frame-interpolation-localization

# Environment Requirement
+ python>=3.7

+ pytorch>=1.8

# Usage

# Required Data
+ First,prepare the database [UCF&Danvis]


+ Second,use [PWC-Net](https://research.nvidia.com/publication/2018-06_pwc-net-cnns-optical-flow-using-pyramid-warping-and-cost-volume) to generate optical flow 

+ Third,use ConvGRU-Z to get predicted video frame. The predicted frame and the real frame are different to obtain the motion abnormity region.

# Train & Test
+ ConvGRU-Z:(Deep-frame -interpolation-localization/ConvGRU-Z/Train ConvGRU-Z)
+ EMBN:(Deep-frame -interpolation-localization/Moudle/run)

# Question
+ If you have any problems or find mistakes in this code, please contact with me: 
Qing Gu: 3334143570@qq.com 



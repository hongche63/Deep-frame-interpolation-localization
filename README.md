# cxzxs
Deep-frame-interpolation-localization

# Environment Requirement
+ python>=3.7

+ pytorch>=1.8

# Usage

# Required Data
+ First,prepare the database [UCF&Danvis]


+ Second,use PWC-Net to generate optical flow 
[PWC-Net:](https://research.nvidia.com/publication/2018-06_pwc-net-cnns-optical-flow-using-pyramid-warping-and-cost-volume).

+ Third,use ConvGRU-Z to get predicted video frame. The predicted frame and the real frame are different to obtain the motion abnormity region.

# Train & Test
+ ConvGRU-Z:(Deep-frame -interpolation-localization/ConvGRU-Z/Train ConvGRU-Z)
+ EMBN:(Deep-frame -interpolation-localization/Moudle/run)

# Question
+ If you have any problems or find mistakes in this code, please contact with me: 
Qing Gu: 3334143570@qq.com 



# Model
Completion of layer refinement graph convolutional network for recommendation(https://arxiv.org/abs/2207.11088)



# Question
+ If you have any problems or find mistakes in this code, please contact with me: 
Cheng Yang: yangchengyjs@163.com 

# cxzxs
Deep-frame-interpolation-localization

# Requirement
.python>=3.7

.pytorch>=1.8

# Usage

# Required Data

    .First,prepare the database [UCF&Danvis]

    .Second,use PWC-Net to generate optical flow 
    [PWC-Net](https://research.nvidia.com/publication/2018-06_pwc-net-cnns-optical-flow-using-pyramid-warping-and-cost-volume).
    [Spring-data-jpa 查询  复杂查询陆续完善中](http://www.cnblogs.com/sxdcgaq8080/p/7894828.html)
    
    .Third,use ConvGRU-Z to get predicted video frame. The predicted frame and the real frame are different to obtain the motion abnormity region.

# Train & Test
    . ConvGRU-Z:(Deep-frame -interpolation-localization/ConvGRU-Z/Train ConvGRU-Z)
    . EMBN:(Deep-frame -interpolation-localization/Moudle/run)

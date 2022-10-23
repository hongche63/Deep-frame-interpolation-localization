# cxzxs
Deep-frame-interpolation-localization

# Requirement
.python>=3.7
.pytorch>=1.8

# Usage

# Required Data

    .First,prepare the database [UCF&Danvis]

    .Second,use PWC-Net to generate optical flow (https://research.nvidia.com/publication/2018-06_pwc-net-cnns-optical-flow-using-pyramid-warping-and-cost-volume)

    .Third,use ConvGRU-Z to get predicted video frame .The predicted frame and the real frame are different to obtain the motion abnormity region.


# cxzxs
Deep-frame-interpolation-localization

# Environment Requirement
.python>=3.7

.pytorch>=1.8

# Usage

# Required Data

    + First,prepare the database [UCF&Danvis]

    + Second,use PWC-Net to generate optical flow 
    [PWC-Net:](https://research.nvidia.com/publication/2018-06_pwc-net-cnns-optical-flow-using-pyramid-warping-and-cost-volume).
    <http://www.gnu.org/licenses/>
    + Third,use ConvGRU-Z to get predicted video frame. The predicted frame and the real frame are different to obtain the motion abnormity region.

# Train & Test
    . ConvGRU-Z:(Deep-frame -interpolation-localization/ConvGRU-Z/Train ConvGRU-Z)
    . EMBN:(Deep-frame -interpolation-localization/Moudle/run)

# Question
+ If you have any problems or find mistakes in this code, please contact with me: 
Qing Gu: 3334143570@qq.com 
# License

Copyright (C) 2022, Cheng Yang (yangchengyjs@163.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.



# LRGCNCDA
LRGCNCDA, which effectively predicts the association between circRNA and disease, is a method based on layer-refined graph convolutional networks.



# Environment Requirement
+ torch == 1.7.1+cu110
+ numpy == 1.19.5
+ matplotlib == 3.5.1
+ dgl-cu110 == 0.5.3



# Model
Completion of layer refinement graph convolutional network for recommendation(https://arxiv.org/abs/2207.11088)



# Question
+ If you have any problems or find mistakes in this code, please contact with me: 
Cheng Yang: yangchengyjs@163.com 

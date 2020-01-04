# Side-Window-Filtering-Python
* Python implementation of CVPR 2019 Oral paper Side Window Filtering
* Python 實作 CVPR 2019 的論文 Side Window Filtering
  * Paper link: https://arxiv.org/pdf/1905.07177.pdf

# Usage 使用方式
* Download python file SideWindowFilter.py
* 下載 SideWindowFilter.py 檔案

'''python
import cv2
from SideWindowFilter import SideWindowFiltering_3d

img = cv2.imread('aiaceo.jpg')
swf_img = SideWindowFiltering_3d(swf_img, kernel=3, mode='mean')
'''

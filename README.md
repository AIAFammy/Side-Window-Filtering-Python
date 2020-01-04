# Side-Window-Filtering-Python
* Python implementation of [CVPR 2019 Oral paper Side Window Filtering](https://arxiv.org/pdf/1905.07177.pdf)
* Python 實作 [CVPR 2019 的論文 Side Window Filtering](https://arxiv.org/pdf/1905.07177.pdf)

# Usage 使用方式
* Download python file SideWindowFilter.py
* 下載 SideWindowFilter.py 檔案

```python
import cv2
from SideWindowFilter import SideWindowFiltering_3d

img = cv2.imread('aiaceo.jpg')
swf_img = SideWindowFiltering_3d(img, kernel=3, mode='mean')
```

* SWF_demo.ipynb demonstrate some examples
* SWF_demo.ipynb 實際示範了幾個使用方式

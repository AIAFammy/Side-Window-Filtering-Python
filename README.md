# Side-Window-Filtering-Python
* Python implementation of [CVPR 2019 Oral paper Side Window Filtering](https://arxiv.org/pdf/1905.07177.pdf)

# Usage
* Download python file SideWindowFilter.py

```python
import cv2
from SideWindowFilter import SideWindowFiltering_3d

img = cv2.imread('aiaceo.jpg')
swf_img = SideWindowFiltering_3d(img, kernel=3, mode='mean')
```

* SWF_demo.ipynb demonstrate some examples

# Requirment

```python
import cv2
import numpy
import numba
```

* Mostly use basic function to implement, version of packages might not be a big deal.
* Use numba just-in-time to complie python code, otherwise it takes extremely long time when execution. 

# DEMO

* Original image & Add salt noise image

![alt](images/origin&noise.png)

* 3x3 Mean Filtering by traditional method and SWF method

![alt](images/mean_3x3_iter1.png)
![alt](images/mean_3x3_iter10.png)
![alt](images/mean_3x3_iter20.png)
![alt](images/mean_3x3_iter50.png)

# Compare mean filter & median filter

* Median filter takes much more time than mean filter

![alt](images/execution_times.png)

* Results by apply (5, 5) kernel size

![alt](images/compare_1.png)
![alt](images/compare_2.png)
![alt](images/compare_3.png)
![alt](images/compare_10.png)
![alt](images/compare_30.png)




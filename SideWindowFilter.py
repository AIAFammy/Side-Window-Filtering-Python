import cv2
import numpy as np
import numba as nb

SWF_base = np.array([
    [[1, 1, 0],
     [1, 1, 0],
     [0, 0, 0]],
    [[0, 1, 1],
     [0, 1, 1],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 1, 1],
     [0, 1, 1]],
    [[0, 0, 0],
     [1, 1, 0],
     [1, 1, 0]],
    [[1, 1, 1],
     [1, 1, 1],
     [0, 0, 0]],
    [[0, 0, 0],
     [1, 1, 1],
     [1, 1, 1]],
    [[1, 1, 0],
     [1, 1, 0],
     [1, 1, 0]],
    [[0, 1, 1],
     [0, 1, 1],
     [0, 1, 1]],
], dtype=np.float32)

SWF_big = np.array([
    [[1, 1, 0],
     [1, 1, 0],
     [0, 0, 0]],
    [[0, 1, 1],
     [0, 1, 1],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 1, 1],
     [0, 1, 1]],
    [[0, 0, 0],
     [1, 1, 0],
     [1, 1, 0]],
    [[1, 1, 1],
     [1, 1, 1],
     [0, 0, 0]],
    [[0, 0, 0],
     [1, 1, 1],
     [1, 1, 1]],
    [[1, 1, 0],
     [1, 1, 0],
     [1, 1, 0]],
    [[0, 1, 1],
     [0, 1, 1],
     [0, 1, 1]],
    [[1, 1, 1],
     [1, 1, 0],
     [1, 0, 0]],
    [[1, 1, 1],
     [0, 1, 1],
     [0, 0, 1]],
    [[0, 0, 1],
     [0, 1, 1],
     [1, 1, 1]],
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]],
], dtype=np.float32)
    
@nb.jit(nopython=True)
def numba_computation(h, w, ori_flatten, each_flattens):
    min_dist = np.full(h * w, np.inf)
    min_idx = np.zeros(h * w, dtype=np.int32)
    for i in range(len(each_flattens)):
        dist = (each_flattens[i] - ori_flatten)**2
        for j in range(h * w):
            if dist[j] < min_dist[j]:
                min_dist[j] = dist[j]
                min_idx[j] = i

    dst = np.zeros_like(ori_flatten)
    for j in range(h * w):
        dst[j] = each_flattens[min_idx[j]][j]
    
    return dst
    
def SideWindowFiltering(img, kernel=3, mode='mean', use_big=False):
    if use_big:
        filters = np.array([cv2.resize(f, (kernel, kernel)) for f in SWF_big])
        filters[filters < 0.99] = 0.
    else:
        filters = np.array([cv2.resize(f, (kernel, kernel)) for f in SWF_base])
        filters[filters < 0.99] = 0.
    
    if mode == 'mean':
        filters = [f / np.sum(f) for f in filters]
        each_flattens = np.array([cv2.filter2D(img ,-1, filters[i]).reshape(-1) for i in range(len(filters))])
    elif mode == 'gaussian':
        k = kernel // 2
        x, y = np.mgrid[-k:k+1,-k:k+1]
        sigma = 0.3*((kernel-1)*0.5 - 1) + 0.8
        gaussian_kernel = np.exp(-((x**2+y**2)/(2*sigma**2)))
        filters = [np.multiply(f, gaussian_kernel) for f in filters]
        filters = [f / np.sum(f) for f in filters]
        each_flattens = np.array([cv2.filter2D(img ,-1, filters[i]).reshape(-1) for i in range(len(filters))])
    elif mode == 'median':
        kernel = filters.shape[-1]
        each_flattens = np.array([median_filter(img, filters[i], kernel).reshape(-1) for i in range(len(filters))])

    h, w = img.shape
    ori_flatten = img.reshape(-1)
    dst = numba_computation(h, w, ori_flatten, each_flattens)
    dst = dst.reshape(h, w)
    return dst

def SideWindowFiltering_3d(img, kernel=3, mode='mean', use_big=False):
    """
    Args:
        
        img: A rgb image
        
        kernel: one of integers 3, 5, 7, ...
                use kernel size (3, 3) or (5, 5) or (7, 7) ...
                
        mode: 'mean' or 'median'
              use mean filter or median filter
              
        use_big: default 'False' use 8 angle filters
                 'True' use 12 angle filters
                 
        
    """
    img = img.copy()
    
    dsts = [0, 0, 0]
    for i in range(3):
        dsts[i] = SideWindowFiltering(img[:,:,i], kernel, mode, use_big)
    
    return np.dstack((dsts[0], dsts[1], dsts[2]))

@nb.jit(nopython=True)
def numba_acceleration(img, mask, kernel, h, w, median_idx, dst):
    for i in range(0, h - kernel + 1):
        for j in range(0, w - kernel + 1):
            tmp_arr = []
            for a in range(kernel):
                for b in range(kernel):
                    if mask[a, b] == 1:
                        tmp_arr.append(img[i+a, j+b])
            tmp_arr.sort()
            dst[i, j] = tmp_arr[median_idx]
    return dst

def median_filter(img, mask, kernel=3):
    mask = mask.astype(np.int32)
    median_idx = int(len(np.where(mask > 0)[0]) / 2)
    dst = np.zeros_like(img)
    img = np.pad(img, kernel//2, 'edge')
    h, w = img.shape
    dst = numba_acceleration(img, mask, kernel, h, w, median_idx, dst)
    return dst
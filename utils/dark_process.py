import math

import numpy as np
import cv2


def get_max_preds(heatmaps):
    '''
    由预测的heatmaps中得到最大值所在的位置

    输入heatmaps：numpy.ndarray([batch_size, num_joints, height, width])
        或者     numpy.ndarray([num_joints, height, width])  

    输出位置preds: numpy.ndarray([batch_size, num_joints, 2])  
        或者       numpy.ndarray([num_joints, 2])  
    '''
    assert isinstance(heatmaps, np.ndarray), \
        'heatmaps should be numpy.ndarray'
        
    if (heatmaps.ndim == 4):       
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        height = heatmaps.shape[2]        
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))  
        idx = np.argmax(heatmaps_reshaped, 2)  #取heatmap上的最大值位置
        idx = idx.reshape((batch_size, num_joints, 1)) 
        preds = np.tile(idx, (1, 1, 2)).astype(np.float32) #[batch_size, num_joints, 2]复制第二维，便于存储(y,x)
        #np.argmax得到的是将(height,width)的heatmap展平后的坐标，需要将此坐标转换为(height,width)内的坐标(y,x)。
        preds[:, :, 0] = (preds[:, :, 0]) // height   
        preds[:, :, 1] = preds[:, :, 1] - preds[:, :, 0] * height
        return preds
    elif(heatmaps.ndim == 3):
        num_joints = heatmaps.shape[0]
        height = heatmaps.shape[1]
        heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 1)
        idx = idx.reshape((num_joints, 1))
        preds = np.tile(idx, (1, 2)).astype(np.float32)
        preds[:, 0] = (preds[:, 0]) // height
        preds[:, 1] = preds[:, 1] - preds[:, 0] * height
        return preds
    else:
        raise Exception('heatmaps should be 4-ndim or 3-ndim')
        


def gaussian_blur(hm, kernel):
    '''
    给预测的heatmaps做Distribution Modulation的过程(包含高斯滤波和transformation)

    输入heatmaps：numpy.ndarray([batch_size, num_joints, height, width])
        或者     numpy.ndarray([num_joints, height, width])  
        kernel:高斯滤波的核

    输出heatmaps：numpy.ndarray([batch_size, num_joints, height, width])
        或者     numpy.ndarray([num_joints, height, width])  
    '''
    if(hm.ndim == 4):
        border = (kernel - 1) // 2
        batch_size = hm.shape[0]
        num_joints = hm.shape[1]
        height = hm.shape[2]
        width = hm.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(hm[i,j])
                #两边取border，类似于卷积取padding
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border: -border, border: -border] = hm[i,j].copy()
                #论文中Distribution Modulation部分的高斯滤波
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                hm[i,j] = dr[border: -border, border: -border].copy()
                #论文中Distribution Modulation部分的transformation，与论文稍有不同，没有减去最小值的部分，因为最小值接近0，所以几乎没影响。
                hm[i,j] *= origin_max / np.max(hm[i,j])  
        return hm
    else:
        border = (kernel - 1) // 2
        num_joints = hm.shape[0]
        height = hm.shape[1]
        width = hm.shape[2]
        for j in range(num_joints):
            origin_max = np.max(hm[j])
            #两边取border，类似于卷积取padding
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[j].copy()
            #论文中Distribution Modulation部分的高斯滤波
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[j] = dr[border: -border, border: -border].copy()
            #论文中Distribution Modulation部分的transformation，与论文稍有不同，没有减去最小值的部分，因为最小值接近0，所以几乎没影响。
            hm[j] *= origin_max / np.max(hm[j])  
        return hm

def taylor(hm, coord):
    '''
    给预测的heatmaps和预测点m做Distribution-aware Maximum Re-localization的过程

    输入单张heatmaps：numpy.ndarray([height, width])
              coord: 预测点m

           输出coord: 修正后的预测点μ
      
    '''
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[1])
    py = int(coord[0])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        #在预测点m的附近用相邻点来近似取微分，以下均是针对m的微分
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1]) 
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]]) #论文中的(6)式，令x=m
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]]) #论文中的(8)式,令x=m,得到m的海森矩阵
        if dxx * dyy - dxy ** 2 != 0:  #保证海森矩阵有逆
            #以下是论文中的(9)式
            hessianinv = hessian.I 
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def dark_process(hm,blur_kernel):
    '''
    由预测的heatmaps中得到经过dark处理后的预测点位置preds（仅适用于一张heatmap一个点的情况）
    输出点preds在heatmap图上的位置如下:     (以heatmaps.ndim==3时第一张heatmap图上的pred为例)
    heatmaps[0][int(preds[0][0])][int(preds[0][1])]


    输入heatmaps：numpy.ndarray([batch_size, num_joints, height, width])
        或者     numpy.ndarray([num_joints, height, width])
    高斯滤波核的大小blur_kernel:int

    输出位置preds: numpy.ndarray([batch_size, num_joints, 2])  
        或者       numpy.ndarray([num_joints, 2])  
    '''

    if (hm.ndim == 4):
        coords = get_max_preds(hm)
        heatmap_height = hm.shape[2]
        heatmap_width = hm.shape[3]
        #Distribution Modulation部分
        hm = gaussian_blur(hm, blur_kernel)
        hm = np.maximum(hm, 1e-10)
        #Distribution-aware Maximum Re-localization部分
        hm = np.log(hm) #先转换为ln(hm)格式便于泰勒展开及后面对预测点m的修正
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n,p] = taylor(hm[n][p], coords[n][p])
        preds = coords.copy()
    else:
        coords = get_max_preds(hm)
        heatmap_height = hm.shape[1]
        heatmap_width = hm.shape[2]
        #Distribution Modulation部分
        hm = gaussian_blur(hm, blur_kernel)
        hm = np.maximum(hm, 1e-10)
        #Distribution-aware Maximum Re-localization部分
        hm = np.log(hm) #先转换为ln(hm)格式便于泰勒展开及后面对预测点m的修正
        for n in range(coords.shape[0]):
            coords[n] = taylor(hm[n], coords[n])
        preds = coords.copy()
    return preds



if __name__ == '__main__':
    a = np.random.rand(2,2,3,3)
    print(a)
    preds = dark_process(a,3)
    print(preds)
    print(a[0][0][int(preds[0][0][0])][int(preds[0][0][1])])

    b = np.random.rand(2,3,3)
    print(b)
    preds = dark_process(b,3)
    print(preds)
    print(b[0][int(preds[0][0])][int(preds[0][1])])


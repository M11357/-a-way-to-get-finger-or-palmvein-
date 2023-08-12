import cv2
import imageio
import numpy as np
import scipy.signal as signal
from time import process_time
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from matplotlib import font_manager
import image
from PIL import Image
import os
'''主曲率方法'''
def PC(img_path):
    img = cv2.imread(img_path, 0)   #确保图像只保留一个通道
    # clahe = cv2.createCLAHE(8, (10, 4))
    # img = clahe.apply(img)
    sig = np.array(img)
    feature = np.zeros([img.shape[1], img.shape[0]])    #初始化特征矩阵，矩阵大小等于图片大小
    for i in range(img.shape[1]):
        key = sig[:, i]   #对数组进行切片 1*128
        k = np.array(key)
        x = np.arange(0,len(k),1)
        y = np.array(k)
        z1 = np.polyfit(x, y, 15)      # 使用polyfit方法来进行拟合,并选择多项式，最小二乘法
        p1 = np.poly1d(z1)         # 使用poly1d方法获得多项式系数,按照阶数由高到低排列
        fx =p1(x)           # 求对应x的各项拟合函数值
        value = signal.argrelextrema(-fx, np.greater)   #求极值的纵坐标
        val = list(value[0])  #value[0]为对value的单行切片，方便后续操作
        val2 = np.array(val)
        e = len(val2)
        for j in range(e):
            feature[i][value[0][j]] = 1    #在特征矩阵中，将检测到的极值点处的值赋为1
    transposed_feature = np.transpose(feature)
    # cv2.imshow('fin', transposed_feature)
    # cv2.waitKey(0)
    return transposed_feature

def PC_180(img_path):
    img = cv2.imread(img_path, 0)   #确保图像只保留一个通道
    # clahe = cv2.createCLAHE(8, (10, 4))
    # img = clahe.apply(img)
    sig = np.array(img)
    feature = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img.shape[0]):
        key = sig[i, :]   #对数组进行切片 1*128
        k = np.array(key)
        x = np.arange(0,len(k),1)
        y = np.array(k)
        z1 = np.polyfit(x, y, 15)      # 使用polyfit方法来进行拟合,并选择多项式，最小二乘法
        p1 = np.poly1d(z1)         # 使用poly1d方法获得多项式系数,按照阶数由高到低排列
        fx =p1(x)           # 求对应x的各项拟合函数值
        # data = fx
        value = signal.argrelextrema(-fx, np.greater)   #求极值的纵坐标
        val = list(value[0])  #value[0]为对value的单行切片，方便后续操作
        val2 = np.array(val)
        e = len(val2)
        for j in range(e):
            feature[i][value[0][j]] = 1
    # cv2.imshow('fin', feature)
    # cv2.waitKey(0)
    return feature
def compare_matrices(f1, f2):   #融合两个矩阵，相同位置有输入两个矩阵一个为1则新矩阵该处数值为1
    new_matrix = np.zeros_like(f1)
    point = []
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            if f1[i][j] == 1 or f2[i][j] == 1:#可改变的逻辑运算符
               new_matrix[i][j] = 255
               # cv2.circle(new_matrix, (i, j), 0, (0, 255, 0), 0)
               # print('...',new_matrix[i][j])
               # p = np.array([i,j])
               # point.append(p)
               # print('pp',p)

    return new_matrix

# def read_path(file_pathname):
    #遍历该目录下的所有图片文件
    list_center=[]
    list_theta=[]
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename,0)
        ####change to gray
      #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # image_np1=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        f1 = PC(file_pathname+'/'+filename)
        f2 = PC_180(file_pathname+'/'+filename)
        new_matrix1 = compare_matrices(f1, f2)
        ou_shi_ju_li = []
        data = new_matrix1
        for i in range(len(data)):
            cenx = 64
            ceny = 64
            point = data[i]
            poix = point[0]
            poiy = point[1]
            EuclideanDistance = ((poix - cenx) ** 2 + (poiy - ceny) ** 2) ** 0.5
            ou_shi_ju_li.append(int(EuclideanDistance))
        # n95 = np.array(new_matrix1)
        image_np_1 = np.sort(ou_shi_ju_li[0:8])
        # image_np_2 =

        np.save(r'D:\finger vein\Schwerer Gustav\feature_str'+"/"+filename,image_np)
        #####save figure
        # cv2.imwrite(r'D:\finger vein\Schwerer Gustav\feature_str'+"/"+filename,image_np)
if __name__ == '__main__':
    # read_path(r"D:\finger vein\Schwerer Gustav\cccvb")
    img_path = r"D:\finger vein\Schwerer Gustav\88lun_wen\531.jpg"
    start_time = time.time()
    f1 = PC(img_path)
    f2 = PC_180(img_path)
    end_time = time.time()
    cost = end_time-start_time
    print('cost',cost)
    new_matrix1 = compare_matrices(f1, f2)
    cv2.imshow('iii',new_matrix1)
    # cv2.imwrite(r"C:\Users\user\Desktop\624wc.jpg",new_matrix1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

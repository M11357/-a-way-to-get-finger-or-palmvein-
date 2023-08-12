import cv2
import numpy as np
import scipy.signal as signal
from time import process_time
import time
from scipy import interpolate


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

        y = np.array(k)
        n = len(y)
        x = range(0, n)
        tck = interpolate.splrep(x, y, s=1280, k=5)
        x_new = np.linspace(min(x), max(x), img.shape[0])
        y_fit = interpolate.BSpline(*tck)(x_new)
        fx = y_fit


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
        y = np.array(k)
        n = len(y)
        x = range(0, n)
        tck = interpolate.splrep(x, y, s=1280, k=5)
        x_new = np.linspace(min(x), max(x), img.shape[1])
        y_fit = interpolate.BSpline(*tck)(x_new)

        fx = y_fit
        value = signal.argrelextrema(-fx, np.greater)
        val = list(value[0])  # value[0]为对value的单行切片，方便后续操作
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

if __name__ == '__main__':
    # read_path(r"D:\finger vein\Schwerer Gustav\cccvb")
    img_path = r"C:\Users\user\Desktop\zjz.png"
    f1 = PC(img_path)
    f2 = PC_180(img_path)
    new_matrix1 = compare_matrices(f1, f2)
    f3 = new_matrix1
    cv2.imshow('iii', f3)
    # cv2.imwrite(r"C:\Users\user\Desktop\724doublezjz.jpg",new_matrix1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

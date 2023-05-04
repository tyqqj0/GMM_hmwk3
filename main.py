# -*- CODING: UTF-8 -*-
# @time 2023/5/3 19:36
# @Author tyqqj
# @File main.py


import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt

import sys

# sys.path.append('O:/mylibt')
# sys.path.append('O:/')

import mylibt as myt

# import cv2


import cv2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import os


# 图片读取器
# 读取路径下的所有图片
def read_img(path):
    # 读取路径下的所有图片
    # 返回值：img_list(list), img_name(list)
    # img_list为图片数据列表，img_name为图片名称列表
    img_list = []
    img_name = []
    for filename in os.listdir(path):
        img_name.append(filename)
        img = plt.imread(path + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)
    return img_list, img_name


def GMM(aimg, n_colors=4):
    # 读取图像并将其转换为浮点数

    # 获取图像的形状
    height, width, channels = aimg.shape

    # 将图像数据重塑为二维数组
    image_2d = aimg.reshape(-1, channels)

    # 创建 GMM 对象
    # n_colors = 4  # 聚类数，也就是要分割成的区域数量
    gmm = GaussianMixture(n_components=n_colors)

    # 对图像像素进行聚类
    gmm.fit(image_2d)
    labels = gmm.predict(image_2d)

    # 使用每个聚类的平均颜色替换原始像素颜色
    segmented_image = np.zeros_like(image_2d)
    for i in range(n_colors):
        mean_color = gmm.means_[i]
        segmented_image[labels == i] = mean_color

    # 将图像数据重塑回原始形状
    segmented_image = segmented_image.reshape(height, width, channels)

    return segmented_image


def GMMs(imgs, n_colors=4):
    results = []
    for img in imgs:
        results.append(GMM(img, n_colors))
    return results


def test_n_n(imgs, img_name, n=[2, 3, 6]):
    n_len = len(n)
    img_len = len(imgs)
    fig, axes = plt.subplots(n_len + 2, img_len, figsize=(img_len * 2, (n_len + 1) * 2))

    # 显示原始图像
    for idx, img in enumerate(imgs):
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f'{img_name[idx]} - Original')
        axes[0, idx].axis('off')

    # 对于每个 n，显示 GMM 结果
    for i, clusters in enumerate(n):
        print(f'GMM with {clusters} clusters')
        results = GMMs(imgs, clusters)
        for j, result in enumerate(results):
            print(f'Image {j}')
            axes[i + 1, j].imshow(result)
            axes[i + 1, j].set_title(f'{img_name[j]} - GMM with {clusters} clusters')
            axes[i + 1, j].axis('off')

    # 计算并显示每个图像颜色曲线
    for idx, img in enumerate(imgs):
        # print(f'Image {idx}')
        temp_img = img.copy()
        # 计算图像颜色直方图
        colors, counts = myt.Draw.img.color_curve(temp_img)

        # 显示颜色曲线
        axes[n_len + 1, idx].stackplot(colors, counts, colors=['r', 'g', 'b'])
        axes[n_len + 1, idx].set_xlim([0, 255])
        # axes[n_len + 1, idx].set_ylim([0, 10000])
        axes[n_len + 1, idx].set_xlabel('Pixel Intensity')
        axes[n_len + 1, idx].set_ylabel('Frequency')
        axes[n_len + 1, idx].set_title(f'{img_name[idx]} - Color Histogram')

    plt.tight_layout()
    plt.show()


path = 'D:/Data/simple_img/'
img_list, img_name = read_img(path)
print(img_name)

test_n_n(img_list, img_name, n=[2, 3, 6])

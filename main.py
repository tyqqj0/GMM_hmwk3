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
    for num, img in enumerate(imgs):
        print(f'Image {num}')
        results.append(GMM(img, n_colors))
    return results


def test_n_n(imgs, img_name, n=[2, 3, 6]):
    n_len = len(n)
    img_len = len(imgs)
    fig, axes = plt.subplots(img_len, n_len + 2, figsize=((n_len + 1) * 3.6, img_len * 3.6))
    col_titles = ['Original'] + [f'GMM with {clusters} clusters' for clusters in n] + ['Color Histogram']
    for idx, title in enumerate(col_titles):
        plt.figtext(0.5 / (n_len + 2) + idx / (n_len + 2), 0.99, title, ha='center', va='top', fontsize=12,
                    fontweight='bold')  # 加粗：fontweight='bold'

    # 显示原始图像
    for idx, img in enumerate(imgs):
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f'image {idx}')
        axes[idx, 0].axis('off')

    # 对于每个 n，显示 GMM 结果
    for i, clusters in enumerate(n):
        print(f'GMM with {clusters} clusters')
        results = []
        # results = GMMs(imgs, clusters)
        for j, result in enumerate(results):
            # print(f'Image {j}')
            axes[j, i + 1].imshow(result)
            axes[j, i + 1].set_title(f'image {j}')
            axes[j, i + 1].axis('off')
    print('GMM complete')

    # 计算并显示每个图像颜色曲线
    for idx, img in enumerate(imgs):
        # print(f'Image {idx}')
        temp_img = img.copy()
        # 计算图像颜色直方图
        colors, counts = myt.Draw.img.color_curve(temp_img)

        # 显示颜色曲线
        axes[idx, n_len + 1].stackplot(colors, counts, colors=['r', 'g', 'b'])
        axes[idx, n_len + 1].set_xlim([0, 255])
        # axes[idx, n_len + 1].set_ylim([0, 10000])
        axes[idx, n_len + 1].set_xlabel('Pixel Intensity')
        axes[idx, n_len + 1].set_ylabel('Frequency')
        axes[idx, n_len + 1].set_title(f'image {idx}\n - Color Histogram')

    plt.tight_layout(pad=1.5)
    plt.show()


path = 'D:/Data/simple_img/'
img_list, img_name = read_img(path)
print(img_name)

test_n_n(img_list, img_name, n=[2, 3, 6])

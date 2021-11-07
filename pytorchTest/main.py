# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import pickle
import random
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    list1 = [1, 2, 3]
    for each in reversed(list1):
        print(each, end=';')


def fun_x():
    x = [5]
    y = 10

    def fun_y():
        x[0] *= x[0]
        nonlocal y
        y *= y
        return x[0], y

    return fun_y(), x[0], y


def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def test_single_star(*param):
    print("有%d个参数" % len(param))
    print("它们分别是：", param)


def test_double_star(**param):
    print("有%d个参数" % len(param))
    print("它们分别是：", param)


def showMaxFactor(num):
    count = num // 2
    while count > 1:
        if num % count == 0:
            print("%d最大公约数是：%d" % (num, count))
            break
        count -= 1
    else:
        print("%d是素数：" % num)


class CapStr(str):
    def __new__(cls, string):
        string = string.upper()
        return str.__new__(cls, string)


class NewInt(int):
    def __add__(self, other):
        return int.__sub__(self, other)

    def __sub__(self, other):
        return int.__add__(self, other)


class NewIntMy(int):
    def __add__(self, other):
        return int(self) + int(other)

    def __sub__(self, other):
        return int(self) - int(other)


class Rectangle:
    def __init__(self, width = 0, height = 0):
        self.width = width
        self.height = height

    def __setattr__(self, name, value):
        if name == 'square':
            self.width = value
            self.height = value
        else:
            super().__setattr__(name, value)

    def getArea(self):
        return self.width * self.height


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # 将多个参数，打包成元组或列表
    # test_single_star(7, 8, 9, 10)
    # # 将元组或列表解包
    # tuple_a = (1, 2, 3, 4, 5)
    # test_single_star(*tuple_a)
    #
    # # 将多个参数，打包成字典
    # test_double_star(a=5, b=7, c=9)
    # # 将字典解包
    # dict_test_double_star = {'one': 1, 'two': 2, 'three': 3}
    # test_double_star(**dict_test_double_star)
    #
    # list1 = list({1, 2, 3, 4, 5, 5, 5})
    #
    # set1 = set(list1)
    # set1.add(6)
    # set1.remove(1)
    #
    # set2 = frozenset(set1)
    # # for each in set2:
    # #     print(each, end=' ')
    #
    # a = NewInt(5)
    # b = NewInt(3)
    # print(a + b)
    # print(a - b)
    #
    # a_my = NewIntMy(5)
    # b_my = NewIntMy(3)
    # print(a_my + b_my)
    # print(a_my - b_my)
    #
    # r1 = Rectangle(4, 5)
    # print(r1.getArea())
    # r1.square = 10
    # print(r1.getArea())
    #
    # string = 'FishC'
    # it = iter(string)
    # print(next(it))

    # # TensorBoard展示结果图
    # write = SummaryWriter("logs")
    # image_path = "train/ants_image/5650366_e22b7e1065.jpg"
    # img_PIL = Image.open(image_path)
    # img_array = np.array(img_PIL)
    # write.add_image("test", img_array, 2, dataformats='HWC')
    # for i in range(100):
    #     write.add_scalar("y=2x", 2*i, i)
    # write.close()

    # # transform使用
    # img_path2 = "train/bees_image/16838648_415acd9e3f.jpg"
    # # PIL格式图片
    # img2 = Image.open(img_path2)
    # tensor_trans = transforms.ToTensor()
    # # 调用def __call__(self, pic)
    # tensor_img = tensor_trans(img2)
    # print(tensor_img)

    # # opencv读取为numpy.ndarray格式图片
    # cv_img = cv2.imread(img_path2)
    # print(cv_img)

    # # Normalize
    # writer = SummaryWriter("logs")
    # img_path2 = "train/bees_image/16838648_415acd9e3f.jpg"
    # img = Image.open(img_path2)
    # trans_totensor = transforms.ToTensor()
    # img_tensor = trans_totensor(img)
    # trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # img_norm = trans_norm(img_tensor)
    # print(img_tensor[0][0][0])
    # print(img_norm[0][0][0])
    # writer.add_image("Normalize", img_norm)
    # writer.close()

    # Resize
    writer = SummaryWriter("logs")
    img_path2 = "train/bees_image/16838648_415acd9e3f.jpg"
    img = Image.open(img_path2)
    print(img.size)
    trans_resize = transforms.Resize((800, 800))
    img_resize = trans_resize(img)
    trans_totensor = transforms.ToTensor()
    img_resize = trans_totensor(img_resize)
    writer.add_image("Resize", img_resize, 0)
    print(img_resize.size)












# See PyCharm help at https://www.jetbrains.com/help/pycharm/

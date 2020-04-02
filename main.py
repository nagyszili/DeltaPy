from math import e

import PIL
from matplotlib import image as mpimg
from pylab import *
from sympy import *
import numpy as np
import glob
from PIL import Image


def relu(x):
    if x <= 0:
        return 0
    else:
        return x


def relu_deriv(x):
    if x <= 0:
        return 0
    else:
        return 1


def sigmoid(x):
    return 1 / (1 + e ** (-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def offline_learning(x, d, f, gradf, lr, stop):
    # [i,n] = size(x)
    n = size(x, 0)
    w = randn(n, size(d, 1))
    epoch = 0
    while 1:
        v = x * w
        y = f(v)
        e = y - d
        g = x * (e * gradf(v))
        w = w - lr * g
        # E = sum(e)
        # map(lambda i: i**2,e)
        E = 0
        for i in e:
            E = E + i ** 2
        if stop(E, epoch):
            break
        epoch = epoch + 1
    return w, E


def load_data():
    X = []
    train_data = [[], []]
    test_data = [[], []]

    i = 0
    for filename in glob.glob('img/cars/*.jpg'):
        img = Image.open(filename)
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = img.convert('L')
        pixels = img.load()
        scaled = []
        for j in range(100):
            for k in range(100):
                pix = pixels[j, k] / 255
                # print(pix)
                scaled.append(pix)
        X.append(scaled)
        if i < 50:
            img.save('img/train_data/' + str(i) + '.jpg')
            train_data[0].append(img)
            train_data[1].append("car")
        elif i < 100:
            img.save('img/test_data/' + str(i) + '.jpg')
            test_data[0].append(img)
            test_data[1].append("car")
        else:
            break
        i = i + 1

    for filename in glob.glob('img/planes/*.jpg'):
        img = Image.open(filename)
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = img.convert('L')
        pixels = img.load()
        scaled = []
        for j in range(100):
            for k in range(100):
                pix = pixels[j, k] / 255
                # print(pix)
                scaled.append(pix)
        X.append(scaled)
        if i < 150:
            img.save('img/train_data/' + str(i) + '.jpg')
            train_data[0].append(img)
            train_data[1].append("plane")
        elif i < 200:
            img.save('img/test_data/' + str(i) + '.jpg')
            test_data[0].append(img)
            test_data[1].append("plane")
        else:
            break
        i = i + 1

    # for j in train_data[1]:
    #     print(j)

    return X,train_data,test_data


def main():
    lr = 0.01
    X,train_data, test_data = load_data()

    # offline_learning(X,tr)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
from collections import namedtuple
from Queue import PriorityQueue, Empty
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

WSHED = -1
UNLABELLED = 0
Pos = namedtuple('Pos', ['r', 'c'])
image = np.array(Image.open("fruits.png"))
markers = np.array(Image.open("markers.png")).astype(np.int)
queue = PriorityQueue()
time = dict(time=0)


def timestep():
    time["time"] += 1
    return time["time"]


def showmarkers(iteration=None, overlap=False):
    if overlap:
        plt.imshow(image)
        plt.imshow(markers, alpha=0.6)
    else:
        plt.imshow(markers)
    if iteration is None:
        plt.title("Result")
    else:
        plt.title("markers iter:{}".format(iteration))
    plt.axis("off")
    plt.show()


def diff(a, b):
    return np.max(np.abs(image[a.r, a.c, :].astype(np.int32) - image[b.r, b.c, :].astype(np.int32)))


def init():
    markers[0, :] = WSHED
    markers[-1, :] = WSHED
    markers[:, 0] = WSHED
    markers[:, -1] = WSHED

    for r in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            if markers[r, c] > 0:
                queue.put((0, timestep(), Pos(r, c)))


def dilate(pos):
    # 四邻域
    u = Pos(pos.r - 1, pos.c)
    d = Pos(pos.r + 1, pos.c)
    l = Pos(pos.r, pos.c - 1)
    r = Pos(pos.r, pos.c + 1)

    labels = [markers[pos]]

    if markers[pos] == WSHED or markers[pos] == UNLABELLED:
        raise Exception("markers[pos] == {}".format(markers[pos]))

    for neighbor in (u, d, l, r):
        try:
            label = markers[neighbor]
            if label > UNLABELLED:
                labels.append(label)
        except IndexError:
            pass

    length = len(set(labels))
    if length > 1:
        # 邻域有>0的不同标记
        markers[pos] = WSHED
        return

    # 满足扩张条件
    for neighbor in (u, d, l, r):
        try:
            if markers[neighbor] == UNLABELLED:
                markers[neighbor] = markers[pos]
                queue.put((diff(neighbor, pos), timestep(), neighbor))
        except IndexError:
            pass


def process():
    iteration = 0
    try:
        while not queue.empty():
            iteration += 1
            if iteration % 10000 == 0:
                showmarkers(iteration)

            _, _, pos = queue.get_nowait()

            dilate(pos)

    except Empty:
        pass


if __name__ == '__main__':
    init()
    process()
    showmarkers(overlap=True)

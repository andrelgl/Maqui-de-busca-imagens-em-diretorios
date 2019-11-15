from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import operator

ssim_min = 0.7
max_obj = 5
imgs = os.listdir('images')

class Model():
    def __init__(self, img, valor):
            self.img = img
            self.valor = valor

original = cv2.imread("jp_gates_original.png")
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

def compare_images(imgs):
    imgList = []
    for i in imgs:
        img = cv2.imread("images/"+i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        s = measure.compare_ssim(original, img)
        if(s > ssim_min):
            image = Model(img, s)
            imgList.append(image)
    
    fig = plt.figure('Comparação')
    fig.subplots_adjust(hspace=0.8, wspace=0.4)
    index = 1
    ax = fig.add_subplot(2, int(len(imgList)-1/2), index)
    plt.imshow(original, cmap=plt.cm.gray)
    plt.title("Original")
    plt.axis("off")
    imgList.sort(key=operator.attrgetter('valor'))

    for i in imgList:
        ax = fig.add_subplot(2, int(len(imgList)-1/2), int(index+1))
        plt.imshow(i.img, cmap=plt.cm.gray)
        plt.margins(0, 4)
        plt.title("SSIM: %.2f" % (i.valor))
        plt.axis("off")
        index += 1 
    plt.show() 

compare_images(imgs)
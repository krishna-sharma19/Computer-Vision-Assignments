import cv2
import numpy


class ImageLoader:

    def __init__(self, path):
        self.img = cv2.imread(path, 1)

    def loadAndDisplayImg(self, img=None):
        if img is None:
            img = self.img
        cv2.imshow('Display', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def addToEveryPixel(self, num):
        self.loadAndDisplayImg(self.img + num)

    def multiplyToEveryPixel(self, num):
        self.loadAndDisplayImg(self.img * num)

    def subtractFromEveryPixel(self, num):
        self.loadAndDisplayImg(self.img - num)

    def divideEveryPixelBy(self, num):
        img = self.img / num
        self.loadAndDisplayImg(img.astype('uint8'))

    def resize(self):
        self.loadAndDisplayImg(
            cv2.resize(self.img, (int(self.img.shape[1] / 2), int(
                self.img.shape[0] / 2))))


path = '/Users/admin/Documents/jokes/icon/icon8.jpg'
obj = ImageLoader(path)
obj.loadAndDisplayImg()
obj.addToEveryPixel(30)
obj.subtractFromEveryPixel(30)
obj.multiplyToEveryPixel(2)
obj.divideEveryPixelBy(2)
obj.resize()


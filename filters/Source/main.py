# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")  # Single input, single output
    print(sys.argv[
              0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, three outputs
    print(sys.argv[
              0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def Question1():
    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


def equalizeChannel(obj):
    newImage = []
    freq, pixels = np.histogram(obj, 256)
    # print(type(pixels))
    # print(pixels)
    pixels = np.asarray(pixels, dtype=int)

    # print(pixels)
    cumsum = np.cumsum(freq)
    N = len(obj) * len(obj[0])

    newPixelValues = (cumsum*1.0 / N)
    newPixelValues = [0] + list(np.asarray(newPixelValues))
    #print(newPixelValues)
    pixMap = {}
    # print(np.asarray(pixels).reshape((1,N)))
    # pixMap[np.asarray(pixels)] = newPixelValues
    i = 0
    for pixel in pixels:
        pixMap[pixel] = newPixelValues[i]
        i += 1

    for arr in obj:
        newArr = []
        for pixel in arr:
            # print(pixel)
            newArr.append(pixMap[pixel])

        newImage.append(newArr)
    newImage = np.asarray(newImage)
    return newImage


def histogram_equalization(img_in):
    # Write histogram equalization here
    #print(img_in)
    b, g, r = cv2.split(img_in)

    blue = equalizeChannel(b)
    green = equalizeChannel(g)
    red = equalizeChannel(r)
    img_out = cv2.merge((blue, green, red))
    #print("_____________________----------")
    #print(img_out)
    img_out = cv2.convertScaleAbs(img_out,alpha = 255.0)
    #print("-------------")
    #print(img_out)
    # Histogram equalization result

    return True, img_out


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
    # Write low pass filter here
    # Low pass filter result
    #print("inside lpf")
    # img_in = cv2.imread('input2.png', 0)
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_in)
    fshift = np.fft.fftshift(f)
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    fshift[:crow - 10, :] = 0
    fshift[crow + 10:, :] = 0
    fshift[:, :ccol - 10] = 0
    fshift[:, ccol + 10:] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    '''plt.subplot(131), plt.imshow(img_in, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])
    plt.show()'''
    img_out = img_back
    return True, img_out


def high_pass_filter(img_in):
    # Write high pass filter here
    # img_in = cv2.cvtColor(img_in,img_in,cv2.COLOR_BGR2GRAY)
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_in)
    fshift = np.fft.fftshift(f)
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    fshift[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    '''plt.subplot(131), plt.imshow(img_in, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.show()'''
    img_out = img_back  # High pass filter result
    #    print(img_out)

    return True, img_out


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)


def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def deconvolution(img_in):
    # Write deconvolution codes here
    imf = ft(img_in, (img_in.shape[0], img_in.shape[1]))  # make sure sizes match
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T
    gkf = ft(gk, (img_in.shape[0], img_in.shape[1]))  # so we can multiple easily
    imconvf = imf / gkf * 255
    img_out = ift(imconvf)

    # Deconvolution resultimg_in = cv2.imread('blurred2.exr',0)

    return True, img_out


def Question2():
    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    input_image2 = cv2.imread("blurred2.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):
    # Write laplacian pyramid blending codes here
    input_image1 = img_in1
    input_image2 = img_in2
    G = input_image1.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = input_image2.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        size = (gpA[i-1].shape[1],gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i],dstsize=size)
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        size = (gpB[i - 1].shape[1], gpB[i - 1].shape[0])
        GE = cv2.pyrUp(gpB[i],dstsize=size)
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connecting each half
    real = np.hstack((input_image1[:, :cols / 2], input_image2[:, cols / 2:]))

    img_out = ls_  # Blending result

    return True, img_out


def Question3():
    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()

# # import cv2
# # import numpy as np
# # import wordSegmentation as segmenation
# #
# # # Load Relevant Knn Files
# # classification = np.loadtxt("classifications.txt", np.float32)
# # flattened_image_features = np.loadtxt("flattenedImageFeatures.txt", np.float32)
# #
# # # Reshape
# # classification = classification.reshape((classification.size, 1))
# #
# # # KNN Training
# # kNearest = cv2.ml.KNearest_create()
# # print(cv2.ml.ROW_SAMPLE)
# # kNearest.train(flattened_image_features, cv2.ml.ROW_SAMPLE, classification)
# #
# # def recognition(charCandidate, charThreshold):
# #     # Find Contours
# #     cnts = cv2.findContours(charCandidate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     cnts = cnts[1]
# #
# #     boxes = []
# #     # Get all boxes
# #     for c in cnts:
# #         (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
# #         boxes.append(((boxX, boxY, boxX + boxW, boxY + boxH)))
# #
# #     # Order boxes from left to right
# #     boxes = sorted(boxes, key=lambda b: b[0])
# #
# #     # Initialize Thresholds
# #     resize_w = 20
# #     resize_h = 30
# #
# #     plate_number = ""
# #
# #     for (startX, startY, endX, endY) in boxes:
# #         current_char = charThreshold[startY:endY, startX:endX]
# #         # cv2.imshow("character", current_char)
# #         # cv2.waitKey(0)
# #
# #         cur_char_resize = cv2.resize(current_char, (resize_w, resize_h))
# #         cur_char_reshape = cur_char_resize.reshape((1, resize_w * resize_h))
# #         cur_feature = np.float32(cur_char_reshape)
# #
# #         _, result, _, _ = kNearest.findNearest(cur_feature, k=1)
# #
# #         plate_char = str(chr(int(result[0][0])))
# #         plate_number += plate_char
# #
# #     print("Detection Completed:")
# #     print(plate_number)
# #
# #     return plate_number
# #
# #
# # if __name__ == "__main__":
# #     plate1 = "tmp/172.png"
# #     plate2 = "plates/plate2.jpeg"
# #     plate3 = "plates/plate3.png"
# #     plate4 = "plates/plate4.png"
# #     plate5 = "plates/plate5.png"
# #     plate6 = "plates/plate6.png"
# #     plate7 = "plates/plate7.png"
# #     plate8 = "plates/plate8.png"
# #     plate9 = "plates/plate9.png"
# #     plate10 = "plates/plate10.png"
# #     plate11 = "plates/plate11.png"
# #     plate12 = "plates/plate12.png"
# #     plate13 = "plates/plate13.png"
# #     plate14 = "plates/plate14.png"
# #     plate15 = "plates/plate15.png"
# #     plate16 = "plates/plate16.png"
# #     plate17 = "plates/plate17.png"
# #     plate18 = "plates/plate18.png"
# #     plate19 = "plates/1.jpg"
# #
# #     charCandidate, charThreshold = segmenation.plate_segmentation(plate1)
# #     recognition(charCandidate, charThreshold)
#
# # import sys
# # import cv2
# # import numpy as np
# #
# # def find_if_close(cnt1,cnt2):
# #     row1,row2 = cnt1.shape[0],cnt2.shape[0]
# #     for i in range(row1):
# #         for j in range(row2):
# #             dist = np.linalg.norm(cnt1[i]-cnt2[j])
# #             if abs(dist) < 25:      # <-- threshold
# #                 return True
# #             elif i==row1-1 and j==row2-1:
# #                 return False
# #
# # img = cv2.imread('tmp/172.png')
# # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #
# # cv2.imshow('input', img)
# #
# #
# # ret,thresh = cv2.threshold(gray,127,255,0)
# # #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 20)
# # mser=False
# # if mser:
# #     mser = cv2.MSER_create()
# #     regions = mser.detectRegions(thresh)
# #     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
# #     contours = hulls
# # else:
# #     thresh = cv2.bitwise_not(thresh) # wants black bg
# #     im2,contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
# #
# # cv2.drawContours(img, contours, -1, (0,0,255), 1)
# # cv2.imshow('base contours', img)
# #
# #
# # LENGTH = len(contours)
# # status = np.zeros((LENGTH,1))
# #
# # print("Elements:", len(contours))
# # for i,cnt1 in enumerate(contours):
# #     x = i
# #     if i != LENGTH-1:
# #         for j,cnt2 in enumerate(contours[i+1:]):
# #             x = x+1
# #             dist = find_if_close(cnt1,cnt2)
# #             if dist == True:
# #                 val = min(status[i],status[x])
# #                 status[x] = status[i] = val
# #             else:
# #                 if status[x]==status[i]:
# #                     status[x] = i+1
# #
# # unified = []
# # maximum = int(status.max())+1
# # for i in range(maximum):
# #     pos = np.where(status==i)[0]
# #     if pos.size != 0:
# #         cont = np.vstack(contours[i] for i in pos)
# #         hull = cv2.convexHull(cont)
# #         unified.append(hull)
# #
# # cv2.drawContours(img,contours,-1,(0,0,255),1)
# # cv2.drawContours(img,unified,-1,(0,255,0),2)
# # #cv2.drawContours(thresh,unified,-1,255,-1)
# #
# # for c in unified:
# #     (x,y,w,h) = cv2.boundingRect(c)
# #     cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
# #
# # cv2.imshow('result', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # import cv2
# # import numpy as np
# #
# # # Reading the input image
# # img = cv2.imread('tmp/172.png', 0)
# #
# # # Taking a matrix of size 5 as the kernel
# # kernel = np.ones((1, 1), np.uint8)
# #
# # # The first parameter is the original image,
# # # kernel is the matrix with which image is
# # # convolved and third parameter is the number
# # # of iterations, which will determine how much
# # # you want to erode/dilate a given image.
# # img_erosion = cv2.erode(img, kernel, iterations=1)
# # img_dilation = cv2.dilate(img, kernel, iterations=1)
# #
# # cv2.imshow('Input', img)
# # cv2.imshow('Erosion', img_erosion)
# # cv2.imshow('Dilation', img_dilation)
# #
# # cv2.waitKey(0)
# # import cv2
# # image = cv2.imread("1.jpg")
# # image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
# # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# #
# # res,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) #threshold
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# #
# # dilated = cv2.dilate(thresh,kernel,iterations = 5)
# #
# # val,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# #
# # coord = []
# # for contour in contours:
# #       [x,y,w,h] = cv2.boundingRect(contour)
# #       coord.append((x,y,w,h))
# #
# # coord.sort(key=lambda tup:tup[0]) # if the image has only one sentence sort in one axis
# #
# # count = 0
# # for cor in coord:
# #         [x,y,w,h] = cor
# #         t = image[y:y+h,x:x+w,:]
# #         cv2.imwrite(str(count)+".png",t)
# # print("number of char in image:", count)
# import os
# import cv2
# import math
# import numpy as np
# import argparse
#
#
# def main():
#     """reads images from data/ and outputs the word-segmentation to out/"""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', help="path to input image")
#     args = parser.parse_args()
#
#     f = 'tmp/4.png'
#
#     print('[INFO] Segmenting words of image: %s' % f)
#
#     # read image, prepare it by resizing it to fixed height and converting it to grayscale
#     img = prepareImg(cv2.imread(f), 50)
#
#     # execute segmentation with given parameters
#     # -kernelSize: size of filter kernel (odd integer)
#     # -sigma: standard deviation of Gaussian function used for filter kernel
#     # -theta: approximated width/height ratio of words, filter function is distorted by this factor
#     # - minArea: ignore word candidates smaller than specified area
#     res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
#
#     # write output to 'out/inputFileName' directory
#     if not os.path.exists("out"):
#         os.mkdir('out')
#
#     # iterate over all segmented words
#     print('[INFO] Segmented into {} words, written to out/ folder'.format(len(res)))
#     for (j, w) in enumerate(res):
#         (wordBox, wordImg) = w
#         (x, y, w, h) = wordBox
#         outpath = "out/{}.png".format(j)
#         cv2.imwrite(outpath, wordImg)  # save word
#     # cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
#
#     # output summary image with bounding boxes around words
#     # cv2.imwrite(r'out/%s/summary.png'%f, img)
#     return
#
#
# def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
#     """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
#
#     Args:
#         img: grayscale uint8 image of the text-line to be segmented.
#         kernelSize: size of filter kernel, must be an odd integer.
#         sigma: standard deviation of Gaussian function used for filter kernel.
#         theta: approximated width/height ratio of words, filter function is distorted by this factor.
#         minArea: ignore word candidates smaller than specified area.
#
#     Returns:
#         List of tuples. Each tuple contains the bounding box and the image of the segmented word.
#     """
#
#     # apply filter kernel
#     kernel = createKernel(kernelSize, sigma, theta)
#     imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
#     (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     imgThres = 255 - imgThres
#
#     # find connected components. OpenCV: return type differs between OpenCV2 and 3
#     if cv2.__version__.startswith('3.'):
#         (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     else:
#         (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     # append components to result
#     res = []
#     for c in components:
#         # skip small word candidates
#         if cv2.contourArea(c) < minArea:
#             continue
#         # append bounding box and image of word to result list
#         currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
#         (x, y, w, h) = currBox
#         currImg = img[y:y + h, x:x + w]
#         res.append((currBox, currImg))
#
#     # return list of words, sorted by x-coordinate
#     return sorted(res, key=lambda entry: entry[0][0])
#
#
# def prepareImg(img, height):
#     """convert given image to grayscale image (if needed) and resize to desired height"""
#     assert img.ndim in (2, 3)
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     h = img.shape[0]
#     factor = height / h
#     return cv2.resize(img, dsize=None, fx=factor, fy=factor)
#
#
# def createKernel(kernelSize, sigma, theta):
#     """create anisotropic filter kernel according to given parameters"""
#     assert kernelSize % 2  # must be odd size
#     halfSize = kernelSize // 2
#
#     kernel = np.zeros([kernelSize, kernelSize])
#     sigmaX = sigma
#     sigmaY = sigma * theta
#
#     for i in range(kernelSize):
#         for j in range(kernelSize):
#             x = i - halfSize
#             y = j - halfSize
#
#             expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
#             xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
#             yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)
#
#             kernel[i, j] = (xTerm + yTerm) * expTerm
#
#     kernel = kernel / np.sum(kernel)
#     return kernel
#
#
# if __name__ == '__main__':
#     main()
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
# # read the image and define the stepSize and window size
# # (width,height)
# image = cv2.imread("7.png")  # your image path
# tmp = image  # for drawing a rectangle
# stepSize = 1
# (w_width, w_height) = (25, 25)  # window size
# for x in range(0, image.shape[1] - w_width, stepSize):
#     for y in range(0, image.shape[0] - w_height, stepSize):
#         window = image[x:x + w_width, y:y + w_height, :]
#
#         # classify content of the window with your classifier and
#         # determine if the window includes an object (cell) or not
#         # draw window on image
#         cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
#         plt.imshow(np.array(tmp).astype('uint8'))
# # show all windows
# plt.show()
#
# import sys
# import cv2
#
# if __name__ == '__main__':
#     # If image path and f/q is not passed as command
#     # line arguments, quit and display help message
#     # if len(sys.argv) < 3:
#     #     print(__doc__)
#     #     sys.exit(1)
#
#     # speed-up using multithreads
#     cv2.setUseOptimized(True)
#     cv2.setNumThreads(4)
#
#     # read image
#     im = cv2.imread('tmp/1.png')
#     # resize image
#     # newHeight = 200
#     # newWidth = int(im.shape[1] * 200 / im.shape[0])
#     # im = cv2.resize(im, (newWidth, newHeight))
#
#     # create Selective Search Segmentation Object using default parameters
#     ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
#
#     # set input image on which we will run segmentation
#     ss.setBaseImage(im)
#
#     # Switch to fast but low recall Selective Search method
#     # if (sys.argv[2] == 'f'):
#     ss.switchToSelectiveSearchFast()
#    # ss.switchToSelectiveSearchQuality()
#     rects = ss.process()
#     print('Total Number of Region Proposals: {}'.format(len(rects)))
#     increment = 5
#     print((im.shape[0]/6)**2)
#     while True:
#         imOut = im.copy()
#         for i, rect in enumerate(rects):
#             x, y, w, h = rect
#             if ((w*h)>50):#(i < numShowRects):
#                 print(w*h)
#                 cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
#                 cv2.imshow("Output", imOut)
#                 cv2.waitKey(0)
#             else:
#                 break
#
#         cv2.imshow("Output", imOut)
#         k = cv2.waitKey(0) & 0xFF
#         if k == 13:
#             break
#     cv2.destroyAllWindows()
from ocr.words import sort_words
import ocr.page
import cv2

img =cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# prepareImg(cv2.imread('tmp/prep2.jpg'), 500)
# cv2.imwrite('tmp/seg')
res = wordSegmentation(img,
                       kernelSize=int(config['segment']['kernelSize']),
                       sigma=int(config['segment']['sigma']),
                       theta=int(config['segment']['teta']),
                       minArea=int(config['segment']['minArea']))
print('Segmented into %d words' % len(res))
resd = sort_words(res)
print(resd)


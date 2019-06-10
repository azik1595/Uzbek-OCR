# import cv2
# import numpy as np
# import wordSegmentation as segmenation
#
# # Load Relevant Knn Files
# classification = np.loadtxt("classifications.txt", np.float32)
# flattened_image_features = np.loadtxt("flattenedImageFeatures.txt", np.float32)
#
# # Reshape
# classification = classification.reshape((classification.size, 1))
#
# # KNN Training
# kNearest = cv2.ml.KNearest_create()
# print(cv2.ml.ROW_SAMPLE)
# kNearest.train(flattened_image_features, cv2.ml.ROW_SAMPLE, classification)
#
# def recognition(charCandidate, charThreshold):
#     # Find Contours
#     cnts = cv2.findContours(charCandidate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[1]
#
#     boxes = []
#     # Get all boxes
#     for c in cnts:
#         (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
#         boxes.append(((boxX, boxY, boxX + boxW, boxY + boxH)))
#
#     # Order boxes from left to right
#     boxes = sorted(boxes, key=lambda b: b[0])
#
#     # Initialize Thresholds
#     resize_w = 20
#     resize_h = 30
#
#     plate_number = ""
#
#     for (startX, startY, endX, endY) in boxes:
#         current_char = charThreshold[startY:endY, startX:endX]
#         # cv2.imshow("character", current_char)
#         # cv2.waitKey(0)
#
#         cur_char_resize = cv2.resize(current_char, (resize_w, resize_h))
#         cur_char_reshape = cur_char_resize.reshape((1, resize_w * resize_h))
#         cur_feature = np.float32(cur_char_reshape)
#
#         _, result, _, _ = kNearest.findNearest(cur_feature, k=1)
#
#         plate_char = str(chr(int(result[0][0])))
#         plate_number += plate_char
#
#     print("Detection Completed:")
#     print(plate_number)
#
#     return plate_number
#
#
# if __name__ == "__main__":
#     plate1 = "tmp/172.png"
#     plate2 = "plates/plate2.jpeg"
#     plate3 = "plates/plate3.png"
#     plate4 = "plates/plate4.png"
#     plate5 = "plates/plate5.png"
#     plate6 = "plates/plate6.png"
#     plate7 = "plates/plate7.png"
#     plate8 = "plates/plate8.png"
#     plate9 = "plates/plate9.png"
#     plate10 = "plates/plate10.png"
#     plate11 = "plates/plate11.png"
#     plate12 = "plates/plate12.png"
#     plate13 = "plates/plate13.png"
#     plate14 = "plates/plate14.png"
#     plate15 = "plates/plate15.png"
#     plate16 = "plates/plate16.png"
#     plate17 = "plates/plate17.png"
#     plate18 = "plates/plate18.png"
#     plate19 = "plates/1.jpg"
#
#     charCandidate, charThreshold = segmenation.plate_segmentation(plate1)
#     recognition(charCandidate, charThreshold)

# import sys
# import cv2
# import numpy as np
#
# def find_if_close(cnt1,cnt2):
#     row1,row2 = cnt1.shape[0],cnt2.shape[0]
#     for i in range(row1):
#         for j in range(row2):
#             dist = np.linalg.norm(cnt1[i]-cnt2[j])
#             if abs(dist) < 25:      # <-- threshold
#                 return True
#             elif i==row1-1 and j==row2-1:
#                 return False
#
# img = cv2.imread('tmp/172.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# cv2.imshow('input', img)
#
#
# ret,thresh = cv2.threshold(gray,127,255,0)
# #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 20)
# mser=False
# if mser:
#     mser = cv2.MSER_create()
#     regions = mser.detectRegions(thresh)
#     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
#     contours = hulls
# else:
#     thresh = cv2.bitwise_not(thresh) # wants black bg
#     im2,contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
#
# cv2.drawContours(img, contours, -1, (0,0,255), 1)
# cv2.imshow('base contours', img)
#
#
# LENGTH = len(contours)
# status = np.zeros((LENGTH,1))
#
# print("Elements:", len(contours))
# for i,cnt1 in enumerate(contours):
#     x = i
#     if i != LENGTH-1:
#         for j,cnt2 in enumerate(contours[i+1:]):
#             x = x+1
#             dist = find_if_close(cnt1,cnt2)
#             if dist == True:
#                 val = min(status[i],status[x])
#                 status[x] = status[i] = val
#             else:
#                 if status[x]==status[i]:
#                     status[x] = i+1
#
# unified = []
# maximum = int(status.max())+1
# for i in range(maximum):
#     pos = np.where(status==i)[0]
#     if pos.size != 0:
#         cont = np.vstack(contours[i] for i in pos)
#         hull = cv2.convexHull(cont)
#         unified.append(hull)
#
# cv2.drawContours(img,contours,-1,(0,0,255),1)
# cv2.drawContours(img,unified,-1,(0,255,0),2)
# #cv2.drawContours(thresh,unified,-1,255,-1)
#
# for c in unified:
#     (x,y,w,h) = cv2.boundingRect(c)
#     cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
#
# cv2.imshow('result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np

# Reading the input image
img = cv2.imread('tmp/172.png', 0)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((1, 1), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)

cv2.waitKey(0)
import cv2
import numpy as np
import imutils
import os
import sys

def knn_training():
    images =['1.jpg']
    classification = []
    img_width = 20
    img_height = 30
    flattened_img = np.empty((0, img_width * img_height))
    for im in images:
     img = cv2.imread(im)
     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     ret, thresh_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
     area_threshold = 17.5
     thresh_img_clone = thresh_img.copy()
     im2, cnts, hierarchy_= cv2.findContours(thresh_img_clone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     contours = cnts
     for contour in contours:
        if cv2.contourArea(contour) >= area_threshold:
            boxX, boxY, boxW, boxH = cv2.boundingRect(contour)
            cv2.rectangle(img, (boxX, boxY), (boxX+boxW, boxY+boxH), (255, 0, 0), 2)
            current_char = thresh_img[boxY:boxY+boxH, boxX:boxX+boxW]
            resized_cur_char = cv2.resize(current_char, (img_width, img_height))
            reshape_cur_char = resized_cur_char.reshape((1, img_width*img_height))
            cv2.imshow("Training Image", img)
            cv2.imshow("Current Character", resized_cur_char)
            cv2.waitKey(1)
            try:
                current_class = ord(input("Введите символ: "))
            except:
                continue
            if str(current_class)!='13':
             print(current_class)
             flattened_img = np.append(flattened_img, reshape_cur_char, axis=0)
             classification.append(current_class)
    output_classification = np.array(classification, np.float32)
    output_classification = output_classification.reshape((output_classification.size, 1))
    np.savetxt("classifications.txt", output_classification)
    np.savetxt("flattenedImageFeatures.txt", flattened_img)

    print("Training Complete!")

    return

if __name__ == "__main__":
    knn_training()















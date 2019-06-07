import math
import cv2
import numpy as np # For general purpose array manipulation
import scipy.fftpack # For FFT2
from skimage import measure
from skimage import segmentation


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=20):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    #key = lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1]
    #return sorted(res, key=lambda entry: entry[0][1] )
    return  sorted(res,key= lambda l:l[0][1])


def prepareImg(img, height):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel


def plate_segmentation(plate_image):
    plate = cv2.imread(plate_image)
    gray_plate = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    # Transform plate to binary
    # ret, threshold = cv2.threshold(gray_plate, 90, 255, cv2.THRESH_BINARY_INV)

    threshold = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 125, 35)
    # cv2.imshow("Thresh Binary Inverse", threshold)
    # cv2.waitKey(0)

    # Find connecting regions of threshold regions
    connecting_regions = measure.label(threshold, neighbors=8, background=0)
    unique_regions = np.unique(connecting_regions)
    charCandidates = np.zeros(threshold.shape, dtype="uint8")
    count = 0

    # loop over the unique components
    for region in unique_regions:
        # if this is the background label, ignore it
        if region == 0:
            continue

        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        regionMask = np.zeros(threshold.shape, dtype="uint8")
        regionMask[connecting_regions == region] = 255
        cnts = cv2.findContours(regionMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow("label", regionMask)
        cv2.waitKey(0)

        cnts = cnts[1]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:
            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            cv2.drawContours(charCandidates, [hull], -1, 255, -1)
            count += 1

    # cv2.imshow("charCandidates", charCandidates)
    # cv2.waitKey(0)
    print("There are: " + str(len(np.unique(connecting_regions))) + " connecting region")
    print(str(count) + " regions are plate characters")

    if True:
        print("Using enhance algorithm")

        threshold = threshold_plate_enhance(plate_image)
        # cv2.imshow("Thresh Binary Inverse Enhance", threshold)
        # cv2.waitKey(0)

        # Find connecting regions of threshold regions
        connecting_regions = measure.label(threshold, neighbors=8, background=0)
        unique_regions = np.unique(connecting_regions)
        charCandidates = np.zeros(threshold.shape, dtype="uint8")
        count = 0

        # loop over the unique components
        for region in unique_regions:
            # if this is the background label, ignore it
            if region == 0:
                continue

            # otherwise, construct the label mask to display only connected components for the
            # current label, then find contours in the label mask
            regionMask = np.zeros(threshold.shape, dtype="uint8")
            regionMask[connecting_regions == region] = 255
            cnts = cv2.findContours(regionMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # cv2.imshow("label", regionMask)
            # cv2.waitKey(0)

            cnts = cnts[1]

            # ensure at least one contour was found in the mask
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)
                count += 1

        # cv2.imshow("charCandidates", charCandidates)
        # cv2.waitKey(0)
        print("There are: " + str(len(np.unique(connecting_regions))) + " connecting region")
        print(str(count) + " regions are plate characters")

    charThreshold = cv2.bitwise_and(threshold, threshold, mask=charCandidates)
    # cv2.imshow("charThreshold", charThreshold)
    # cv2.waitKey(0)

    return (charCandidates, charThreshold)

def binary_threshold(plate_image):
    pass

def threshold_plate_enhance(plate_image):
    img = cv2.imread(plate_image, 0)

    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255 * Ihmf, dtype="uint8")

    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < 90
    Ithresh = 255 * Ithresh.astype("uint8")

    return Ithresh

def scissor(plate_image):
    charCandidate, charThreshold = segmentation(plate_image)

    cnts =  cv2.findContours(charCandidate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]

    boxes = []
    chars = []

    for c in cnts:
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
        boxes.append(((boxX, boxY, boxX + boxW, boxY + boxH)))

    # Order boxes from left to right
    boxes = sorted(boxes, key=lambda b:b[0])

    for (startX, startY, endX, endY) in boxes:
        current_char = charThreshold[startY:endY, startX:endX]
        chars.append(current_char)
        cv2.imshow("character", current_char)
        cv2.waitKey(0)

    return chars




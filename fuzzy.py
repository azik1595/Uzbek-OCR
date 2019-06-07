import numpy as np
import matplotlib.image as mpimg
import skfuzzy as fuzz
import time


def fcm_images(img, c):
    # Reshape the image into a 2D array
    data = img.reshape(img.shape[0]*img.shape[1], img.shape[2]).T
    if data.shape[0] > 3:
        data = data[:3, :]
    # Fuzzy C-Means Clustering function call
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data, c, m=2, error=0.005, maxiter=1000, init=None)
    # Assign the maximum values of membership of each pixel to the 2D array
    labels = np.argmax(u, axis=0).reshape(img.shape[0], img.shape[1])
    # Create an image for each cluster
    return labels, cntr


def creat_image(labels, centers):
    img = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = centers[labels[i, j]]
    if(img.max() > 1):
        img /= 255
    mpimg.imsave('tmp/prep.jpg', img)
    return img


def compactness(img, labels, centers):
    WSS = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            WSS += np.sum(np.power(img[i, j]-centers[labels[i, j]], 2))
    return WSS/(labels.shape[0]*labels.shape[1])


def separation(labels, centers):
    BSS = 0
    cluster_size = np.zeros(centers.shape[0])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            cluster_size[labels[i, j]] += 1
    mean = np.mean(centers, axis=0)
    for k in range(centers.shape[0]):
            BSS += cluster_size[k]*np.sum(np.power(mean - centers[k], 2))
    return BSS/(labels.shape[0]*labels.shape[1])
def fuzzy_filter(impath,config):
    clusters = int(config['fuzzy']['n_clusters'])
    start_time = time.clock()
    img = mpimg.imread(impath)
    labels, centers = fcm_images(img, clusters)
    creat_image(labels, centers)
    elapsed_time = time.clock() - start_time
    print("elapsed time : {:0.3f} seconds".format(elapsed_time))

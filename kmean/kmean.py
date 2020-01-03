import cv2 as cv 
import numpy as np 
import math  
import matplotlib.pyplot as plt
import random
from sklearn.datasets.samples_generator import make_blobs
np.random.seed(11)


def labels_init(cluster_number):
    labels = list(range(1,cluster_number+1))
    return np.reshape(labels,(-1,1))

def norm2_calculator(element,center):
    norm2 = math.sqrt((element[0] - center[0])**2 + (element[1] - center[1])**2)
    return norm2

def add_row(array,number):
    number_convert = np.reshape(number,(-1,array.shape[1]));
    return np.append(array,number_convert,axis = 0);


def array_filter(array):
    return array[1:array.shape[0],:]

def centers_init(elements,cluster_number):
    return elements[np.random.choice(elements.shape[0], cluster_number, replace=False)]

def assign_label(elements,centers,labels):
    assigned_labels = np.zeros([1,1])
    for element in elements:
        norm2_arr = np.zeros([1,1])
        for center in centers:
            norm2 = norm2_calculator(element,center)
            norm2_arr = add_row(norm2_arr,norm2)
        norm2_arr = array_filter(norm2_arr) 
        chosen_label = labels[norm2_arr.argmin()]
        assigned_labels = add_row(assigned_labels,chosen_label)
    output = array_filter(assigned_labels);
    return output

def clustering(elements,labelOfElements,labels):
    clustered = []
    for label in labels:
        cluster = np.zeros([1,elements.shape[1]])
        for index in range(len(labelOfElements)):
            if  (labelOfElements[index] == label):
                cluster = add_row(cluster,elements[index])
        cluster = array_filter(cluster)
        clustered.append(cluster) 
    return clustered

def update_centers(clusters,labels):
    new_centers = np.zeros([1,elements.shape[1]])  
    for index in range(len(labels)):
        new_center = np.mean(clusters[index],axis=0)
        new_centers =  add_row(new_centers, new_center)
    new_centers = array_filter(new_centers)
    return new_centers

def isEqual(mtrx1,mtrx2):
    if mtrx1.shape == mtrx2.shape:
        return (mtrx1 == mtrx2).all()
    else:
        return False

def kmean(elements,cluster_number):
    #khoi tao so luong nhan
    labels = labels_init(cluster_number)
    centers  = centers_init(elements,cluster_number)
    while True:
        #gan nhan cho element
        labelOfElements = assign_label(elements,centers,labels)
        #phan cum theo nhan
        clusters = clustering(elements,labelOfElements,labels)
        #cap nhat centers
        new_centers = update_centers(clusters,labels)
        #kiem tra dieu kien lap
        if (isEqual(centers,new_centers)):
            break;
        else:
            centers = new_centers 
    return (clusters,centers)

def random_color():
    color = random.randint(1,8)
    if color == 1:
        return 'k'
    elif color == 2:
        return 'b'
    elif color == 3:
        return 'g'
    elif color == 4:
        return 'r'
    elif color == 5:
        return 'c'
    elif color == 6:
        return 'm'
    else:
        return 'y'

def random_marker():
    color = random.randint(1,8)
    if color == 1:
        return 'o'
    elif color == 2:
        return 'v'
    elif color == 3:
        return 'p'
    elif color == 4:
        return '+'
    elif color == 5:
        return 'D'
    elif color == 6:
        return 'h'
    else:
        return '*'

def display(clusters):
    for index in range(len(clusters)):
        X = clusters[index]
        prandom = random_color() + random_marker() 
        plt.plot(X[:, 0], X[:, 1], prandom, markersize = 4, alpha = .8)
    plt.axis('equal')
    plt.plot()
    plt.show()

def displayAll(clusters,centers):
    for index in range(len(clusters)):
        X = clusters[index]
        prandom = random_color() + random_marker() 
        plt.plot(X[:, 0], X[:, 1], prandom, markersize = 4, alpha = .8)

    X = centers
    prandom = random_color() + random_marker() 
    plt.plot(X[:, 0], X[:, 1], prandom, markersize = 10, alpha = .8)
    plt.axis('equal')
    plt.plot()
    plt.show()

X, y = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])

elements = X

# elements = np.reshape(X,(-1,2))
cluster_number = 5

#Bo doan code nay
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)



# elements = np.concatenate((X0, X1, X2), axis = 0) 
clusters,centers = kmean(elements,cluster_number)
# display(clusters)
displayAll(clusters,centers)

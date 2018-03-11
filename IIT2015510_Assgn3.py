import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import math
import matplotlib.pyplot as plt

learning_rate = 1e-3
epochs = 1000
threshold = 0.995

"""Read image as grayscale"""
river = cv2.imread('river.png',0)

test = []
train = []

label_train = [] 
label_test = []

k = 0

""" Dividing top half of image as training set and bottom half as testing set"""

for i in range(int(river.shape[0]/2)):
    for j in range(int(river.shape[1])):
        train.append(river[i][j])
        if ( river[i][j] >= 245 and j >= 130 and j <= 230):
            # print (i,j,river[i][j])
            label_train.append(1.0)
        else:
            label_train.append(0.0)

for i in range(int(river.shape[0]/2)):
    for j in range(int(river.shape[1])):
        test.append(river[i+int(river.shape[0]/2)][j])

        if ( river[i][j] >= 245 and j >= 130 and j <= 230):
            # print (i,j,river[i][j])
            label_test.append(1.0)
        else:
            label_test.append(0.0)

"""Defining weights, model and cost function"""

w0 = tf.Variable(np.random.randn(), dtype=tf.float32)
w1 = tf.Variable(np.random.randn(), dtype=tf.float32)

x1 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

# model = 1/(1 + tf.exp(w0 + w1*x1 + w2*x2 + w3*x3)) 
model = tf.nn.sigmoid(0 - w0 - w1*x1)

# cost = tf.reduce_sum(tf.log(model) + (1 - Y)*tf.log(1-model) ) / (-1 * len(label_train))
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = 0 - w0 - w1*x1,labels = label_train))

""" Training """
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(epochs):
    sess.run(optimizer,feed_dict={x1:train,Y:label_train})

""" Printing weights and Predicting values for test data"""

wa, wb = sess.run([w0, w1], {x1:train,Y:label_train})
print("w0: %s w1: %s "%(wa, wb))

# w1 = [0.018886536]

y_ = tf.nn.sigmoid(w0 + w1*x1)
pred_y = sess.run(y_, feed_dict={x1:test})

print (pred_y)

temp = [[0 for i in range(river.shape[1])] for j in range (int(river.shape[0]/2))]

k = 0

for i in range (int(river.shape[0]/2)):
    for j in range (river.shape[1]):
        if (pred_y[k] >= threshold):
            # print (pred_y[k], i, j)
            temp[i][j] = 255
        k = k + 1

img = np.array(temp)

im = Image.fromarray(img.astype('uint8')).convert('L')
im.save('result_image.png')
im.show()

riv = cv2.imread('river.png')
cv2.imshow('river',riv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# t = np.arange( 0., 256., step = 1.0)
# # print (t)

# plt.plot ( [1,2,3,4])
# plt.show()

# # fig, ax = plt.subplots()
# # ax.scatter([0,255], pred_y)
# # ax.plot([0, 1], [0, 1], 'k-', lw=3)
# # ax.set_xlabel('Measured')
# # ax.set_ylabel('Predicted')
# # plt.show()
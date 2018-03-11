import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import math

learning_rate = 1e-10
epochs = 10000

"""Read image as grayscale"""
river = cv2.imread('river.png',0)

test = []
train = []

x_coord_test = []
y_coord_test = []
x_coord_train = []
y_coord_train = []

label_train = []

k = 0

""" Dividing top half of image as training set and bottom half as testing set"""

for i in range(int(river.shape[0]/2)):
    for j in range(int(river.shape[1])):
        train.append(river[i][j])
        x_coord_train.append(i)
        y_coord_train.append(j)

        if ( river[i][j] >= 240 and i >=130 and j <=230):
            # print (i,j,river[i][j])
            label_train.append(1)
        else:
            label_train.append(0)

for i in range(int(river.shape[0]/2)):
    for j in range(int(river.shape[1])):
        test.append(river[i+int(river.shape[0]/2)][j])
        x_coord_test.append(i+int(river.shape[0]/2))
        y_coord_test.append(j)

"""Defining weights, model and cost function"""

w0 = tf.Variable([1.0], dtype=tf.float32)
w1 = tf.Variable([1.0], dtype=tf.float32)
w2 = tf.Variable([1.0], dtype=tf.float32)
w3 = tf.Variable([1.0], dtype=tf.float32)

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

model = 1/(1 + tf.exp(w0 + w1*x1 + w2*x2 + w3*x3)) 

cost = tf.reduce_sum(tf.log(model) + (1 - Y)*tf.log(1-model) ) / (-1 * len(label_train))

""" Training """
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(epochs):
    sess.run(optimizer,feed_dict={x1:x_coord_train,x2:y_coord_train,x3:train,Y:label_train})

""" Printing weights and Predicting values for test data"""

wa, wb, wc, wd = sess.run([w0, w1, w2, w3], {x1:x_coord_train,x2:y_coord_train,x3:train,Y:label_train})
print("w0: %s w1: %s w2: %s w3: %s "%(wa, wb, wc, wd))

y_ = 1/(1 + tf.exp(w0 + w1*x1 + w2*x2 + w3*x3)) 
pred_y = sess.run(y_, feed_dict={x1:x_coord_train,x2:y_coord_train,x3:train})



riv = cv2.imread('river.png')
cv2.imshow('river',riv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# a = [[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[255,255,255],[2,2,2],[2,2,2],[2,2,2]],[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[255,255,255],[2,2,2],[2,2,2],[2,2,2]],[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[255,255,255],[2,2,2],[2,2,2],[2,2,2]]]
# arr = np.array(a)
# print (a,arr)
# im = Image.fromarray(test.astype('uint8')).convert('L')
# im.save('result_image.png')
# im.show()
#coding=utf-8



import numpy as np
import tensorflow as tf


# 产生数据


n_observations = 100

xs = np.linspace(-3,3,n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5,0.5,n_observations)



#  定义输入 输出
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")


# 定义系数
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name='bias')


# 线性方程定义  y = wx + b
Y_pred = tf.add(tf.mul(X,W),b)


# 定义损失函数 
loss = tf.square(Y - Y_pred,name = "loss")


# 定义学习率
learn_rate = 0.01


# 梯度下降


optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)


n_samples = xs.shape[0]

#初始化变量
init = tf.initialize_all_variables()


with tf.Session() as sess:

    sess.run(init)

    # tensorflow 没有迭代这个事情，只有自己写迭代

    for i in range(50):
        total_loss = 0
        for x,y in zip(xs,ys):
            _,l = sess.run([optimizer,loss],feed_dict = {X:x,Y:y})
            total_loss += l
        if i % 5 == 0:
            print('Epoch {0} : {1}'.format(i,total_loss/n_samples))
    W,b = sess.run([W,b])

print W,b


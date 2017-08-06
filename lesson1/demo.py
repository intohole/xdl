#coding=utf-8



#coding=utf-8

import tensorflow as tf 




# 声明常量
a = tf.constant([5,5],tf.float32)
b = tf.constant([6,6],tf.float32)
# 声明操作
add = tf.add(a,b)


# 运行整个定义网络
with tf.Session() as session:
    print session.run(add)

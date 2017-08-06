#coding=utf-8



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time




# 读取tensorflow自带的工具家在MNIST手写数字集合
mnist = input_data.read_data_sets('./data/mnist', one_hot= True)


# 查看训练数据的维度
print mnist.train.images.shape

# 


# 建立输入层

batch_size = 128
X = tf.placeholder(tf.float32,[None,784],name = "X")
Y = tf.placeholder(tf.float32,[None,10],name = "Y")




n_hidden_1 = 256 
n_hidden_2 = 256
n_input = 784
n_classes = 10



weights = {
    'h1' : tf.Variable(tf.random_normal([n_input,n_hidden_1]),name = "W1"),
    "h2" : tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name = "W2"),
    "out" : tf.Variable(tf.random_normal([n_hidden_2,n_classes]),name = "W")
}

biases = {
    "b1" : tf.Variable(tf.random_normal([n_hidden_1]),name = "b1"),
    "b2" : tf.Variable(tf.random_normal([n_hidden_2]),name = "b2"),
    "out": tf.Variable(tf.random_normal([n_classes]),name = "bias")
}


def multilayer_perceptron(x, weights, biases):
    # 第一个隐层 
    layer_1 = tf.add(tf.matmul(x,weights["h1"]),biases["b1"], name="fc_1")
    layer_1_relu = tf.nn.relu(layer_1 , name = "relu_1")


    # 第二个隐层
    layer_2 = tf.add(tf.matmul(layer_1_relu,weights["h2"]),biases["b2"], name="fc_2")
    layer_2_relu = tf.nn.relu(layer_2 , name = "relu_2")

    # 全联接
    output_layer = tf.add(tf.matmul(layer_2_relu,weights["out"]),biases["out"], name="fc_3")
    return output_layer

pred = multilayer_perceptron(X,weights,biases)



learning_rate = 0.001 
loss_all = tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels =Y , name = "cross_entropy_loss")
loss = tf.reduce_mean(loss_all, name = "avg_loss")
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

init = tf.initialize_all_variables()

training_epochs = 15
batch_size = 128
display_step = 1

with tf.Session() as session:
    session.run(init)

    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x , batch_y = mnist.train.next_batch(batch_size)
            _, l = session.run([optimizer,loss],feed_dict = {X:batch_x, Y:batch_y})
            avg_loss += l / total_batch
        if epoch % display_step == 0:
            print "Epoch:{:4d} cost:{:.9f}".format(epoch + 1 , avg_loss)
    # 在测试集上评估
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))     

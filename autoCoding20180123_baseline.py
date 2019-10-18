#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:41:04 2018

@author: wangtingyan
"""
from __future__ import print_function
import pandas as pd
import jieba
#import codecs
import gensim
import numpy as np
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import classification_report

import time
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random



#=========================
#=========================
# part 1:
# data preprocessing
#
#=========================

#=========================
# part 1.1
# sentence segment: jieba
#=========================


#=========================
# part 1.2
# word embedding: word2vec
#=========================

## 读取已分词的语料库
with open('./corpus/corpus.jieba.txt', 'r') as f:
    corpus_data = [line.split() for line in f.readlines()]

# 词向量的训练
seq_len = 50 # embedding的维度

#model = gensim.models.Word2Vec(corpus_data,size = seq_len, window=5, min_count=3, workers=4)
#model.save('./model/medicalTermEmbeddingModel2')

# 停词表
#stop_f = codecs.open(u'./corpus/stopwords.txt', 'r', encoding='utf-8')
#stoplist = {}.fromkeys([line.strip() for line in stop_f])
stoplist = ['+',"/","-","(",")","（","）","[","]",",","，","的","为","、"," ",'→','4.','1、','2、','3、','4、','5、','6、','7、','8、','9、','10、','1.','2.','3.','4.','5.','6.','7.','8.','9.','10.']
# 对样本数据进行分词
def clean_str(string):
    string=[segment for segment in jieba.cut(string,cut_all=False) if segment not in stoplist]
    return string
def load_data_and_labels(filePath):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    CodeDf = pd.read_excel(filePath)
    x_text=[]
    for s  in CodeDf.diagnosis.values:
        #某一行为nan,为float，所以不读进来
        if not isinstance(s, float): 
            x_text.append(s.strip()) 
    
    # Split by words
    # clean_str() 将每条文本进行分词
    x_text = [clean_str(sent) for sent in x_text]
    
    # Generate labels
    y=[s.strip() for s in CodeDf.code.values]

    return [x_text,y]
# 对样本数据进行分词
[x,y]=load_data_and_labels("./data/dataSytStandard_DL.xlsx")
'''
#测试集
#[data_test,label_test]=load_data_and_labels("./data/testDL.xlsx")


# 加载词向量模型，对神经网络的输入进行word embedding
def dataEmbedding(x,y):
	model = gensim.models.Word2Vec.load('./model/medicalTermEmbeddingModel2')
	input_data = []
	input_score = []
	for item,sc in zip(x,y):
	    g = []
	    for i in item:
	        try:
	            z = list(model.wv[i])
	            g.append(z)
	        except:
	            pass
	    if len(g) != 0:
	        input_data.append(g)
	        input_score.append(sc)
	return input_data, input_score

          
# y: 从编码转化成one-hot向量
def label_encoding(y):
    y_new=[]
    for i in y:
        #if '*' in i:
        	#=i.split('+')
        # M 开头表示肿瘤编码
        if i[0] != 'M' and '*' not in i:
#            tmp_i =[] 
#        else:
            tmp_i= i[0:5]
            
        y_new.append(tmp_i)
    

    # y_new去重得到 y_new_uni
    y_new_uni = []
    for item in y_new:
        if item not in y_new_uni:
            y_new_uni.append(item)
    
    label_len=len(y_new_uni)
    
    int_encoded =[]
    
    for j in y_new:
        idx=y_new_uni.index(j) #j的索引，就是类标值
        int_encoded.append(idx)    
     
    
    onehot_encoded = list()
    for value in int_encoded:
        label_tmp = [0 for _ in range(label_len)]
        label_tmp[value] = 1
        onehot_encoded.append(label_tmp)
        
    label= onehot_encoded
        
    return label,label_len,y_new_uni

# x，y进行了word embedding
input_data,input_score=dataEmbedding(x,y)

#测试集
#input_data_test,input_score_test=dataEmbedding(data_test,label_test)

# y进行了one-hot编码，并得到类标的维度
input_score,label_len, y_new_uni=label_encoding(input_score)
#测试集
#input_score_test,label_len_test, y_new_uni_test=label_encoding(input_score_test)

print(label_len)
y_new_uni=np.array(y_new_uni)
y_new_uni1=pd.DataFrame(y_new_uni)
with open('label_code.csv','a') as f:
    y_new_uni1.to_csv(f, header = False)

# 随机分成 80%训练样本 和 20%测试样本
# x_train,x_test,y_train,y_test = train_test_split(input_data,input_score,test_size=0.10,random_state=33)

#x_train=input_data[:64516]
#y_train=input_score[:64516]
#
#x_test=input_data[64516:]
#y_test=input_score[64516:]

x_train=input_data[:23600]
y_train=input_score[:23600]

x_test=input_data[23600:]
y_test=input_score[23600:]

# shuffle数据
c = list(zip(x_train, y_train))
random.shuffle(c)
x_train, y_train = zip(*c)

print(len(x_train))



num = len(x_train)

x_train_len = [len(item) for item in x_train] #train每个样本的真实长度
x_test_len = [len(item) for item in x_test]
x_len= x_train_len + x_test_len
seq_max_len = max(x_len)

# 补成长度一样的数据
def transfer(x_inp,max_len):
    ret_list = []
    for item in x_inp:
        x = item
        x += [[0.]*seq_len for item in range(max_len - len(item))]
        ret_list.append(x)
    return ret_list

x_train = transfer(x_train,seq_max_len)
x_test = transfer(x_test,seq_max_len)


# minibatch训练： 生成批量数据
class GenerateData:
    def __init__(self,data_input,len_seq,labels):
        
        self.data = data_input
        self.labels = labels 
#        for item in labels:
#            if item == '-1':
#                self.labels.append([1.0,0.0])
#            else:
#                self.labels.append([0.0,1.0])
        
        self.seqlen = len_seq
        self.batch_id = 0
        
#    def next(self, batch_size):
#        if self.batch_id == len(self.data):
#            self.batch_id = 0
#        batch_data = (self.data[self.batch_id:min(self.batch_id +
#                                                  batch_size, len(self.data))])
#        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
#                                                  batch_size, len(self.data))])
#        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
#                                                  batch_size, len(self.data))])
#        self.batch_id = min(self.batch_id + batch_size, len(self.data))
#        return batch_data, batch_labels, batch_seqlen
    
    def next(self, batch_size):
        train_index=[i for i in range(len(x_train))] #生成一个index数列，用于每个epoch随机打乱train数据        
        if self.batch_id == len(self.data):
            self.batch_id = 0
            np.random.shuffle(train_index)#随机排列索引
        
        batch_index= train_index[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))] #通过随机排列后的索引来随机取数据
        batch_data = [(self.data[i]) for i in batch_index]
        batch_labels = [(self.labels[i]) for i in batch_index]#(self.labels[batch_index])
        batch_seqlen = [(self.seqlen[i]) for i in batch_index]#(self.seqlen[batch_index])
        

        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen



#=============
# Model: regression
#=============

t0=time.time()
#print(len(x_train))
# Parameters
lr_start=0.01 # base learning rate
lr_decay_rate=0.96 # each epoch： learning rate*lr_decay_rate
training_steps= 47200 # 3126steps*20epoches   #one epoch = (len(x_train)/batch_size) steps, total epoches:100 epoches
batch_size = 100
epoch_steps=len(x_train)/batch_size #
display_step = 10

# Network Parameters
n_input = seq_len #embedding的维度
#seq_max_len = 12
#seq_max_len = max(x_len) # Sequence max length

n_hidden = 180
#n_hidden_2 = 200
#n_hidden_3= 200
number_of_layers=2
n_classes = label_len

keep_rate=1 #dropout keep rate
regular_rate=0.0001
MOVING_AVERAGE_DECAY=0.99

#trainset = ADSequenceData(mark=1, feature_len=n_input, max_seq_len=seq_max_len)
#testset = ADSequenceData(mark=0, feature_len=n_input, max_seq_len=seq_max_len)

trainset = GenerateData(x_train,x_train_len,y_train)
testset = GenerateData(x_test,x_test_len,y_test)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, n_input])
y = tf.placeholder("float", [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])
drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")
# Define weights
weights = {
    #'out': tf.Variable(tf.random_normal([n_hidden, n_classes],stddev=0.0001))
    'out': tf.Variable(tf.truncated_normal([n_hidden, n_classes],mean=0.0,stddev=0.001))
}
biases = {
    #'out': tf.Variable(tf.zeros([n_classes]))
    'out': tf.Variable(tf.constant(0.001,shape=[n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):

    def lstm_cell():
        #return tf.contrib.rnn.GRUCell(n_hidden,kernel_initializer=tf.orthogonal_initializer())  
        return tf.contrib.rnn.GRUCell(n_hidden) 
        #return tf.contrib.rnn.LSTMCell(n_hidden,initializer=tf.orthogonal_initializer(),forget_bias=1.0) #orthogonal_initializer        
        #return tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden) # layer normalization
    
    # add dropout to the network  
    def lstm_cell_dropout():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(),input_keep_prob=drop_keep_rate)
    # add residual connect
    #def resWrapper_lstm_cell_dropout(): 
        #return tf.contrib.rnn.ResidualWrapper(lstm_cell_dropout())

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell_dropout() for _ in range(number_of_layers)])
    #stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell_dropout(),lstm_cell_dropout()])
    outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, sequence_length=seqlen, dtype=tf.float32,time_major=False)
    #outputs  [batch_size, n_step, n_hidden]
    # to get output of the last visit for each patient
    batch_size = tf.shape(outputs)[0] 
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index) 
    # outputs:[batch_size,n_hidden]  weights:[n_hidden, n_classes]
    linarg_event = tf.matmul(outputs, weights['out']) + biases['out'] #linarg_event: (batch_size,n_classes)
    return linarg_event    

pred = dynamicRNN(x, seqlen, weights, biases)
# 定义训练轮数及相关的滑动平均类
global_step = tf.Variable(0,trainable=False) 
#variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#variable_to_average =(tf.trainable_variables()+tf.moving_average_variables())
#variable_averages_op = variable_averages.apply(tf.trainable_variables())


# Define loss and optimizer
#vars   = tf.trainable_variables() 
#lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
#                if 'bias' not in v.name ]) * regular_rate #0.001 
#cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))+lossL2 
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) 

########
## changing learning rate in the training process
########
#global_step =tf.Variable(0,trainable=False)
# exponential_decay function to generate learning rate

learning_rate = tf.train.exponential_decay(lr_start, global_step, epoch_steps,lr_decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step= global_step)

# 反向传播更新参数和更新每一个参数的滑动平均
#with tf.control_dependencies([optimizer, variable_averages_op]):
#     train_op = tf.no_op(name='train')

# # 第二种方式
# train_op= tf.group(optimizer, variable_averages_op)


pred_label=tf.argmax(pred, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

# Initializing the variables
init = tf.global_variables_initializer()
loss_total=[]
accu_total=[]
test_loss_total=[]
test_accu_total=[]
# to save model
saver = tf.train.Saver()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)   
    for step in range(training_steps):
        batch_x, batch_y, batch_seqlen= trainset.next(batch_size)
        _, loss,acc=sess.run([optimizer,cost,accuracy], feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, drop_keep_rate: keep_rate},) 
        
        # 使用滑动平均
        #_, loss,acc=sess.run([train_op,cost,accuracy], feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, drop_keep_rate: keep_rate},)        
        loss_total.append(loss)
        accu_total.append(acc)
        
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = accuracy.eval({x : batch_x, y : batch_y, seqlen: batch_seqlen, drop_keep_rate:keep_rate})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)) 
        if step % epoch_steps== 0: # after completing an epoch, test the model
            test_data = testset.data
            test_label = testset.labels
            test_seqlen = testset.seqlen
            _,test_loss,test_acc=sess.run([optimizer,cost,accuracy], feed_dict={x: test_data, y: test_label, seqlen: test_seqlen,drop_keep_rate:1.0})
            test_loss_total.append(test_loss)
            test_accu_total.append(test_acc)
    print("Optimization Finished!")
    # save model
    model_dir="./model"
    model_name="autoCodingRnn"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    saver.save(sess, os.path.join(model_dir, model_name))
    print("Model is saved successfully")

    plt.plot(loss_total)
    plt.title('train loss')
    plt.show()

    plt.plot(test_loss_total,'r+')
    plt.title('test loss')
    plt.show()
    
    plt.plot(accu_total)
    plt.title('train accuracy')
    plt.show()

    plt.plot(test_accu_total,'r+')
    plt.title('test accuracy')
    plt.show()
    
    
    # Calculate test data accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
       
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen,drop_keep_rate:1.0}))

    # write the loss and accuracy to csv file
#    loss_total1=pd.DataFrame(loss_total)
#    with open('trainLoss.csv', 'a') as f:
#        loss_total1.to_csv(f, header=False)
#        
#    test_loss_total1=pd.DataFrame(test_loss_total)
#    with open('testLoss.csv', 'a') as f:
#        test_loss_total1.to_csv(f, header=False)
#        
#    accu_total1=pd.DataFrame(accu_total)
#    with open('trainACCU.csv','a') as f:
#        accu_total1.to_csv(f,header=False)
#        
#    test_accu_total1=pd.DataFrame(test_accu_total)
#    with open('testACCU.csv','a') as f:
#        test_accu_total1.to_csv(f,header=False)  
    
    #write the predicted label to an exitsting csv file
    test_pred_label=sess.run(pred_label, feed_dict={x: test_data, seqlen: test_seqlen,drop_keep_rate:1.0})
    test_pred_label1=pd.DataFrame(test_pred_label)
    with open('predictedLabel.csv', 'a') as f:
        test_pred_label1.to_csv(f, header=False)
    test_label1=pd.DataFrame(test_label)
    with open('testLabel.csv', 'a') as f2:
        test_label1.to_csv(f2, header=False)
    

    
run_time = time.time() - t0
print(label_len)
print('run time: %.3f s' % run_time)
'''
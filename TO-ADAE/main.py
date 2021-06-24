from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import csv
import math
import pywt
# import time
import os
from conf import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

n_input=64
step=16
train_rate=0.8
Threshol=8


classTrainEpoch = 2000
classTrainEpisode = 20

allTrainEpisode = 20
ADAETrainEpoch = 2500

Ker_Deep=128
i2=int(Ker_Deep*2)
i3=int(i2*2)

LR=0.0001
kernel = 3


classCostWeight=100
localCostWeight=100
Threshold_im=0
lambd = 5 

Fix_input = tf.placeholder(tf.float32, shape=[None,n_input,8], name='Fix_input')
UAV_input = tf.placeholder(tf.float32, shape=[None,n_input,8], name='UAV_input')

batch_size = tf.placeholder(tf.int32, name='b')   #batchsize
learning_rate= tf.placeholder(tf.float32, name='learning_rate')
Class_label = tf.placeholder(tf.int64, shape=[None, 2], name='Class_label')
keep_prob = tf.placeholder(tf.float32)
localWeight = tf.placeholder(tf.float32, name='localWeight')
classWeight = tf.placeholder(tf.float32, name='classWeight')
RTI_label = tf.placeholder(tf.float32, shape=[None,1225], name='RTI_label')
is_training = tf.placeholder(tf.bool, [], name='is_training')
lambd_t=tf.placeholder(tf.float32, name='lambd_t')


def setup():
    RW=17
    RL=17
    RX_positions=np.array([RW,RL])
    AP_positions=np.array([[0,0],[0,RL],[0,RL*2],[RW,0],[RW,RL*2],[RW*2,0],[RW*2,RL],[RW*2,RL*2]])
    b1=np.loadtxt("RTI_A1.csv",delimiter=",")
    b2=np.loadtxt("RTI_A2.csv",delimiter=",")
    return b1,b2,AP_positions,RX_positions

def take_data(file):    #batch n_input 8
    csvData = np.genfromtxt(file,delimiter=",")
    outData = []
    segment=int((len( csvData) - n_input) / step ) + 1
    for i in range(segment):
        nextData = csvData[int(step*i):int(step*i+n_input)]
        nextDataMean = np.mean(nextData,axis=0)
        nextDataVar = np.var(nextData,axis=0)
        nextData = np.clip(nextData, nextDataMean - Threshol*nextDataVar, nextDataMean + Threshol*nextDataVar)
        outData.append(nextData)
    outData = np.array(outData)
    outTrain, outTest=outData[0:int(len(outData)*train_rate)],outData[int(len(outData)*train_rate):len(outData)]
    return outData, outTrain, outTest

def random_batch(X_train,size):
    data_size=len(X_train)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    X_batch = X_train[shuffle_indices[0:int(size)]]
    return X_batch

def classifier(data,reuse_variables=None):
    data=tf.reshape(data, [batch_size,int(n_input*8)])
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        c_w1=tf.get_variable('c_w1', [int(n_input*8),256], initializer=tf.truncated_normal_initializer(stddev=0.03))
        c_w2=tf.get_variable('c_w2', [256,128], initializer=tf.truncated_normal_initializer(stddev=0.03))
        c_w3=tf.get_variable('c_w3', [128,64], initializer=tf.truncated_normal_initializer(stddev=0.03))
        c_w6=tf.get_variable('c_w6', [64,2], initializer=tf.truncated_normal_initializer(stddev=0.03))
        c_b1= tf.get_variable('c_b1', [256], initializer=tf.constant_initializer(0))
        c_b2= tf.get_variable('c_b2', [128], initializer=tf.constant_initializer(0))
        c_b3= tf.get_variable('c_b3', [64], initializer=tf.constant_initializer(0))
        c_b6= tf.get_variable('c_b6', [2], initializer=tf.constant_initializer(0))
        layer_1 = tf.nn.dropout(tf.nn.leaky_relu(tf.add(tf.matmul(data, c_w1), c_b1))  , keep_prob)
        layer_2 = tf.nn.dropout(tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, c_w2), c_b2))  , keep_prob)
        layer_3 = tf.nn.dropout(tf.nn.leaky_relu(tf.add(tf.matmul(layer_2, c_w3), c_b3))  , keep_prob)
        layer_6 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_3, c_w6),c_b6))
        layer_6 = tf.reshape(layer_6,[-1,2])
        return layer_6

def Accuracy(data,data_label):
    get_num=tf.reshape(data,[batch_size,2])
    get_num = tf.argmax(get_num, 1)
    get_num_label=tf.reshape(data_label,[batch_size,2])
    get_num_label = tf.argmax(get_num_label, 1)
    data_accuracy=tf.dtypes.cast(tf.reduce_sum((get_num-get_num_label)**2),tf.int32)
    data_accuracy=(batch_size-data_accuracy)/batch_size

    PASize = tf.cast(batch_size/2,dtype=tf.int32)
    ScenarioSize = tf.cast(batch_size/6,dtype=tf.int32)

    presentLabel = get_num_label[0:ScenarioSize]
    abesentLabel = get_num_label[PASize:(PASize+ScenarioSize)]

    presentAccuracy1 = get_num[0 : ScenarioSize]
    absentAccuracy1 = get_num[PASize:(PASize+ScenarioSize)]

    presentAccuracy1 = tf.dtypes.cast(tf.reduce_sum((presentAccuracy1 - presentLabel)**2),tf.int32)
    presentAccuracy1 = presentAccuracy1 / ScenarioSize


    absentAccuracy1 = tf.dtypes.cast(tf.reduce_sum((absentAccuracy1 - abesentLabel)**2),tf.int32)
    absentAccuracy1 = absentAccuracy1 / ScenarioSize

    presentAccuracy2 = get_num[ScenarioSize : (ScenarioSize+ScenarioSize)]
    absentAccuracy2 = get_num[(PASize+ScenarioSize):(PASize+ScenarioSize+ScenarioSize)]


    presentAccuracy2 = tf.dtypes.cast(tf.reduce_sum((presentAccuracy2 - presentLabel)**2),tf.int32)
    presentAccuracy2 = presentAccuracy2 / ScenarioSize

    absentAccuracy2 = tf.dtypes.cast(tf.reduce_sum((absentAccuracy2 - abesentLabel)**2),tf.int32)
    absentAccuracy2 = absentAccuracy2 / ScenarioSize

    presentAccuracy3 = get_num[(ScenarioSize+ScenarioSize):(ScenarioSize+ScenarioSize+ScenarioSize)]
    absentAccuracy3 = get_num[(PASize+ScenarioSize+ScenarioSize):(PASize+ScenarioSize+ScenarioSize+ScenarioSize)]


    presentAccuracy3 = tf.dtypes.cast(tf.reduce_sum((presentAccuracy3 - presentLabel)**2),tf.int32)
    presentAccuracy3 =  presentAccuracy3 / ScenarioSize

    absentAccuracy3 = tf.dtypes.cast(tf.reduce_sum((absentAccuracy3 - abesentLabel)**2),tf.int32)
    absentAccuracy3 =  absentAccuracy3 / ScenarioSize

    return data_accuracy, presentAccuracy1, absentAccuracy1, presentAccuracy2, absentAccuracy2, presentAccuracy3, absentAccuracy3

def train_classify():
    check = 0
    rec=[]

    Test_batch=np.concatenate([Fix_s1_test,Fix_s3_test,Fix_s2_test,
        Fix_e1_test,Fix_e3_test,Fix_e2_test],axis=0)
    batch_test=len(Test_batch)

    for rang in range(classTrainEpoch):
        batch_F=np.concatenate([random_batch(Fix_s1_train,int(group_train/6)),random_batch(Fix_s3_train,int(group_train/6)),random_batch(Fix_s2_train,int(group_train/6)),
            random_batch(Fix_e1_train,int(group_train/6)),random_batch(Fix_e3_train,int(group_train/6)),random_batch(Fix_e2_train,int(group_train/6))],axis=0)

        batch=len(batch_F)
        _= sess.run(c_trainer,feed_dict={batch_size:batch, learning_rate:LR, Class_label:Class_train_label,Fix_input: batch_F,keep_prob:0.5})

        if rang %1==0:
            ac=sess.run(FixAcc, feed_dict={batch_size:batch_test, learning_rate:LR, Class_label:Fix_test_label, Fix_input: Test_batch, keep_prob:1})
            rec.append(ac)
            # print(ac)
            if ac ==1 :
                check += 1
                if check == 5:
                    break
            else:
                check = 0
    return rec

##################################################################################

def angle_loss(data):
    labelA=AP_positions[5]
    if len(data[:,])==1:
        pass
    else:
        data=np.array([np.mean(data[0]),np.mean(data[1])])
    data=data.reshape((1,2))
    a=np.linalg.norm(data-labelA)
    b=np.linalg.norm(RX_positions-data)
    c=np.linalg.norm(RX_positions-labelA)
    cos=(a*a-b*b-c*c)/((-2*b*c)+1e-12)
    if cos>1:
        cos=1
    elif cos<-1:
        cos=-1
    loss=math.degrees(math.acos(cos))
    local_loss=np.linalg.norm(data-np.array([27,7]))
    return loss,local_loss
def bn(x, is_training, scope):
    return tf.layers.batch_normalization(x,axis=-1,momentum=0.99,epsilon=1e-12,training=is_training,name=scope)
def deconv2d(x, W, output_shape):
    return tf.nn.conv1d_transpose(x, W, output_shape, strides =  2, padding = 'SAME')

def DAE(x_origin, is_training, reuse_variables=None):
    with tf.variable_scope("Generator", reuse=reuse_variables)as scope:
        # x_origin = tf.reshape(x_origin, [batch_size,n_input,1])     #b*1,72,1
        newBatchSize = 8 * batch_size
        x_origin = tf.reshape(tf.transpose(x_origin,[0,2,1]), [newBatchSize,n_input,1])
        g_w1 = tf.get_variable('g_w1', [kernel, 1,Ker_Deep], initializer=tf.truncated_normal_initializer(stddev=0.03))
        g_b1 = tf.get_variable('g_b1', [Ker_Deep], initializer=tf.constant_initializer(0))
        h_e_conv1=bn(tf.nn.conv1d(x_origin, filters=g_w1, stride=2, padding='SAME')+ g_b1,is_training=is_training, scope='g_bn1')
        h_e_conv1 = tf.nn.leaky_relu(h_e_conv1)

        g_w2 = tf.get_variable('g_w2', [kernel, Ker_Deep, i2], initializer=tf.truncated_normal_initializer(stddev=0.03))
        g_b2 = tf.get_variable('g_b2', [i2], initializer=tf.constant_initializer(0))
        h_e_conv2 =tf.nn.leaky_relu(bn(tf.nn.conv1d(h_e_conv1, filters=g_w2, stride=2, padding='SAME')+ g_b2,is_training=is_training, scope='g_bn2'))
        
        g_w5 = tf.get_variable('g_w5', [kernel, Ker_Deep, i2], initializer=tf.truncated_normal_initializer(stddev=0.03))
        g_b5 = tf.get_variable('g_b5', [Ker_Deep], initializer=tf.constant_initializer(0))
        h_d_conv2 = tf.nn.leaky_relu(bn(tf.add(deconv2d(h_e_conv2, g_w5, [newBatchSize,int(n_input/2),Ker_Deep]),g_b5),is_training=is_training, scope='g_bn5'))  #b*20)

        g_w6 = tf.get_variable('g_w6', [kernel, 1, Ker_Deep], initializer=tf.truncated_normal_initializer(stddev=0.03))
        g_b6 = tf.get_variable('g_b6', [1], initializer=tf.constant_initializer(0))
        h_d_conv3 = tf.add(deconv2d(h_d_conv2, g_w6, [newBatchSize,n_input,1]),g_b6)  #b*40

        h_d_conv3 = tf.transpose(tf.reshape(h_d_conv3, [batch_size,8,n_input]),[0,2,1])
        return h_d_conv3

def discriminator(x_origin, is_training, reuse_variables=None):
    with tf.variable_scope("Discriminator", reuse=reuse_variables)as scope:
        newBatchSize = 8 * batch_size
        x_origin = tf.reshape(tf.transpose(x_origin,[0,2,1]), [-1,n_input,1])
        # x_origin = tf.reshape(x_origin, [-1,n_input,1])
        d_w1 = tf.get_variable('d_w1', [kernel, 1, Ker_Deep], initializer=tf.truncated_normal_initializer(stddev=0.03))
        d_b1 = tf.get_variable('d_b1', [Ker_Deep], initializer=tf.constant_initializer(0))
        d1 = tf.nn.leaky_relu( 
            bn(tf.nn.conv1d(x_origin, filters=d_w1, stride=2, padding='SAME') + d_b1,is_training=is_training, scope='d_bn1'))

        d_w2 = tf.get_variable('d_w2', [kernel, Ker_Deep, i2], initializer=tf.truncated_normal_initializer(stddev=0.03))
        d_b2 = tf.get_variable('d_b2', [i2], initializer=tf.constant_initializer(0))
        d2 = tf.nn.leaky_relu( 
            bn(tf.nn.conv1d(d1      , filters=d_w2, stride=2, padding='SAME') + d_b2,is_training=is_training, scope='d_bn2'))
        
        full_len=512
        d4 = tf.reshape(d2, [newBatchSize, int(i2*n_input/4)])   
        d_w4 = tf.get_variable('d_w4', [int(i2*n_input/4), full_len], initializer=tf.truncated_normal_initializer(stddev=0.03))
        d_b4 = tf.get_variable('d_b4', [full_len], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d4, d_w4) + d_b4 
        d4 = tf.nn.leaky_relu(bn(d4,is_training=is_training, scope='d_bn4'))

        d6=tf.reshape(d4, [-1, full_len])
        d_w6 = tf.get_variable('d_w6', [full_len, 1], initializer=tf.truncated_normal_initializer(stddev=0.03))
        d_b6 = tf.get_variable('d_b6', [1], initializer=tf.constant_initializer(0))
        d6 = tf.matmul(d6, d_w6) + d_b6

        d6=tf.reshape(d6,[-1,1])
        return d6

def normalize_data(data,S_size):
    data_max = tf.reshape( tf.reduce_max(data,axis=1),[S_size,1])
    data_min = tf.reshape( tf.reduce_min(data,axis=1),[S_size,1])
    return (data-data_min) / ((data_max - data_min)+1e-4)

def setup_RTI():
    true=np.array([1,1,1, 1.5, 1, 3, 1.5, 1])
    map_true=np.dot(true,b1)
    map_true[map_true < Threshold_im*np.max(map_true)] = 0
    data_max = np.max(map_true,axis=0)
    data_min = np.min(map_true,axis=0)
    map_true=(map_true-data_min) / ((data_max - data_min)+1e-4)
    map_true=map_true.reshape(1,1225)
    walk_label=np.ones([100,1225])*map_true

    map_true2=np.dot(true,b2)
    map_true2[map_true2 < Threshold_im*np.max(map_true2)] = 0
    data_max = np.max(map_true2,axis=0)
    data_min = np.min(map_true2,axis=0)
    map_true2=(map_true2-data_min) / ((data_max - data_min)+1e-4)
    map_true2=map_true2.reshape(1,1225)
    walk_label2=np.ones([100,1225])*map_true2

    return walk_label,walk_label2

def RTI(data, batch_size_new): # batch_size, n_input, 8
    S13_size=tf.cast(batch_size_new*2/3,dtype=tf.int32)
    S2_size=tf.cast(batch_size_new*1/3,dtype=tf.int32)
    RTI_label_13 = tf.slice(RTI_label, [0, 0], [S13_size, 1225])
    RTI_label_2  = tf.slice(RTI_label, [S13_size, 0], [S2_size, 1225])

    # RSS=tf.reshape(data,[batch_size_new,8,n_input])
    RSS = tf.reshape(tf.transpose(data,[0,2,1]), [batch_size_new, 8, n_input])
    RSS=tf.reshape(tf.math.reduce_std(RSS, axis=2),[batch_size_new,8])

    RSS_13 = tf.slice(RSS, [0, 0], [S13_size, 8])
    b_tensor = tf.cast(b1,dtype=tf.float32)
    map_b1=normalize_data(tf.matmul(RSS_13,b_tensor),S13_size)
    I=tf.reshape(tf.reduce_max(map_b1,axis=1),[S13_size,1])
    greater = tf.greater(map_b1, Threshold_im*I)
    map_b1=tf.where(greater,map_b1,tf.subtract(map_b1,map_b1))
    loss1 =tf.reduce_mean(tf.pow(RTI_label_13 - map_b1, 2))

    RSS_2 = tf.slice(RSS, [S13_size, 0], [S2_size, 8])
    b_tensor = tf.cast(b2,dtype=tf.float32)
    map_b2=normalize_data(tf.matmul(RSS_2,b_tensor),S2_size)
    I=tf.reshape(tf.reduce_max(map_b2,axis=1),[S2_size,1])
    greater = tf.greater(map_b2, Threshold_im*I)
    map_b2=tf.where(greater,map_b2,tf.subtract(map_b2,map_b2))
    loss2 =tf.reduce_mean(tf.pow(RTI_label_2 - map_b2, 2))

    return loss1,map_b1,loss2,map_b2

def train_GAN(head,file):
    check = 0
    prec1=np.array([0.5])
    erec1=np.array([0.5])

    prec2=np.array([0.5])
    erec2=np.array([0.5])

    prec3=np.array([0.5])
    erec3=np.array([0.5])
    # save_accuracy_S=[]
    # save_accuracy_E=[]

    file=str(file)
    angle_los1_,angle_los2_,angle_los3_,local_los1_,local_los2_,local_los3_,cdf_angle1,cdf_location1,cdf_angle2,cdf_location2,cdf_angle3,cdf_location3=Output_loss()
    # batch_RTIlabel=np.reshape(np.concatenate([walk_label[0:int(group_train*2/3)],walk_label2[0:int(group_train/3)]],axis=0),(group_train,1225))

    # Class_train_label = np.array([[0,1]])* np.ones([group_train,2])
    batch_RTIlabel=np.concatenate([walk_label[0:int(group_train*2/3/2)],walk_label2[0:int(group_train/3/2)]],axis=0)
    

    for j in range(ADAETrainEpoch):
        
        ####################Train D #########################
        batch_F=np.concatenate([random_batch(Fix_s1_train,int(group_train/6)),random_batch(Fix_s3_train,int(group_train/6)),random_batch(Fix_s2_train,int(group_train/6)),
            random_batch(Fix_e1_train,int(group_train/6)),random_batch(Fix_e3_train,int(group_train/6)),random_batch(Fix_e2_train,int(group_train/6))],axis=0)

        batch_U=np.concatenate([random_batch(UAV_s1_train,int(group_train/6)),random_batch(UAV_s3_train,int(group_train/6)),random_batch(UAV_s2_train,int(group_train/6)),
            random_batch(UAV_e1_train,int(group_train/6)),random_batch(UAV_e3_train,int(group_train/6)),random_batch(UAV_e2_train,int(group_train/6))],axis=0)

        batch=len(batch_U)

        # batch_F=np.concatenate([random_batch(Fix_s1_train,int(group_train/3)),random_batch(Fix_s3_train,int(group_train/3)),random_batch(Fix_s2_train,int(group_train/3))],axis=0)
        # batch_U=np.concatenate([random_batch(UAV_s1_train,int(group_train/3)),random_batch(UAV_s3_train,int(group_train/3)),random_batch(UAV_s2_train,int(group_train/3))],axis=0)

        

        dl,_= sess.run([d_loss_real,d_trainer],
            feed_dict={lambd_t:lambd ,learning_rate:LR,batch_size:batch, classWeight:classCostWeight, localWeight:localCostWeight, Fix_input:batch_F, UAV_input:batch_U,is_training:True})

        ############################Train G #########################
        batch_U=np.concatenate([random_batch(UAV_s1_train,int(group_train/6)),random_batch(UAV_s3_train,int(group_train/6)),random_batch(UAV_s2_train,int(group_train/6)),
            random_batch(UAV_e1_train,int(group_train/6)),random_batch(UAV_e3_train,int(group_train/6)),random_batch(UAV_e2_train,int(group_train/6))],axis=0)
        # batch_U=np.concatenate([random_batch(UAV_s1_train,int(group_train/3)),random_batch(UAV_s3_train,int(group_train/3)),random_batch(UAV_s2_train,int(group_train/3))],axis=0)
        batch=len(batch_U)

        dg,_= sess.run([d_loss_fake,g_trainer],
            feed_dict={lambd_t:lambd, learning_rate:LR, batch_size:batch, classWeight:classCostWeight, localWeight:localCostWeight, 
            UAV_input:batch_U, RTI_label:batch_RTIlabel , is_training:True, keep_prob:0.5, Class_label:Class_train_label})
        
        # if j % 1 == 0:
        
        # batch_U=np.concatenate([random_batch(UAV_s1_test,int(group_test/3)),random_batch(UAV_s3_test,int(group_test/3)),random_batch(UAV_s2_test,int(group_test/3))],axis=0)
        batch_U=np.concatenate([random_batch(UAV_s1_test,int(group_test/3)),random_batch(UAV_s3_test,int(group_test/3)),random_batch(UAV_s2_test,int(group_test/3)),
                    random_batch(UAV_e1_test,int(group_test/3)),random_batch(UAV_e3_test,int(group_test/3)),random_batch(UAV_e2_test,int(group_test/3))],axis=0)


        pac1, eac1, pac2, eac2, pac3, eac3=sess.run([PACC1, EACC1, PACC2, EACC2 ,PACC3, EACC3], feed_dict={batch_size:int(group_test*2), 
        UAV_input:batch_U, is_training:False, keep_prob:1, Class_label:CADAE_test_label})

        prec1 = np.vstack((prec1,pac1))
        erec1 = np.vstack((erec1,eac1))

        prec2 = np.vstack((prec2,pac2))
        erec2 = np.vstack((erec2,eac2))

        prec3 = np.vstack((prec3,pac3))
        erec3 = np.vstack((erec3,eac3))

        # save_accuracy_S.append(ac1)
        # save_accuracy_E.append(ac2)

        # print(ac1, ac2)
        # if (ac1+ac2) ==2 :
        #     check += 1
        #     if check == 5:
        #         print("Over")
        #         break
        # else:
        #     check = 0

        #####################Test#########################

        # angle_los1,angle_los2,angle_los3,local_los1,local_los2,local_los3,cdf_a1,cdf_l1,cdf_a2,cdf_l2,cdf_a3,cdf_l3=Output_loss()
        # angle_los1_=np.concatenate([angle_los1_,angle_los1],axis=0)
        # angle_los2_=np.concatenate([angle_los2_,angle_los2],axis=0)
        # angle_los3_=np.concatenate([angle_los3_,angle_los3],axis=0)
        # local_los1_=np.concatenate([local_los1_,local_los1],axis=0)
        # local_los2_=np.concatenate([local_los2_,local_los2],axis=0)
        # local_los3_=np.concatenate([local_los3_,local_los3],axis=0)

        # cdf_angle1=np.concatenate([cdf_angle1,cdf_a1],axis=0)
        # cdf_angle2=np.concatenate([cdf_angle2,cdf_a2],axis=0)
        # cdf_angle3=np.concatenate([cdf_angle3,cdf_a3],axis=0)
        # cdf_location1=np.concatenate([cdf_location1,cdf_l1],axis=0)
        # cdf_location2=np.concatenate([cdf_location2,cdf_l2],axis=0)
        # cdf_location3=np.concatenate([cdf_location3,cdf_l3],axis=0)


    # np.savetxt(head+"/angle_los1_"+file+".csv", angle_los1_, delimiter=",")
    # np.savetxt(head+"/angle_los2_"+file+".csv", angle_los2_, delimiter=",")
    # np.savetxt(head+"/angle_los3_"+file+".csv", angle_los3_, delimiter=",")

    # np.savetxt(head+"/local_los1_"+file+".csv", local_los1_, delimiter=",")
    # np.savetxt(head+"/local_los2_"+file+".csv", local_los2_, delimiter=",")
    # np.savetxt(head+"/local_los3_"+file+".csv", local_los3_, delimiter=",")

    # np.savetxt(head+"/cdf_angle_los1_"+file+".csv", cdf_angle1, delimiter=",")
    # np.savetxt(head+"/cdf_angle_los2_"+file+".csv", cdf_angle2, delimiter=",")
    # np.savetxt(head+"/cdf_angle_los3_"+file+".csv", cdf_angle1, delimiter=",")

    # np.savetxt(head+"/cdf_local_los1"+file+".csv", cdf_location1, delimiter=",")
    # np.savetxt(head+"/cdf_local_los2"+file+".csv", cdf_location2, delimiter=",")
    # np.savetxt(head+"/cdf_local_los3"+file+".csv", cdf_location3, delimiter=",")

    


    np.savetxt(head+"/presentAccuracy1"+file+".csv", prec1, delimiter=",")
    np.savetxt(head+"/absentAccuracy1"+file+".csv", erec1, delimiter=",")

    np.savetxt(head+"/presentAccuracy2"+file+".csv", prec2, delimiter=",")
    np.savetxt(head+"/absentAccuracy2"+file+".csv", erec2, delimiter=",")

    np.savetxt(head+"/presentAccuracy3"+file+".csv", prec3, delimiter=",")
    np.savetxt(head+"/absentAccuracy3"+file+".csv", erec3, delimiter=",")
    # np.savetxt(head+"/save_accuracy.csv"+file+".csv", save_accuracy, delimiter=",")


def Output_loss():
    batch_U=np.concatenate([random_batch(UAV_s1_train,int(group_test/3)),random_batch(UAV_s3_train,int(group_test/3)),random_batch(UAV_s2_train,int(group_test/3)),
        random_batch(UAV_e1_train,int(group_test/3)),random_batch(UAV_e3_train,int(group_test/3)),random_batch(UAV_e2_train,int(group_test/3))],axis=0)
    # batch_U=np.concatenate([random_batch(UAV_s1_train,int(group_train/3)),random_batch(UAV_s3_train,int(group_train/3)),random_batch(UAV_s2_train,int(group_train/3))],axis=0)
    batch=len(batch_U)

    # batch_U=np.concatenate([random_batch(UAV_s1_test,int(group_test/3)),random_batch(UAV_s3_test,int(group_test/3)),random_batch(UAV_s2_test,int(group_test/3))],axis=0)
    batch_RTIlabel=np.concatenate([walk_label[0:int(group_test*2/3)],walk_label2[0:int(group_test/3)]],axis=0)
    image1,image2=sess.run([map_b1,map_b2], feed_dict={lambd_t:lambd, learning_rate:LR, batch_size:batch, UAV_input:batch_U, 
        classWeight:classCostWeight, localWeight:localCostWeight, RTI_label:batch_RTIlabel, is_training:False})

    angle_los1=0
    local_los1=0
    angle_los2=0
    local_los2=0
    angle_los3=0
    local_los3=0
    cdf_a1=[]
    cdf_l1=[]
    cdf_a2=[]
    cdf_l2=[]
    cdf_a3=[]
    cdf_l3=[]
    # data_len=int(len(image)/3)

    for i in range (int(group_test/3)):
        oldmap_b=image1[i].reshape((35, 35))
        los,local_los=angle_loss(np.array(np.where(oldmap_b==np.max(oldmap_b))))
        cdf_a1.append(los)
        cdf_l1.append(local_los)
        angle_los1+=los**2
        local_los1+=local_los**2

    for i in range (int(group_test/3),int(group_test*2/3)):
        oldmap_b=image1[i].reshape((35, 35))
        los,local_los=angle_loss(np.array(np.where(oldmap_b==np.max(oldmap_b))))
        cdf_a3.append(los)
        cdf_l3.append(local_los)
        angle_los3+=los**2
        local_los3+=local_los**2

    for i in range (int(group_test/3)):
        oldmap_b=image2[i].reshape((35, 35))
        los,local_los=angle_loss(np.array(np.where(oldmap_b==np.max(oldmap_b))))
        cdf_a2.append(los)
        cdf_l2.append(local_los)
        angle_los2+=los**2
        local_los2+=local_los**2



    angle_los1=np.array([(angle_los1/int(group_test/3))**0.5])
    angle_los2=np.array([(angle_los2/int(group_test/3))**0.5])
    angle_los3=np.array([(angle_los3/int(group_test/3))**0.5])
    local_los1=np.array([(local_los1/int(group_test/3))**0.5])
    local_los2=np.array([(local_los2/int(group_test/3))**0.5])
    local_los3=np.array([(local_los3/int(group_test/3))**0.5])
    # print(local_los1,size1)
    return angle_los1,angle_los2,angle_los3,local_los1,local_los2,local_los3,cdf_a1,cdf_l1,cdf_a2,cdf_l2,cdf_a3,cdf_l3





b1,b2,AP_positions,RX_positions=setup()
########Train Fix data########
Predict_Fix_out= classifier(Fix_input)
FixAcc, FPACC1, FEACC1, FPACC2, FEACC2 ,FPACC3, FEACC3 = Accuracy(Predict_Fix_out,Class_label)
loss_c=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Class_label,logits=Predict_Fix_out,name='loss_c2'))
##############################
tvars = tf.trainable_variables()
c_vars = [var for var in tvars if 'c_' in var.name]
c_trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss_c, var_list=c_vars)

DAE_out=DAE(UAV_input,is_training)
Dx= discriminator(Fix_input,is_training)
Dg= discriminator(DAE_out,is_training, reuse_variables=True)

Predict_DAE_out= classifier(DAE_out, reuse_variables=True)
DAEAcc, PACC1, EACC1, PACC2, EACC2 ,PACC3, EACC3=Accuracy(Predict_DAE_out,Class_label)
loss_DAE=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Class_label,logits=Predict_DAE_out))

batch_size_s=tf.cast(batch_size/2,dtype=tf.int32)
DAE_out_s = tf.slice(DAE_out, [0, 0, 0], [batch_size_s, n_input, 8])
cost_1,map_b1,cost_2,map_b2=RTI(DAE_out_s, batch_size_s)


##################### Define W-GP############################
alpha = tf.random_uniform(shape=[batch_size, n_input, 8], minval=0., maxval=1.)
differences=Fix_input - DAE_out
interpolates = Fix_input + (alpha * differences)
D_inter=discriminator(interpolates,is_training, reuse_variables=True)
gradients = tf.gradients(D_inter, [interpolates])[0]
red_idx = tf.range(1, interpolates.shape.ndims)
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) ))
gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

##################### Define Loss############################
d_loss_real = -tf.reduce_mean(Dx)
d_loss_fake = tf.reduce_mean(Dg)
g_loss = localWeight*(cost_1 + cost_2) + classWeight*loss_DAE - d_loss_fake
d_loss = d_loss_fake+d_loss_real + (lambd_t * gradient_penalty)

#################### Define train ########################################
optimizer_gen =tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
optimizer_disc =tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator')
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
with tf.control_dependencies(disc_update_ops):
    d_trainer = optimizer_disc.minimize(d_loss, var_list=disc_vars)

gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
with tf.control_dependencies(gen_update_ops):
    g_trainer = optimizer_gen.minimize(g_loss, var_list=gen_vars)


tf.get_variable_scope().reuse_variables()
saver=tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
# sess = tf.Session(config=config)
print('------------------------------------------------------------------------')
print(tf.test.is_gpu_available())
print('------------------------------------------------------------------------')


walk_label,walk_label2=setup_RTI()


Fix_s1,Fix_s1_train,Fix_s1_test=take_data("humanPresence/Clean/FixP_200h_D1_L1.csv")
Fix_s2,Fix_s2_train,Fix_s2_test=take_data("humanPresence/Clean/FixP_200h_D2_L1.csv")
Fix_s3,Fix_s3_train,Fix_s3_test=take_data("humanPresence/Clean/FixP_200h_D3_L1.csv")

Fix_e1,Fix_e1_train,Fix_e1_test=take_data("humanAbsence/Clean/FixA_200h_D1_L1.csv")
Fix_e2,Fix_e2_train,Fix_e2_test=take_data("humanAbsence/Clean/FixA_200h_D2_L1.csv")
Fix_e3,Fix_e3_train,Fix_e3_test=take_data("humanAbsence/Clean/FixA_200h_D3_L1.csv")

FixEest_E_size=len(Fix_e1_test)+len(Fix_e2_test)+len(Fix_e3_test)
FixTest_S_size=len(Fix_s1_test)+len(Fix_s2_test)+len(Fix_s3_test)
######################################################################
UAV_e1,UAV_e1_train,UAV_e1_test=take_data("humanAbsence/Corrupted/UAVA_200h_D1_L1.csv")
UAV_e2,UAV_e2_train,UAV_e2_test=take_data("humanAbsence/Corrupted/UAVA_200h_D2_L1.csv")
UAV_e3,UAV_e3_train,UAV_e3_test=take_data("humanAbsence/Corrupted/UAVA_200h_D3_L1.csv")

UAV_s1,UAV_s1_train,UAV_s1_test=take_data("humanPresence/Corrupted/UAVP_200h_D1_L1.csv")
UAV_s2,UAV_s2_train,UAV_s2_test=take_data("humanPresence/Corrupted/UAVP_200h_D2_L1.csv")
UAV_s3,UAV_s3_train,UAV_s3_test=take_data("humanPresence/Corrupted/UAVP_200h_D3_L1.csv")

print(UAV_s1_train.shape, UAV_s1_test.shape)
print(UAV_s2_train.shape, UAV_s2_test.shape)
print(UAV_s3_train.shape, UAV_s3_test.shape)

group_train = 12
group_test = int ( min(len(UAV_s1_test),len(UAV_s2_test),len(UAV_s3_test))*3)

Class_train_label=np.concatenate([np.array([[0,1]])* np.ones([int(group_train/2),2]),np.array([[1,0]])* np.ones([int(group_train/2),2])],axis=0)
Fix_test_label=np.concatenate([np.array([[0,1]])* np.ones([FixTest_S_size,2]),np.array([[1,0]])* np.ones([FixEest_E_size,2])],axis=0)
# CADAE_test_label1 = np.array([[0,1]])* np.ones([group_test,2])
# CADAE_test_label2 = np.array([[1,0]])* np.ones([group_test,2])
CADAE_test_label=np.concatenate([np.array([[0,1]])* np.ones([group_test,2]),np.array([[1,0]])* np.ones([group_test,2])],axis=0)



with tf.Session(config=config) as sess:
    init = tf.global_variables_initializer()
    for k in range (allTrainEpisode):
        sess.run(init)
        for i in range (classTrainEpisode):
            print(i)
            sess.run(init)
            save_ac=train_classify()
            print(np.max(save_ac))
            if np.max(save_ac) == 1 :
                print('Classify Success')
                break
        train_GAN('mainclasslocalloss32',k)
        print('###',k)


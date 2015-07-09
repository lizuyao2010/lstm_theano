#!/usr/bin/python
# -*- coding: utf-8 -*-
from theano import *
from theano import tensor as T
import numpy as np
from numpy import *
from keras.preprocessing import sequence
import pickle

negative_sample_size=80
dtype=theano.config.floatX
volsize=3114+1
relsize=3358+1
max_sen_len=14


def parseline(line):
    line=line.split()
    code=[int(x) for x in line]
    return code

def encode_indexs(indexs):
    return np.array(indexs,dtype=int32)


# def get_traindata(filename):
#     f=open(filename,'r')
#     count=1
#     # total=[]
#     xs=[]
#     ys=[]
#     zs=[]
#     data=[]
#     x=[]
#     y=[]
#     z=[]
#     for line in f:
#         line=line.strip()
#         if line=='':
#            random.shuffle(data)
#            # total+=data[:negative_sample_size]
#            for i in range(min(negative_sample_size,len(data))):
#             xs.append(data[i][0])
#             ys.append(data[i][1])
#             zs.append(data[i][2])
#            data=[]
#            count=1
#            continue
#         if count==1:
#            x=parseline(line)
#            x=encode_indexs(x)
#            count+=1
#            continue
#         elif count==2:
#            y=parseline(line)
#            y=encode_indexs(y)
#            count=0
#            continue
#         else:
#            z=parseline(line)
#            z=encode_indexs(z)
#            data.append([x,y,z])
#     f.close()
#     # return total
#     return xs,ys,zs

def get_traindata(filename):
    f=open(filename,'r')
    count=1
    # total=[]
    data=[]
    x=[]
    y=[]
    z=[]
    xs=[]
    ys=[]
    zs=[]
    for line in f:
        line=line.strip()
        if line=='':
           random.shuffle(data)
           for i in range(min(negative_sample_size,len(data))):
            xs.append(data[i][0])
            ys.append(data[i][1])
            zs.append(data[i][2])
           data=[]
           count=1
           continue
        if count==1:
           x=parseline(line)
           count+=1
           continue
        elif count==2:
           y=parseline(line)
           count=0
           continue
        else:
           z=parseline(line)
           data.append([x,y,z])
    f.close()
    return xs,ys,zs


xs,ys,zs=get_traindata('/share/project/zuyao/data/train_web_soft_0.8_code.txt')
xs = encode_indexs(sequence.pad_sequences(xs))
ys = encode_indexs(sequence.pad_sequences(ys))
zs = encode_indexs(sequence.pad_sequences(zs))
traindata=[xs,ys,zs]
f=open('traindata.pickle','w')
pickle.dump(traindata,f)
f.close()
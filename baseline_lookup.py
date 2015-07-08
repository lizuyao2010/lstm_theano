#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:31:15 2015

@author: lizuyao2010
"""
from theano import *
from theano import tensor as T
import numpy as np
from numpy import *
import time
import os
import random
import sys
import json
import pickle
from collections import OrderedDict

batchsize=1
max_sen_len=14
negative_sample_size=80
dtype=theano.config.floatX
volsize=3114+1
relsize=3358+1
margin=1
threshold=0
gammavalue = np.float32(0.0)
nmodel = 100
nen = nmodel
inputbias = np.float32(-0.0)
outputbias = np.float32(-0.0)
forgetbias = np.float32(0.0)
pretrained_model=''


def computeF1(goldList,predictedList):

  """Assume all questions have at least one answer"""
  if len(goldList)==0:
    raise Exception("gold list may not be empty")
  """If we return an empty list recall is zero and precision is one"""
  if len(predictedList)==0:
    return (0,1,0)
  """It is guaranteed now that both lists are not empty"""

  precision = 0
  for entity in predictedList:
    if entity in goldList:
      precision+=1
  precision = float(precision) / len(predictedList)

  recall=0
  for entity in goldList:
    if entity in predictedList:
      recall+=1
  recall = float(recall) / len(goldList)

  f1 = 0
  if precision+recall>0:
    f1 = 2*recall*precision / (precision + recall)
  return (recall,precision,f1)

def parseline(line):
    line=line.split()
    code=[int(x) for x in line]
    return code

# def encode_indexs(indexs,size):
#     sent = np.zeros((len(indexs),size),dtype=dtype)
#     for i, w in enumerate(indexs):
#         sent[i,w] = 1
#     return sent

def encode_indexs(indexs,size):
    return np.array(indexs,dtype=int32)

def getbatch(data,offset):
    batch=[]
    for i in range(batchsize):
        index=shuffle_index[i+offset]
        x=data[index][0]
        y=data[index][1]
        z=data[index][2]
        example=[encode_indexs(x,volsize),encode_indexs(y,relsize),encode_indexs(z,relsize)]
        batch.append(example)
    return batch

def get_traindata(filename):
    f=open(filename,'r')
    count=1
    total=[]
    data=[]
    x=[]
    y=[]
    z=[]
    for line in f:
        line=line.strip()
        if line=='':
           random.shuffle(data)
           total+=data[:negative_sample_size]
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
    return total


def get_testData(filename):
  f=open(filename,'r')
  test_data=[]
  count=1
  x=[]
  z=[]
  item={}
  j=1
  for line in f:
        line=line.strip()
        if line=='':
           test_data.append(item)
           item={}    
           count=1
           continue
        if count==1:
           q,gold=line.split(' # ')
           gold=json.loads(gold)
           item['gold']=gold
           item['question']=q
           count+=1
           continue
        elif count==2:
           x=parseline(line)
           item['qcode']=x
           count=0
           continue
        else:
           name,path=line.split(' # ')
           z=parseline(path)
           item[name]=z
  f.close()
  return test_data

def test(testdata,f_log,epi):
    averageRecall=0
    averagePrecision=0
    averageF1=0
    count=0
    start_time=time.time()
    testdataSize=len(testdata)
    for item in testdata:
      scoretable=[]
      qcode=q_encoder(encode_indexs(item['qcode'],volsize))
      for key in item:
        if key=='qcode' or key=='gold' or key=='question':
          continue
        ccode=c_encoder(encode_indexs(item[key],relsize))
        score=np.dot(qcode,ccode)
        scoretable.append((key,float(score)))
      sorted_table = sorted(scoretable, key=lambda tup: tup[1], reverse=True)
      maxscore=sorted_table[0][1]
      predicts=[]
      for tup in sorted_table:
        if (maxscore-tup[1])>threshold*margin:
          break
        else:
          if tup[0] not in predicts:
            predicts.append(tup[0])
      # print >>f_log,item['question'],item['gold'],predicts
      recall, precision, f1 = computeF1(item['gold'],predicts)
      averageRecall += recall
      averagePrecision += precision
      averageF1 += f1
      count+=1
      # print 'finish test data',count,'/',testdataSize,'epoch',epi
      print '[testing] epoch %i >> %2.2f%%'%(epi,(count+1)*100./testdataSize),'completed in %.2f (sec) <<\r'%(time.time()-start_time),
      sys.stdout.flush()
    """Print final results"""
    averageRecall = float(averageRecall) / count
    averagePrecision = float(averagePrecision) / count
    averageF1 = float(averageF1) / count
    print "Number of questions: " + str(count),'epoch',epi
    print "Average recall over questions: " + str(averageRecall),'epoch',epi
    print "Average precision over questions: " + str(averagePrecision),'epoch',epi
    print "Average f1 over questions (accuracy): " + str(averageF1),'epoch',epi
    if (averagePrecision + averageRecall)==0:
      averageNewF1=0.0
    else:
      averageNewF1 = 2 * averageRecall * averagePrecision / (averagePrecision + averageRecall)
    print "F1 of average recall and average precision: " + str(averageNewF1),'epoch',epi


traindata=get_traindata('/share/project/zuyao/data/train_web_soft_0.8_code.txt')
testdata=get_testData('/share/project/zuyao/data/dev_web_soft_code_list.txt')
trainsize=len(traindata)
shuffle_index=range(trainsize)
'''define module'''

# therefore we use the logistic function
sigma = lambda x: 1 / (1 + T.exp(-x))
# for the other activation function we use the tanh
act = T.tanh
def sample_weights(nrow, ncol):
    # bound = (np.sqrt(6.0) / np.sqrt(nrow+ncol) ) * 1.0
    bound = 0.08
    # nrow -- # of prev layer units, ncol -- # of this layer units
    # this is form Bengio's 2010 paper
    values = \
    np.random.uniform(low=-bound, high=bound, size=(nrow, ncol))
    return np.cast[dtype](values)

if pretrained_model:
  f = open(pretrained_model,'r')
  model=pickle.load(f)
  f.close()
  Een = theano.shared(model['Een'],name='Een')
  Ren = theano.shared(model['Ren'],name='Ren')


else:
  Een = theano.shared(sample_weights(volsize,nen),name='Een')
  Ren = theano.shared(sample_weights(relsize,nen),name='Ren')


# X = T.imatrix('X')
# Y = T.imatrix('Y')
# Z = T.imatrix('Z')
X = T.ivector('X')
Y = T.ivector('Y')
Z = T.ivector('Z')


# X_emb=T.sum(theano.dot(X,Een),axis=0)
# Y_emb=T.sum(theano.dot(Y,Ren),axis=0)
# Z_emb=T.sum(theano.dot(Z,Ren),axis=0)

X_emb=T.sum(Een[X],axis=0)
Y_emb=T.sum(Ren[Y],axis=0)
Z_emb=T.sum(Ren[Z],axis=0)

print "finish encoder"

l2 = T.sum(Een**2) + T.sum(Ren**2)
gam = theano.shared(gammavalue,name='gam')
XY_score=T.dot(X_emb,Y_emb)
XZ_score=T.dot(X_emb,Z_emb)
Cost = T.maximum(0,margin-(XY_score-XZ_score)) + gam * l2

print "finish cost"
#TODO: Adam
print "adagrad"
#TODO: take grads
params = [Een, Ren]

gEen, gRen = T.grad(Cost, params)
# compute the gradients with respect to the model parameters
gparams = [gEen, gRen]# compute list of weights updates

aEen = theano.shared(np.zeros((volsize,nen),dtype=dtype),name='aEen')
aRen = theano.shared(np.zeros((relsize,nen),dtype=dtype),name='aRen')
_accugrads=[aEen,aRen]

print "finish grads"

eps=1.E-6

##############
# ADAGRAD    #
##############
""" Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
"""
# batch_x = T.fmatrix('batch_x')
# batch_y = T.ivector('batch_y')
learning_rate = T.fscalar('lr')  # learning rate to use
updates = OrderedDict()
for accugrad, param, gparam in zip(_accugrads, params, gparams):
    # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
    agrad = accugrad + gparam * gparam
    dx = - (learning_rate / T.sqrt(agrad + eps)) * gparam
    updates[param] = param + dx
    updates[accugrad] = agrad    

#TODO: define training function and score function
learn_model_fn = theano.function(inputs = [X, Y, Z,learning_rate],
                                 outputs = Cost,
                                 updates = updates)
q_encoder = theano.function(inputs = [X], outputs = X_emb)
c_encoder = theano.function(inputs = [Y], outputs = Y_emb)
print "finish functions"



lr=0.05
#TODO: training
print "start training"
max_epoch = 20
train_errs = np.ndarray(max_epoch)
f_log=open('log','a')
for epi in xrange(max_epoch):
    start_time = time.time()
    random.shuffle(shuffle_index)
    # print 'epoch',epi
    err = 0.0
    for offset in range(0,trainsize,batchsize):
        if offset+batchsize>trainsize:
            break
        batch=getbatch(traindata,offset)[0]
        train_cost=learn_model_fn(batch[0],batch[1],batch[2],0.05)
        err+=train_cost
        # print "finish train data, ", offset,'/',trainsize, 'epoch',epi
        print '[learning] epoch %i >> %2.2f%%'%(epi,(offset+1)*100./trainsize),'completed in %.2f (sec) <<\r'%(time.time()-start_time),
        sys.stdout.flush()
    train_errs[epi] = err / trainsize
    # print "finish epoch", epi
    print 'training error', train_errs[epi]
    # print "start testing ", epi
    test(testdata,f_log,epi)
    end_time = time.time()
    print "finish testing, save model"
    # save the model first
    model = {}
    # for encoder
    model['Een'] = Een.get_value()
    model['Ren'] = Ren.get_value()

    fname = 'model'+str(epi)+'.pickle'
    f = open(fname,'w')
    pickle.dump(model,f)
    f.close()
    # print >> f_log,'run',(end_time-start_time)/60,'mins so far'
print "finish training"
f_log.close()
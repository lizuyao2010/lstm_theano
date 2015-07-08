#!/usr/bin/python
from theano import *
from theano import tensor as T

import numpy as np
import time
import os
import random
import sys
import json
import pickle
import theano
from theano import sandbox
import os
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams

ne=225961
de=50
margin=3
lr=0.05
maxLen=15
batchsize=800
negative_sample_size=1

itensor5 = T.TensorType('int32', (False,)*5)
dtype=theano.config.floatX

'''trainning model'''
matrix_ndarray=np.random.uniform(-0.08,0.08,(ne+1,de)).astype(dtype)
subtract=np.array([1,-1])

idxs=T.itensor4('ids')
mask=itensor5('mask')
emb = theano.shared(name='embeddings',value=matrix_ndarray)
subset = emb[idxs]
# mask subset
subset_m=subset*mask
x = T.sum(subset_m,axis=3)
p=T.prod(x,axis=2)
s=T.sum(p,axis=2)
mul=theano.shared(name="mul",value=subtract)
diff=T.dot(s,mul)
cost=T.sum(T.maximum(0,margin-diff))

'''testing model'''
idxs_t=T.imatrix('ids')
mask_t=T.itensor3('mask')
subset_t = emb[idxs_t]
# mask subset
subset_t_m=subset_t*mask_t
x_t = T.sum(subset_t_m,axis=1)
p_t=T.prod(x_t,axis=0)
s_t=T.sum(p_t,axis=0)



#TODO: Adam
print "adam"
alpha = theano.shared(np.float32(0.001),'alpha')
beta1 = theano.shared(np.float32(0.9),'beta1')
beta2 = theano.shared(np.float32(0.999), 'beta2')
eps = theano.shared(np.float32(0.00000001),'eps')
lam = theano.shared(np.float32(1.0 - 0.00000001), 'lam')

# adam - m
mEmb = theano.shared(np.zeros((ne+1,de),dtype=dtype),name='mEmb')
mparams = [mEmb]
# adam - v
vEmb = theano.shared(np.zeros((ne+1,de),dtype=dtype),name='vEmb')
vparams = [vEmb]

timestep = theano.shared(np.float32(1))
beta1_t = beta1*(lam**(timestep-1))

#TODO: take grads
params = [emb]
gEmb = T.grad(cost, params)
gparams = gEmb

print "finish grads"

#TODO: update
updates = []
for param, gparam, mparam, vparam in zip(params, gparams, mparams, vparams):
    newm0 = beta1_t * mparam + (1-beta1_t) * gparam
    newv0 = beta2 * vparam + (1-beta2) * (gparam**2)
    newm = newm0 / (1-(beta1**timestep) )
    newv = newv0 / (1-(beta2**timestep) )
    newparam0 = param - alpha*( newm/(T.sqrt(newv)+eps) )
    updates.append((param, newparam0))
    updates.append((mparam, newm0))
    updates.append((vparam, newv0))
updates.append((timestep, timestep+1.0))
print "finish updates"
#TODO: define training function and validation function
train_model=theano.function(inputs=[idxs,mask],outputs=cost,updates=updates)
test_model=theano.function(inputs=[idxs_t,mask_t],outputs=s_t)

print "finish functions"


def parseline(line):
    line=line.split()
    code=[int(x)+1 for x in line]
    codeSize=len(code)
    for i in range(codeSize,maxLen):
        code.append(0)
    return [code,codeSize]
    
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
    
"""return a tuple with recall, precision, and f1 for one example"""
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
           count=1
           continue
        if count==1:
           q,gold=line.split(' # ')
           gold=json.loads(gold)
           test_data.append(item)
           item.clear()
           item['gold']=gold
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

def test(testdata):
    averageRecall=0
    averagePrecision=0
    averageF1=0
    count=0
    for item in testdata:
      scoretable=[]
      for key in item:
        if key=='qcode' or key=='gold':
          continue
        qcode,qsize=item['qcode']
        ccode,csize=item[key]
        input_pair=np.array([qcode,ccode],dtype=np.int32)
        qcode_m=get_mask(qsize)
        ccode_m=get_mask(csize)
        input_mask=np.array([qcode_m,ccode_m],dtype=np.int32)
        score=float(test_model(input_pair,input_mask))
        # print 'score',score
        scoretable.append((key,score))
      sorted_table = sorted(scoretable, key=lambda tup: tup[1], reverse=True)
      maxscore=sorted_table[0][1]
      predicts=[]
      for tup in sorted_table:
        if (maxscore-tup[1])>0.3*margin:
          break
        predicts.append(tup[0])
      recall, precision, f1 = computeF1(item['gold'],predicts)
      averageRecall += recall
      averagePrecision += precision
      averageF1 += f1
      count+=1
      #print 'test',count
    """Print final results"""
    averageRecall = float(averageRecall) / count
    averagePrecision = float(averagePrecision) / count
    averageF1 = float(averageF1) / count
    print "Number of questions: " + str(count)
    print "Average recall over questions: " + str(averageRecall)
    print "Average precision over questions: " + str(averagePrecision)
    print "Average f1 over questions (accuracy): " + str(averageF1)
    averageNewF1 = 2 * averageRecall * averagePrecision / (averagePrecision + averageRecall)
    print "F1 of average recall and average precision: " + str(averageNewF1)

def get_mask(size):
    l=[1]*size+[0]*(maxLen-size)
    l=np.array([l]*de,dtype=np.int32)
    return np.transpose(l)

def getbatch(data,offset):
    batch=[]
    batch_m=[]
    for i in range(batchsize):
        x,x_size=data[i+offset][0]
        y,y_size=data[i+offset][1]
        z,z_size=data[i+offset][2]
        corr=[x,y]
        wrong=[x,z]
        example=[corr,wrong]
        batch.append(example)
        # mask
        x_m=get_mask(x_size)
        y_m=get_mask(y_size)
        z_m=get_mask(z_size)
        corr_m=[x_m,y_m]
        wrong_m=[x_m,z_m]
        example_m=[corr_m,wrong_m]
        batch_m.append(example_m)
    return [batch,batch_m]
    
traindata=get_traindata('/share/project/zuyao/data/train_web.txt')
testdata=get_testData('/share/project/zuyao/data/test.txt')
datasize=len(traindata)


epoch=10

for i in xrange(epoch):
    start_time = time.clock()
    print 'epoch',i
    for offset in range(0,datasize,batchsize):
        if offset+batchsize>datasize:
            break
        batch,batch_m=getbatch(traindata,offset)
        batch=np.array(batch,dtype=np.int32)
        batch_m=np.array(batch_m,dtype=np.int32)
        l=train_model(batch,batch_m)
        print 'train',offset,'/',datasize
    test(testdata)
    end_time = time.clock()
    print 'run',(end_time-start_time)/60,'mins so far'
    

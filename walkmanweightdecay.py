# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:30:11 2015

this code gets the model of LSTM for planer
1) use uni-direction
2) use attention

this code deletes the b_y -- no bias 
this code also get the new feats:
if it is a wall and can not go this way, set it to -1 instead of 0
same change is made in greedy search

this code also uses deep-out-put -- use the top-layer in Attention model 

add norm constrains to the parameters such that the model is balanced

use new initial method - Xavier 2010

#use new decaying rate -- *(1/2) each X epochs

use Adam

this model uses Jelly and L to train
Grid to test

use all info in all directions this time
use weight decay

@author: hongyuan
"""

import pickle
import time
import numpy as np
import theano
from theano import sandbox
import theano.tensor as T
import os
import scipy.io
from collections import defaultdict
import numpy as np
from gs2 import gs2
from theano.tensor.shared_randomstreams import RandomStreams

dtype=theano.config.floatX

#TODO: some hyper-parameters
dropoutvalue = np.float32(1.0)
gammavalue = np.float32(0.0)
nmodel = 80

#TODO: get all the data
f = open('../databag3.pickle','r')
bag = pickle.load(f)
f.close()

t1 = 'jelly'
t2 = 'l'
data1 = bag[t1]
data2 = bag[t2]

f = open('../stat.pickle','r')
stat = pickle.load(f)
f.close()
#stat['ind2word'] = ind2word
word2ind = stat['word2ind']
vocabmat = stat['vocabmat']
volsize = stat['volsize']
#wordfreq = stat['wordfreq']

f = open('../mapscap1000.pickle','r')
maps = pickle.load(f)
f.close()

f = open('../valselect.pickle','r')
perms = pickle.load(f)
f.close()
perm1 = perms[t1]
perm2 = perms[t2]

print "finish preparing the data"
#TODO: get traindata valdata
traindata = []
valdata = []
for i, data in enumerate(data1):
    if i in perm1:
        valdata.append(data)
    else:
        traindata.append(data)
#
for i, data in enumerate(data2):
    if i in perm2:
        valdata.append(data)
    else:
        traindata.append(data)

#TODO: some initialization
# therefore we use the logistic function
sigma = lambda x: 1 / (1 + T.exp(-x))
# for the other activation function we use the tanh
act = T.tanh
#
inputbias = np.float32(-0.0)
outputbias = np.float32(-0.0)
forgetbias = np.float32(0.0)
#
nen = nmodel # size of LSTM encoder
nde = nmodel # size of LSTM decoder
nbeta = nmodel
ny = 4
#D = 57 # 6 + 3 *(8+3+6)
D = 78 # 6 + 4 *(3+8+6+1)

print "finish initialiazation"

#TODO: dropout
srng = RandomStreams(seed=0)
windows = srng.uniform((nde,)) < dropoutvalue
getwins = theano.function([],windows)

#TODO: Use a initialization method

def sample_weights(nrow, ncol):
    bound = (np.sqrt(6.0) / np.sqrt(nrow+ncol) ) * 1.0
    # nrow -- # of prev layer units, ncol -- # of this layer units
# this is form Bengio's 2010 paper
    values = \
    np.random.uniform(low=-bound, high=bound, size=(nrow, ncol))
    return np.cast[dtype](values)

'''
def sample_weights(numrow, numcol):
    values = np.ndarray([numrow, numcol], dtype=dtype)
    for coli in xrange(numcol):
        vals = np.random.normal(loc=0.0, scale=0.01,  size=(numrow,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[:,coli] = vals
    #_,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value                      
    #values = values / svs[0]
    return values
'''
#TODO: define the encoder
def encoder(wordt, htm1, ctm1, 
            Een, Wxien, Whien, bien, Wxfen, Whfen, bfen, 
            Wxcen, Whcen, bcen, Wxoen, Whoen, boen):
    xt = theano.dot(wordt, Een)
    it = sigma(theano.dot(xt,Wxien) + theano.dot(htm1,Whien) + bien )
    ft = sigma(theano.dot(xt,Wxfen) + theano.dot(htm1,Whfen) + bfen )
    ct = ft * ctm1 + it*act(theano.dot(xt,Wxcen)+theano.dot(htm1,Whcen)+bcen )
    ot = sigma(theano.dot(xt,Wxoen) + theano.dot(htm1,Whoen) + boen )
    ht = ot * act(ct)
#    yt = T.concatenate([addzero,tempyt],axis=0)
    return ht, ct

Een = theano.shared(sample_weights(volsize,nen),name='Een')
Wxien = theano.shared(sample_weights(nen,nen), name='Wxien')
Whien = theano.shared(sample_weights(nen,nen), name='Whien')
bien = theano.shared(inputbias*np.ones((nen,),dtype=dtype), name='bien')
Wxfen = theano.shared(sample_weights(nen,nen), name='Wxfen')
Whfen = theano.shared(sample_weights(nen,nen), name='Whfen')
bfen = theano.shared(forgetbias*np.ones((nen,),dtype=dtype), name='bfen')
Wxcen = theano.shared(sample_weights(nen,nen), name='Wxcen')
Whcen = theano.shared(sample_weights(nen,nen), name='Whcen')
bcen = theano.shared(np.zeros((nen,),dtype=dtype),name='bcen')
Wxoen = theano.shared(sample_weights(nen,nen),name='Wxoen')
Whoen = theano.shared(sample_weights(nen,nen),name='Whoen')
boen = theano.shared(outputbias*np.ones((nen,),dtype=dtype), name='boen')

c0en = theano.shared(np.zeros(nen, dtype=dtype))
h0en = theano.shared(np.zeros(nen, dtype=dtype))

X = T.matrix(dtype=dtype)
[hvals_en, cvals_en], _ = \
theano.scan(fn=encoder,
            sequences=dict(input=X,taps=[0]),
            outputs_info=[dict(initial=h0en,taps=[-1]), dict(initial=c0en,taps=[-1])],
            non_sequences=[Een, Wxien, Whien, bien, Wxfen, Whfen, bfen,
                           Wxcen, Whcen, bcen, Wxoen, Whoen, boen])
print "finish encoder"
#TODO: define the decoder
# hmat is the matrix generated by encoder -- length of sentence * size of h
#beta = \
#theano.dot( act((theano.dot(T.outer(outerh,htm1),Wbeta)) + (theano.dot(A,Ubeta))) , vbeta)

def decoder(localt, stm1, cstm1, hmat,
            Wbeta, Ubeta, vbeta,
            Wzide, Wzfde, Wzcde, Wzode,
            Ede, Wxide, Wside, bide, Wxfde, Wsfde, bfde, 
            Wxcde, Wscde, bcde, Wxode, Wsode, bode,
            L0, Ls, Lz):
    xt = theano.dot(localt, Ede)
    # get z from hmat (sentlen * nen), stm1
    beta = \
    theano.dot( act( theano.dot(hmat,Ubeta) + theano.dot(stm1,Wbeta) ) , vbeta )
    alpha = T.exp(beta-T.max(beta)) / T.sum(T.exp(beta-T.max(beta)) )
    zt = theano.dot(alpha, hmat)
    #
    it = sigma(theano.dot(xt,Wxide) + theano.dot(stm1,Wside) + theano.dot(zt,Wzide) + bide )
    ft = sigma(theano.dot(xt,Wxfde) + theano.dot(stm1,Wsfde) + theano.dot(zt,Wzfde) + bfde )
    cst = ft * cstm1 + it*act(theano.dot(xt,Wxcde)+theano.dot(stm1,Wscde)+ theano.dot(zt,Wzcde) +bcde )
    ot = sigma(theano.dot(xt,Wxode) + theano.dot(stm1,Wsode) + theano.dot(zt,Wzode) +bode )
    st = ot * act(cst)
    #
    winst = getwins()
    stfory = st * winst
    #
    yt0 = T.dot( (xt + T.dot(stfory, Ls) + T.dot(zt, Lz) ) , L0)
    #yt0 = theano.dot(st,Wsyde)
    yt0max = T.max(yt0)
    #yt0maxvec = T.maximum(yt0, yt0max)
    yt = T.exp(yt0-yt0max) / T.sum(T.exp(yt0-yt0max))
    logyt = yt0-yt0max-T.log(T.sum(T.exp(yt0-yt0max)))
    #yt = T.exp(yt0-yt0maxvec) / T.sum(T.exp(yt0-yt0maxvec))
    #logyt = yt0-yt0maxvec-T.log(T.sum(T.exp(yt0-yt0maxvec)))
#    yt = T.concatenate([addzero,tempyt],axis=0)
    return st, cst, yt, logyt

Wbeta = theano.shared(sample_weights(nde,nbeta), name='Wbeta')
Ubeta = theano.shared(sample_weights(nen,nbeta), name='Ubeta')
vbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='vbeta')
#
Wzide = theano.shared(sample_weights(nen,nde), name='Wzide')
Wzfde = theano.shared(sample_weights(nen,nde), name='Wzfde')
Wzcde = theano.shared(sample_weights(nen,nde), name='Wzcde')
Wzode = theano.shared(sample_weights(nen,nde), name='Wzode')
#
Ede = theano.shared(sample_weights(D,nde), name='Ede')
Wxide = theano.shared(sample_weights(nde,nde), name='Wxide')
Wside = theano.shared(sample_weights(nde,nde), name='Wside')
bide = theano.shared(inputbias*np.ones((nde,),dtype=dtype), name='bide')
Wxfde = theano.shared(sample_weights(nde,nde), name='Wxfde')
Wsfde = theano.shared(sample_weights(nde,nde), name='Wsfde')
bfde = theano.shared(forgetbias*np.ones((nde,),dtype=dtype), name='bfde')
Wxcde = theano.shared(sample_weights(nde,nde), name='Wxcde')
Wscde = theano.shared(sample_weights(nde,nde), name='Wscde')
bcde = theano.shared(np.zeros((nde,),dtype=dtype), name='bcde')
Wxode = theano.shared(sample_weights(nde,nde), name='Wxode')
Wsode = theano.shared(sample_weights(nde,nde), name='Wsode')
bode = theano.shared(outputbias*np.ones((nde,),dtype=dtype), name='bode')
#Wsyde = theano.shared(sample_weights(nde,ny), name='Wsyde')
#byde = theano.shared(actfreq, name='byde')
L0 = theano.shared(sample_weights(nde,ny), name='L0')
Ls = theano.shared(sample_weights(nde,nde), name='Ls')
Lz = theano.shared(sample_weights(nen,nde), name='Lz')
#
cs0de = theano.shared(np.zeros(nde, dtype=dtype))
s0de = theano.shared(np.zeros(nde, dtype=dtype))
#
L = T.matrix(dtype=dtype) # local feats
[svals_de, csvals_de, yvals_de, logyvals_de], _ = \
theano.scan(fn=decoder,
            sequences=dict(input=L,taps=[0]),
            outputs_info=[dict(initial=s0de,taps=[-1]), dict(initial=cs0de,taps=[-1]), None, None],
            non_sequences=[hvals_en,
                           Wbeta, Ubeta, vbeta,
                           Wzide, Wzfde, Wzcde, Wzode,
                           Ede, Wxide, Wside, bide, Wxfde, Wsfde, bfde,
                           Wxcde, Wscde, bcde, Wxode, Wsode, bode,
                           L0, Ls, Lz])
print "finish decoder"                           
#TODO: Now can define loss here!!!!
Target = T.matrix(dtype=dtype) # target matrix
#cost = -T.sum(Target * logyvals_de)
# for weight decay
'''
params = [Een, Wxien, Whien, bien, Wxfen, Whfen, bfen,
          Wxcen, Whcen, bcen, Wxoen, Whoen, boen,
          Wbeta, Ubeta, vbeta,
          Wzide, Wzfde, Wzcde, Wzode,
          Ede, Wxide, Wside, bide, Wxfde, Wsfde, bfde,
          Wxcde, Wscde, bcde, Wxode, Wsode, bode,
          L0, Ls, Lz]
'''
l2 = T.sum(Een**2) + T.sum(Wxien**2) + T.sum(Whien**2) + T.sum(bien**2) + \
T.sum(Wxfen**2) + T.sum(Whfen**2) + T.sum(bfen**2) + \
T.sum(Wxcen**2) + T.sum(Whcen**2) + T.sum(bcen**2) + \
T.sum(Wxoen**2) + T.sum(Whoen**2) + T.sum(boen**2) + \
T.sum(Wbeta**2) + T.sum(Ubeta**2) + T.sum(vbeta**2) + \
T.sum(Wzide**2) + T.sum(Wzfde**2) + T.sum(Wzcde**2) + T.sum(Wzode**2) + \
T.sum(Ede**2) + T.sum(Wxide**2) + T.sum(Wside**2) + T.sum(bide**2) + \
T.sum(Wxfde**2) + T.sum(Wsfde**2) + T.sum(bfde**2) + \
T.sum(Wxcde**2) + T.sum(Wscde**2) + T.sum(bcde**2) + \
T.sum(Wxode**2) + T.sum(Wsode**2) + T.sum(bode**2) + \
T.sum(L0**2) + T.sum(Ls**2) + T.sum(Lz**2)

gam = theano.shared(gammavalue,name='gam')
#
cost = -T.mean(T.sum((Target * logyvals_de),axis=1) ) + gam * l2

print "finish cost"
#TODO: about learning rate set up
#decreasing_factor = theano.shared(np.float32( (len(traindata))*5.0 ))
#increment = theano.shared(np.float32(1.0))
#decreasing_interval = theano.shared(np.float32( (len(traindata))*5.0 ))
# we use 3000 here cuz there are 3000 training points in each epoch
# so we want to decrease every 3000 steps -- controlled by decreasing_factor/decressing_interval
# for example, at beginning, df/din = 1, so eta = rate/(1), ...then 
# when df = 6001, df/din = 2, so eta = rate/(2), ...
#starting_rate = theano.shared(np.float32(0.1))
#learning_rate = theano.shared(np.float32(0.1))
#TODO: Adam
print "adam"
alpha = theano.shared(np.float32(0.001),'alpha')
beta1 = theano.shared(np.float32(0.9),'beta1')
beta2 = theano.shared(np.float32(0.999), 'beta2')
eps = theano.shared(np.float32(0.00000001),'eps')
lam = theano.shared(np.float32(1.0 - 0.00000001), 'lam')

# adam - m
mEen = theano.shared(np.zeros((volsize,nen),dtype=dtype),name='mEen')
mWxien = theano.shared(np.zeros((nen,nen),dtype=dtype), name='mWxien')
mWhien = theano.shared(np.zeros((nen,nen),dtype=dtype), name='mWhien')
mbien = theano.shared(np.zeros((nen,),dtype=dtype), name='mbien')
mWxfen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='mWxfen')
mWhfen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='mWhfen')
mbfen = theano.shared(np.zeros((nen,),dtype=dtype), name='mbfen')
mWxcen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='mWxcen')
mWhcen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='mWhcen')
mbcen = theano.shared(np.zeros((nen,),dtype=dtype),name='mbcen')
mWxoen = theano.shared(np.zeros((nen,nen),dtype=dtype),name='mWxoen')
mWhoen = theano.shared(np.zeros((nen,nen),dtype=dtype),name='mWhoen')
mboen = theano.shared(np.zeros((nen,),dtype=dtype), name='mboen')

mWbeta = theano.shared(np.zeros((nde,nbeta),dtype=dtype), name='mWbeta')
mUbeta = theano.shared(np.zeros((nen,nbeta),dtype=dtype), name='mUbeta')
mvbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='mvbeta')
#
mWzide = theano.shared(np.zeros((nen,nde),dtype=dtype), name='mWzide')
mWzfde = theano.shared(np.zeros((nen,nde),dtype=dtype), name='mWzfde')
mWzcde = theano.shared(np.zeros((nen,nde),dtype=dtype), name='mWzcde')
mWzode = theano.shared(np.zeros((nen,nde),dtype=dtype), name='mWzode')
#
mEde = theano.shared(np.zeros((D,nde),dtype=dtype), name='mEde')
mWxide = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWxide')
mWside = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWside')
mbide = theano.shared(np.zeros((nde,),dtype=dtype), name='mbide')
mWxfde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWxfde')
mWsfde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWsfde')
mbfde = theano.shared(np.zeros((nde,),dtype=dtype), name='mbfde')
mWxcde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWxcde')
mWscde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWscde')
mbcde = theano.shared(np.zeros((nde,),dtype=dtype), name='mbcde')
mWxode = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWxode')
mWsode = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mWsode')
mbode = theano.shared(np.zeros((nde,),dtype=dtype), name='mbode')
#Wsyde = theano.shared(sample_weights(nde,ny), name='Wsyde')
#byde = theano.shared(actfreq, name='byde')
mL0 = theano.shared(np.zeros((nde,ny),dtype=dtype), name='mL0')
mLs = theano.shared(np.zeros((nde,nde),dtype=dtype), name='mLs')
mLz = theano.shared(np.zeros((nen,nde),dtype=dtype), name='mLz')

mparams = [mEen, mWxien, mWhien, mbien, mWxfen, mWhfen, mbfen,
           mWxcen, mWhcen, mbcen, mWxoen, mWhoen, mboen,
           mWbeta, mUbeta, mvbeta,
           mWzide, mWzfde, mWzcde, mWzode,
           mEde, mWxide, mWside, mbide, mWxfde, mWsfde, mbfde,
           mWxcde, mWscde, mbcde, mWxode, mWsode, mbode,
           mL0, mLs, mLz]
# adam - v
vEen = theano.shared(np.zeros((volsize,nen),dtype=dtype),name='vEen')
vWxien = theano.shared(np.zeros((nen,nen),dtype=dtype), name='vWxien')
vWhien = theano.shared(np.zeros((nen,nen),dtype=dtype), name='vWhien')
vbien = theano.shared(np.zeros((nen,),dtype=dtype), name='vbien')
vWxfen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='vWxfen')
vWhfen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='vWhfen')
vbfen = theano.shared(np.zeros((nen,),dtype=dtype), name='vbfen')
vWxcen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='vWxcen')
vWhcen = theano.shared(np.zeros((nen,nen),dtype=dtype), name='vWhcen')
vbcen = theano.shared(np.zeros((nen,),dtype=dtype),name='vbcen')
vWxoen = theano.shared(np.zeros((nen,nen),dtype=dtype),name='vWxoen')
vWhoen = theano.shared(np.zeros((nen,nen),dtype=dtype),name='vWhoen')
vboen = theano.shared(np.zeros((nen,),dtype=dtype), name='vboen')

vWbeta = theano.shared(np.zeros((nde,nbeta),dtype=dtype), name='vWbeta')
vUbeta = theano.shared(np.zeros((nen,nbeta),dtype=dtype), name='vUbeta')
vvbeta = theano.shared(np.zeros((nbeta,),dtype=dtype), name='vvbeta')
#
vWzide = theano.shared(np.zeros((nen,nde),dtype=dtype), name='vWzide')
vWzfde = theano.shared(np.zeros((nen,nde),dtype=dtype), name='vWzfde')
vWzcde = theano.shared(np.zeros((nen,nde),dtype=dtype), name='vWzcde')
vWzode = theano.shared(np.zeros((nen,nde),dtype=dtype), name='vWzode')
#
vEde = theano.shared(np.zeros((D,nde),dtype=dtype), name='vEde')
vWxide = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWxide')
vWside = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWside')
vbide = theano.shared(np.zeros((nde,),dtype=dtype), name='vbide')
vWxfde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWxfde')
vWsfde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWsfde')
vbfde = theano.shared(np.zeros((nde,),dtype=dtype), name='vbfde')
vWxcde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWxcde')
vWscde = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWscde')
vbcde = theano.shared(np.zeros((nde,),dtype=dtype), name='vbcde')
vWxode = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWxode')
vWsode = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vWsode')
vbode = theano.shared(np.zeros((nde,),dtype=dtype), name='vbode')
#Wsyde = theano.shared(sample_weights(nde,ny), name='Wsyde')
#byde = theano.shared(actfreq, name='byde')
vL0 = theano.shared(np.zeros((nde,ny),dtype=dtype), name='vL0')
vLs = theano.shared(np.zeros((nde,nde),dtype=dtype), name='vLs')
vLz = theano.shared(np.zeros((nen,nde),dtype=dtype), name='vLz')

vparams = [vEen, vWxien, vWhien, vbien, vWxfen, vWhfen, vbfen,
           vWxcen, vWhcen, vbcen, vWxoen, vWhoen, vboen,
           vWbeta, vUbeta, vvbeta,
           vWzide, vWzfde, vWzcde, vWzode,
           vEde, vWxide, vWside, vbide, vWxfde, vWsfde, vbfde,
           vWxcde, vWscde, vbcde, vWxode, vWsode, vbode,
           vL0, vLs, vLz]

timestep = theano.shared(np.float32(1))
beta1_t = beta1*(lam**(timestep-1))

print "finish rate"
#TODO: take grads
params = [Een, Wxien, Whien, bien, Wxfen, Whfen, bfen,
          Wxcen, Whcen, bcen, Wxoen, Whoen, boen,
          Wbeta, Ubeta, vbeta,
          Wzide, Wzfde, Wzcde, Wzode,
          Ede, Wxide, Wside, bide, Wxfde, Wsfde, bfde,
          Wxcde, Wscde, bcde, Wxode, Wsode, bode,
          L0, Ls, Lz]

gEen, gWxien, gWhien, gbien, gWxfen, gWhfen, gbfen, \
gWxcen, gWhcen, gbcen, gWxoen, gWhoen, gboen, \
gWbeta, gUbeta, gvbeta, \
gWzide, gWzfde, gWzcde, gWzode, \
gEde, gWxide, gWside, gbide, gWxfde, gWsfde, gbfde, \
gWxcde, gWscde, gbcde, gWxode, gWsode, gbode, \
gL0, gLs, gLz = T.grad(cost, params)

gparams = [gEen, gWxien, gWhien, gbien, gWxfen, gWhfen, gbfen,
           gWxcen, gWhcen, gbcen, gWxoen, gWhoen, gboen,
           gWbeta, gUbeta, gvbeta, gWzide, gWzfde, gWzcde, gWzode,
           gEde, gWxide, gWside, gbide, gWxfde, gWsfde, gbfde,
           gWxcde, gWscde, gbcde, gWxode, gWsode, gbode,
           gL0, gLs, gLz]

print "finish grads"
#TODO: update
updates = []
#updates.append((decreasing_factor, decreasing_factor + increment))
#updates.append((learning_rate, starting_rate * (0.5**( ((decreasing_factor - decreasing_factor%decreasing_interval)/decreasing_interval )-1.0  )) ) )
#updates.append((learning_rate, starting_rate/((decreasing_factor - decreasing_factor%decreasing_interval)/decreasing_interval ) ))
for param, gparam, mparam, vparam in zip(params, gparams, mparams, vparams):
    #updates.append((param, param - gparam*learning_rate))
    newm0 = beta1_t * mparam + (1-beta1_t) * gparam
    newv0 = beta2 * vparam + (1-beta2) * (gparam**2)
    newm = newm0 / (1-(beta1**timestep) )
    newv = newv0 / (1-(beta2**timestep) )
    newparam0 = param - alpha*( newm/(T.sqrt(newv)+eps) )
    #
    #newparam = \
    #newparam0 * (T.clip( (normcap/T.sqrt(T.sum(newparam0**2))) ,np.float32(0.0),np.float32(1.0)))
    #
    updates.append((param, newparam0))
    #updates.append((param, newparam))
    updates.append((mparam, newm0))
    updates.append((vparam, newv0))
updates.append((timestep, timestep+1.0))

print "finish updates"
#TODO: define training function and validation function
learn_model_fn = theano.function(inputs = [X, L, Target],
                                 outputs = cost,
                                 updates = updates)
val_fn = theano.function(inputs = [X, L], outputs = logyvals_de)

print "finish functions"
# now preparation is over
#TODO: functions for main func
# get the correct left and right for this position
def getleftright(direc):
    # direc can be 0 , 90, 180, 270
    left = direc - 90
    if left == -90:
        left = 270
    right = direc + 90
    if right == 360:
        right = 0
    behind = direc + 180
    if behind == 360:
        behind = 0
    elif behind == 450:
        behind = 90
    return left, right, behind

# get data for training functions
def getdata(onedata):
    # get sent
    #instructlist = [w for w in onedata['cleaninstruction'] if w in word2ind]
    instructlist = [w for w in onedata['instruction'] if w in word2ind]
    sent = np.zeros((len(instructlist),volsize),dtype=dtype)
    for i, w in enumerate(instructlist):
        sent[i,:] = vocabmat[:,word2ind[w]]
    # get local features
    local = np.zeros((len(onedata['cleanpath']),D),dtype=dtype)
    for i, pos in enumerate(onedata['cleanpath']):
        thisx = pos[0]
        thisy = pos[1]
        thisdirec = pos[2]
        # find the correct map
        if onedata['map'] == 'Jelly':
            thismap = 1
        elif onedata['map'] == 'L':
            thismap = 2
        elif onedata['map'] == 'Grid':            
            thismap = 0
        else:
            print "there must be one data point nowhere, that is wrong"
        # find this node in map
        somelab = 0
        for j, node in enumerate(maps[thismap]['nodes']):
            # check the coordinates
            if ( node['x']==thisx and node['y']==thisy ):
                # find this node
                somelab += 2
                thisleft, thisright, thisbehind = getleftright(thisdirec)
                # here we need to do things:
                # for node, we keep it as [0,..,1,..,0] one hoc
                # but for wall, we need to distinguish:
                # 1) it is hallway with wall and floor -- one-hoc feat
                # 2) it is WALL and can not go this way -- use -1.0 -- it is not hallway
                nodefeat = np.cast[dtype](node['objvec'])
                forwardfeat = node['capfeat'][thisdirec]
                leftfeat = node['capfeat'][thisleft]
                rightfeat = node['capfeat'][thisright]
                behindfeat = node['capfeat'][thisbehind]
                # now it is over
                thisfeat = np.concatenate((nodefeat,forwardfeat,leftfeat,rightfeat,behindfeat),axis=0)
                #thisfeat = \
                #np.concatenate((node['objvec'],node['nbfeats'][thisdirec],node['nbfeats'][thisleft],node['nbfeats'][thisright]), 
                #               axis=0)
                local[i,:] = np.cast[dtype](thisfeat)
        if somelab > 1:
            somelab += 0 # do thing, since we find it
        else:
            # it is wrong, since we did not find this position in this map
                # did not find it, that is wrong
            print "no such a postion in this map?"
    # get the target actions
    target = np.zeros((len(onedata['action']), ny),dtype=dtype)
    for i, act in enumerate(onedata['action']):
        target[i,:] = np.cast[dtype](act)
    # finish all these
    return sent, local, target

# get the ppl
def getppl(valdata):
    ppl = 0.0
    N = 0.0
    deverr = 0.0
    for i, onedata in enumerate(valdata):
        sent, local, target = getdata(onedata)
        logy = val_fn(sent, local)
        ppl += np.sum(target * logy)
        N += len(target[:,0])
        #deverr += np.sum(np.sum((target * logy),axis=1))
        deverr += np.mean(np.sum((target * logy),axis=1))
    ppl2 = np.exp(-ppl/N)
    deverr = deverr / len(valdata)
    deverr = -1.0*deverr
    return ppl2, deverr
    

#TODO: training
print "start training"
max_epoch = 50
train_errs = np.ndarray(max_epoch)
pplvec = np.ndarray(max_epoch)
deverrvec = np.ndarray(max_epoch) # for error on dev set -- val data
sratevec = np.ndarray(max_epoch)
lratevec = np.ndarray(max_epoch)
saveallseq = {}

trainsize = len(traindata)
valsize = len(valdata)

f = open('log','w')
f.write('This is the training log. It records the training error and the validation perplexity.\n')
f.write('epoch & training err & perplexity & dev err & time in sec \n')
f.close()

print "training"
for epi in range(max_epoch):
    print "training epoch", epi
    start = time.time()
    err = 0.0
    
    for i, onedata in enumerate(traindata):
        sent, local, target = getdata(onedata)
        train_cost = learn_model_fn(sent, local, target)
        err += train_cost
        print "finish this data, ", i, epi
    train_errs[epi] = err / trainsize
    print "finish this epoch ", epi
    print "start validating ", epi
    
    thisppl, thisdeverr = getppl(valdata)
    #
    pplvec[epi] = thisppl
    deverrvec[epi] = thisdeverr
    
    print 'training error', train_errs[epi]
    print 'validation perplexity', pplvec[epi]
    print 'validation error', deverrvec[epi]
    
    end = time.time()
    timetrain = end - start
    #
    # save the model first
    model = {}
    # for encoder
    model['Een'] = Een.get_value()
    model['Wxien'] = Wxien.get_value()
    model['Whien'] = Whien.get_value()
    model['bien'] = bien.get_value()
    model['Wxfen'] = Wxfen.get_value()
    model['Whfen'] = Whfen.get_value()
    model['bfen'] = bfen.get_value()
    model['Wxcen'] = Wxcen.get_value()
    model['Whcen'] = Whcen.get_value()
    model['bcen'] = bcen.get_value()
    model['Wxoen'] = Wxoen.get_value()
    model['Whoen'] = Whoen.get_value()
    model['boen'] = boen.get_value()

    model['c0en'] = c0en.get_value()
    model['h0en'] = h0en.get_value()

    # for decoder
    model['Wbeta'] = Wbeta.get_value()
    model['Ubeta'] = Ubeta.get_value()
    model['vbeta'] = vbeta.get_value()
    #
    model['Wzide'] = Wzide.get_value()
    model['Wzfde'] = Wzfde.get_value()
    model['Wzcde'] = Wzcde.get_value()
    model['Wzode'] = Wzode.get_value()
    #
    model['Ede'] = Ede.get_value()
    model['Wxide'] = Wxide.get_value()
    model['Wside'] = Wside.get_value()
    model['bide'] = bide.get_value()
    model['Wxfde'] = Wxfde.get_value()
    model['Wsfde'] = Wsfde.get_value()
    model['bfde'] = bfde.get_value()
    model['Wxcde'] = Wxcde.get_value()
    model['Wscde'] = Wscde.get_value()
    model['bcde'] = bcde.get_value()
    model['Wxode'] = Wxode.get_value()
    model['Wsode'] = Wsode.get_value()
    model['bode'] = bode.get_value()
    #model['Wsyde'] = Wsyde.get_value()
    #model['byde'] = byde.get_value()
    #
    model['L0'] = L0.get_value()
    model['Ls'] = Ls.get_value()
    model['Lz'] = Lz.get_value()    
    #
    model['dropout'] = dropoutvalue
    #
    model['cs0de'] = cs0de.get_value()
    model['s0de'] = s0de.get_value()
    
    #
    fname = 'model'+str(epi)+'.pickle'
    f = open(fname,'w')
    #f = open('model.pickle','w')
    pickle.dump(model,f)
    f.close()
    # get the final accuracy by greedy search
    slabvec = np.zeros((valsize,),dtype=dtype)
    llabvec = np.zeros((valsize,),dtype=dtype)
    
    #for i, onedata in enumerate(valdata):
    #for i in range(1):
    saveallseq[epi] = {}
    #
    for i in range(len(valdata)):
        onedata = valdata[i]
        nextpos, endpos, strictlab, looselab, outputseq = \
        gs2(model, onedata, maps, vocabmat, word2ind)
        slabvec[i] = strictlab
        llabvec[i] = looselab
        saveallseq[epi][i] = [item for item in outputseq]
        print "finish one sequence ", i, epi
    
    srate = np.mean(slabvec)
    lrate = np.mean(llabvec)
    
    sratevec[epi] = srate
    lratevec[epi] = lrate
    #
    f = open('log','a')
    f.write(str(epi))
    f.write(' & ')
    f.write(str(train_errs[epi]))
    f.write(' & ')
    f.write(str(pplvec[epi]))
    f.write(' & ')
    f.write(str(deverrvec[epi]))
    f.write(' & ')
    f.write(str(sratevec[epi]))
    f.write(' & ')
    f.write(str(lratevec[epi]))
    f.write(' & ')
    f.write(str(timetrain))
    f.write('\n')
    f.close()
    time.sleep(5)

    
    print "finish this epoch"
print "finish training"









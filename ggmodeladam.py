# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:08:16 2015

this is the mini-batch version of previous working Oriol
change is :
1) remove ypro operations
2) initialize the weights of LSTM in a smarter way -- bias terms
3) without peephole
4) do not use image to generate '.'

5) this code has Adam

@author: hongyuan
"""

import pickle
import time
import numpy as np
import theano
from theano import sandbox
import theano.tensor as T
import json
import os
import scipy.io
from collections import defaultdict
import numpy as np
from beamsearch2 import beamsearch2

dtype=theano.config.floatX

# get all the data
f = open('../data.pickle','r')
data = pickle.load(f)
f.close()
ind2word = data['ind2word']
#traindata = data['sorttraindata'] 
traindata = data['traindata'] 
valdata = data['valdata']
allreferences = data['valref']

# prepare for BLEU score
for q in xrange(5):
    open('reference'+`q`,'w').write('\n'.join([x[q] for x in allreferences]))

# perform the training

print "finish reading data"

D = 4096
volsize = 2538
volmat = np.identity(volsize,dtype=dtype)
# squashing of the gates should result in values between 0 and 1
# therefore we use the logistic function
sigma = lambda x: 1 / (1 + T.exp(-x))
# for the other activation function we use the tanh
act = T.tanh
# sequences: s -- can be I or S
# st is the pre-chosen vector -- can be image or 
batchsize = 100
inputbias = np.float32(-2.0)
outputbias = np.float32(-2.0)
forgetbias = np.float32(2.0)
bybias = np.float32(1.0/volsize)
## for training
#TODO: finish the supplementary functions 
# begining step, use image to get h0
def image_step_train(Imat, htm1mat, ctm1mat, 
                     Wcnn, Wxi, Whi, bi, Wxf, Whf, bf, 
                     Wxc, Whc, bc, Wxo, Who, bo, Why, by, forbatch):
    xtmat = theano.dot(Imat, Wcnn)
    itmat = sigma(theano.dot(xtmat,Wxi) + theano.dot(htm1mat,Whi) + T.outer(forbatch,bi) )
    ftmat = sigma(theano.dot(xtmat,Wxf) + theano.dot(htm1mat,Whf) + T.outer(forbatch,bf) )
    ctmat = ftmat * ctm1mat + itmat*act(theano.dot(xtmat,Wxc)+theano.dot(htm1mat,Whc)+T.outer(forbatch,bc) )
    otmat = sigma(theano.dot(xtmat,Wxo) + theano.dot(htm1mat,Who) + T.outer(forbatch,bo) )
    htmat = otmat * act(ctmat)
#    yt = T.concatenate([addzero,tempyt],axis=0)
    return htmat, ctmat
# following step, use h0 and start token to generate words
def word_step_train(stmat, htm1mat, ctm1mat, 
                    We, Wxi, Whi, bi, Wxf, Whf, bf, 
                    Wxc, Whc, bc, Wxo, Who, bo, Why, by, forbatch, fory):
    xtmat = theano.dot(stmat, We)
    itmat = sigma(theano.dot(xtmat,Wxi) + theano.dot(htm1mat,Whi) + T.outer(forbatch,bi) )
    ftmat = sigma(theano.dot(xtmat,Wxf) + theano.dot(htm1mat,Whf) + T.outer(forbatch,bf) )
    ctmat = ftmat * ctm1mat + itmat*act(theano.dot(xtmat,Wxc)+theano.dot(htm1mat,Whc)+T.outer(forbatch,bc) )
    otmat = sigma(theano.dot(xtmat,Wxo) + theano.dot(htm1mat,Who) + T.outer(forbatch,bo) )
    htmat = otmat * act(ctmat)
    
    at0 = theano.dot(htmat,Why) + T.outer(forbatch,by)
    at = at0 - T.outer(T.max(at0,axis=1),fory)
    ytmat = T.exp(at) / T.outer(T.sum(T.exp(at),axis=1),fory)
    logytmat = at - T.log(T.outer(T.sum(T.exp(at),axis=1),fory))
    
#    yt = T.concatenate([addzero,tempyt],axis=0)
    return htmat, ctmat, ytmat, logytmat
## for validation
# use image to get h0    
def image_step_val(Imat, htm1mat, ctm1mat, 
                   Wcnn, Wxi, Whi, bi, Wxf, Whf, bf, 
                   Wxc, Whc, bc, Wxo, Who, bo, Why, by, forbatch):
    xtmat = theano.dot(Imat, Wcnn)
    itmat = sigma(theano.dot(xtmat,Wxi) + theano.dot(htm1mat,Whi) + T.outer(forbatch,bi) )
    ftmat = sigma(theano.dot(xtmat,Wxf) + theano.dot(htm1mat,Whf) + T.outer(forbatch,bf) )
    ctmat = ftmat * ctm1mat + itmat*act(theano.dot(xtmat,Wxc)+theano.dot(htm1mat,Whc)+T.outer(forbatch,bc) )
    otmat = sigma(theano.dot(xtmat,Wxo) + theano.dot(htm1mat,Who) + T.outer(forbatch,bo) )
    htmat = otmat * act(ctmat)
#    yt = T.concatenate([addzero,tempyt],axis=0)
    return htmat, ctmat    
# use h0 and start token to generate words
def word_step_val(stmat, htm1mat, ctm1mat, 
                  We, Wxi, Whi, bi, Wxf, Whf, bf, 
                  Wxc, Whc, bc, Wxo, Who, bo, Why, by, forbatch, fory):
    xtmat = theano.dot(stmat, We)
    itmat = sigma(theano.dot(xtmat,Wxi) + theano.dot(htm1mat,Whi) + T.outer(forbatch,bi) )
    ftmat = sigma(theano.dot(xtmat,Wxf) + theano.dot(htm1mat,Whf) + T.outer(forbatch,bf) )
    ctmat = ftmat * ctm1mat + itmat*act(theano.dot(xtmat,Wxc)+theano.dot(htm1mat,Whc)+T.outer(forbatch,bc) )
    otmat = sigma(theano.dot(xtmat,Wxo) + theano.dot(htm1mat,Who) + T.outer(forbatch,bo) )
    htmat = otmat * act(ctmat)
    
    at0 = theano.dot(htmat,Why) + T.outer(forbatch,by)
    at = at0 - T.outer(T.max(at0,axis=1),fory)
    ytmat = T.exp(at) / T.outer(T.sum(T.exp(at),axis=1),fory)
    logytmat = at - T.log(T.outer(T.sum(T.exp(at),axis=1),fory))
    
#    yt = T.concatenate([addzero,tempyt],axis=0)
    return htmat, ctmat, ytmat, logytmat
###
#TODO: Use a more appropriate initialization method
def sample_weights(numrow, numcol):
    values = np.ndarray([numrow, numcol], dtype=dtype)
    for coli in xrange(numcol):
        vals = np.random.uniform(low=-1., high=1.,  size=(numrow,))
        #vals_norm = np.sqrt((vals**2).sum())
        #vals = vals / vals_norm
        values[:,coli] = vals
    _,svs,_ = np.linalg.svd(values)
    #svs[0] is the largest singular value                      
    values = values / svs[0]
    return values  
    
ns = D + volsize
n = nx = nh = ni = nc = no = nf = 512
ny = volsize

Wcnn = theano.shared(sample_weights(D, n))
We = theano.shared(sample_weights(ny, n))
Wxi = theano.shared(sample_weights(n, n))  
Whi = theano.shared(sample_weights(n, n))  
bi = theano.shared(inputbias * np.ones((n,),dtype=dtype))
#bi = theano.shared(np.cast[dtype](np.random.uniform(low=-0.5,high=0.5,size=(n,))))
Wxf = theano.shared(sample_weights(n, n)) 
Whf = theano.shared(sample_weights(n, n))
bf = theano.shared(forgetbias * np.ones((n,),dtype=dtype))
#bf = theano.shared(np.cast[dtype](np.random.uniform(0, 1.,size = (n,))))
Wxc = theano.shared(sample_weights(n, n))
Whc = theano.shared(sample_weights(n, n))
bc = theano.shared(np.zeros((n,),dtype=dtype))
#bc = theano.shared(np.zeros(n, dtype=dtype))
Wxo = theano.shared(sample_weights(n, n))
Who = theano.shared(sample_weights(n, n))
bo = theano.shared(outputbias * np.ones((n,),dtype=dtype))
#bo = theano.shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = (n,))))
Why = theano.shared(sample_weights(n, ny))
by = theano.shared(bybias * np.ones((ny,),dtype=dtype))
#by = theano.shared(np.zeros(ny, dtype=dtype))
forbatch = theano.shared(np.ones((batchsize,),dtype=dtype))
fory = theano.shared(np.ones((ny,),dtype=dtype))


c0 = theano.shared(np.zeros(n, dtype=dtype))
h0 = theano.shared(np.zeros(n, dtype=dtype))
#h0 = T.tanh(c0)

params = [Wcnn, We, Wxi, Whi, bi, Wxf, Whf, bf, Wxc, Whc, bc, Wxo, Who, bo, Why, by]
#first dimension is alwyas time!!!
# input- v, target = target
img_train = T.matrix(dtype=dtype)
v_train = T.tensor3(dtype = dtype)
target_train = T.tensor3(dtype=dtype)

img_val = T.matrix(dtype=dtype)
v_val = T.tensor3(dtype=dtype)
target_val = T.tensor3(dtype=dtype)
# hidden and outputs of the sequence
print "finish supplementary functions"

#TODO: get the theano function for loss
h0mat_train, c0mat_train = \
image_step_train(img_train, T.outer(forbatch,h0), T.outer(forbatch,c0), 
                 Wcnn, Wxi, Whi, bi, Wxf, Whf, bf, 
                 Wxc, Whc, bc, Wxo, Who, bo, Why, by, forbatch)
#
[hvals_train, cvals_train, yvals_train, logyvals_train], _ = \
theano.scan(fn=word_step_train,
            sequences=dict(input=v_train,taps=[0]),
            outputs_info=[dict(initial=h0mat_train,taps=[-1]),dict(initial=c0mat_train,taps=[-1]),None,None],
            non_sequences=[We, Wxi, Whi, bi, Wxf, Whf, bf, 
                           Wxc, Whc, bc, Wxo, Who, bo, 
                           Why, by, forbatch, fory])
#
cost = -T.mean(T.sum( (target_train * logyvals_train ),[2,0] ) )
# for val
h0mat_val, c0mat_val = \
image_step_val(img_val, T.outer(forbatch,h0), T.outer(forbatch,c0), 
               Wcnn, Wxi, Whi, bi, Wxf, Whf, bf, 
               Wxc, Whc, bc, Wxo, Who, bo, Why, by, forbatch)
#
[hvals_val, cvals_val, yvals_val, logyvals_val], _ = \
theano.scan(fn=word_step_val,
            sequences=dict(input=v_val,taps=[0]),
            outputs_info=[dict(initial=h0mat_val,taps=[-1]), dict(initial=c0mat_val,taps=[-1]), None, None],
            non_sequences=[We, Wxi, Whi, bi, Wxf, Whf, bf, 
                           Wxc, Whc, bc, Wxo, Who, bo, 
                           Why, by, forbatch, fory])
# now we finish the useful theano.functions
# get the updates
print "grads"
gWcnn, gWe, gWxi, gWhi, gbi, gWxf, gWhf, gbf, gWxc, gWhc, gbc, gWxo, gWho, gbo, gWhy, gby = \
T.grad(cost, params)
gparams = [gWcnn, gWe, gWxi, gWhi, gbi, gWxf, gWhf, gbf, gWxc, gWhc, gbc, gWxo, gWho, gbo, gWhy, gby]

print "Adam"
alpha = theano.shared(np.float32(0.001),'alpha')
beta1 = theano.shared(np.float32(0.9),'beta1')
beta2 = theano.shared(np.float32(0.999), 'beta2')
eps = theano.shared(np.float32(0.00000001),'eps')
lam = theano.shared(np.float32(1.0 - 0.00000001), 'lam')
## For Adam
mWcnn = theano.shared(np.zeros((D,nx),dtype=dtype), name='mWcnn')
mWe = theano.shared(np.zeros((ny,nx),dtype=dtype), name='mWe')
mWxi = theano.shared(np.zeros((nx,ni),dtype=dtype), name='mWxi')
mWhi = theano.shared(np.zeros((nh,ni),dtype=dtype), name='mWhi')
mbi = theano.shared( np.zeros((ni,),dtype=dtype), name = 'mbi')
mWxf = theano.shared(np.zeros((nx,nf),dtype=dtype), name='mWxf')
mWhf = theano.shared(np.zeros((nh,nf),dtype=dtype), name='mWhf')
mbf = theano.shared( np.zeros((nf,),dtype=dtype) ,name='mbf')
mWxc = theano.shared(np.zeros((nx,ni),dtype=dtype), name='mWxc')
mWhc = theano.shared(np.zeros((nh,ni),dtype=dtype), name='mWhc')
mbc = theano.shared( np.zeros((ni,),dtype=dtype), name = 'mbc')
mWxo = theano.shared(np.zeros((nx,no),dtype=dtype), name='mWxo')
mWho = theano.shared(np.zeros((nh,no),dtype=dtype), name='mWho')
mbo = theano.shared( np.zeros((no,),dtype=dtype), name='mbo')
mWhy = theano.shared(np.zeros((nh,ny),dtype=dtype), name='mWhy')
mby = theano.shared( np.zeros((ny,),dtype=dtype), name='mby')
ms = [mWcnn, mWe, mWxi, mWhi, mbi, mWxf, mWhf, mbf, 
      mWxc, mWhc, mbc,
      mWxo, mWho, mbo,
      mWhy, mby]
#
vWcnn = theano.shared(np.zeros((D,nx),dtype=dtype), name='vWcnn')
vWe = theano.shared(np.zeros((ny,nx),dtype=dtype), name='vWe')
vWxi = theano.shared(np.zeros((nx,ni),dtype=dtype), name='vWxi')
vWhi = theano.shared(np.zeros((nh,ni),dtype=dtype), name='vWhi')
vbi = theano.shared( np.zeros((ni,),dtype=dtype), name = 'vbi')
vWxf = theano.shared(np.zeros((nx,nf),dtype=dtype), name='vWxf')
vWhf = theano.shared(np.zeros((nh,nf),dtype=dtype), name='vWhf')
vbf = theano.shared( np.zeros((nf,),dtype=dtype) ,name='vbf')
vWxc = theano.shared(np.zeros((nx,ni),dtype=dtype), name='vWxc')
vWhc = theano.shared(np.zeros((nh,ni),dtype=dtype), name='vWhc')
vbc = theano.shared( np.zeros((ni,),dtype=dtype), name = 'vbc')
vWxo = theano.shared(np.zeros((nx,no),dtype=dtype), name='vWxo')
vWho = theano.shared(np.zeros((nh,no),dtype=dtype), name='vWho')
vbo = theano.shared( np.zeros((no,),dtype=dtype), name='vbo')
vWhy = theano.shared(np.zeros((nh,ny),dtype=dtype), name='vWhy')
vby = theano.shared( np.zeros((ny,),dtype=dtype), name='vby')
vs = [vWcnn, vWe, vWxi, vWhi, vbi, vWxf, vWhf, vbf, 
      vWxc, vWhc, vbc,
      vWxo, vWho, vbo,
      vWhy, vby]
#
timestep = theano.shared(np.float32(1))
beta1_t = beta1*(lam**(timestep-1))
print "updates"

#
updates = []

for pi, (param, gparam, mparam, vparam) in enumerate(zip(params, gparams, ms, vs)):
    newm0 = beta1_t * mparam + (1-beta1_t) * gparam
    newv0 = beta2 * vparam + (1-beta2) * (gparam**2)
    newm = newm0 / (1-(beta1**timestep) )
    newv = newv0 / (1-(beta2**timestep) )
    newparam = param - alpha*( newm/(T.sqrt(newv)+eps) )
    updates.append((param, newparam))
    updates.append((mparam, newm0))
    updates.append((vparam, newv0))

updates.append((timestep, timestep+1.0))
#    
print "functions"    
learn_model_fn = theano.function(inputs = [img_train, v_train, target_train],
                                 outputs = cost,
                                 updates = updates)
# for prediction and validation
print "train func"
predictions = theano.function(inputs = [img_val, v_val], outputs = logyvals_val) 
print "finish preparing expressions"
#
def getcube(pairvec, volmat):
    # this function is used to get Imat, vi, vo
    # Imat is image feats, vi, vo is for words    
    veclen = len(pairvec)
    # get maxlen
    maxlen = 0
    for i, pair in enumerate(pairvec):
        if pair['sentlen'] > maxlen:
            maxlen = pair['sentlen']
    #
    batchfeats = np.zeros((veclen, D), dtype=dtype)
    vi = np.zeros(((maxlen+1),veclen, ny),dtype=dtype)      
    vo = np.zeros(((maxlen+1),veclen, ny),dtype=dtype)       
    for i, pair in enumerate(pairvec):
        batchfeats[i,:] = pair['imgvec']
        gind = [ind for ind in pair['sentcode'] if ind > 0]
        sentlen = len(gind)
        vi[0,i,:] = volmat[:,0]
        vo[(sentlen),i,:] = volmat[:,0]
        for j, vecnum in enumerate(gind):
            vi[(j+1), i, :] = volmat[:,vecnum]
            vo[j,i,:] = volmat[:,vecnum]
    return batchfeats, vi, vo
#

def getppl(valdata, volmat):
    ppl = 0 # perplexity
    N = 0 # sum of (length of each sent)
    steps = np.int(len(valdata)/batchsize)    
    deverr = 0.0
    for step in range(steps):
        pairvec = []
        for i in range(batchsize):
            pairvec.append(valdata[step*batchsize+i])
            N += (valdata[step*batchsize+i]['sentlen']+1)
        batchfeats, vi, vo = getcube(pairvec, volmat)
        pred = predictions(batchfeats, vi)
        ppl += np.sum(vo * pred)
        deverr += np.mean( np.sum((np.sum((vo*pred), axis=2)), axis=0) )        
    ppl2 = np.exp(-ppl/N)
    deverr = deverr / steps
    deverr = -1.0*deverr
    # deverr is the same loss as measured in training, just use it for comparison
    return ppl2, deverr


print "start training"
max_epoch = 100
train_errs = np.ndarray(max_epoch)
pplvec = np.ndarray(max_epoch)
deverrvec = np.ndarray(max_epoch)

trainsize = len(traindata)
    #
f = open('oriollog', 'w')
f.write('This is the training log. It records the training error and the validation perplexity.\n')
f.write('epoch & training err & perplexity & dev err & time in sec \n')
f.close()
print "start training"
for epi in range(max_epoch):
    print "training the epoch", epi
    start = time.time()
    err = 0.0
    # randomly sample the order and then get the pair
    #neworder = np.random.permutation(trainsize)
    #
    steps = np.int(trainsize/batchsize)
    for step in range(steps):
        pairvec = []
        for i in range(batchsize):
            pairvec.append(traindata[step*batchsize + i])
        #pairvec = traindata[(step*batchsize):(step*batchsize + batchsize)]
        batchfeats, vi, vo = getcube(pairvec, volmat)
    
        print "get batch outside"
        train_cost = learn_model_fn(batchfeats, vi, vo)
        err += train_cost
        print "finish the batch in the epoch", step, epi
    train_errs[epi] = err / steps
    print "finish the epoch", epi
    # now -- every epoch, we start validating        
    print "validating", epi
        
#        out = {}
#        out['result'] = getppl(valset)
#        pplvec.append(out)
    thisppl, thisdeverr = getppl(valdata, volmat)
    pplvec[epi] = thisppl
    deverrvec[epi] = thisdeverr
    #pplvec[epi] = getppl(valdata, volmat, ns)
        #
    print 'training error', train_errs[epi]
    print 'validation perplexity', pplvec[epi]
    print 'validation error', deverrvec[epi]
        
    end = time.time()
    timetrain = end-start
    #
    f = open('oriollog','a')
    f.write(str(epi))
    f.write(' & ')
    f.write(str(train_errs[epi]))
    f.write(' & ')
    f.write(str(pplvec[epi]))
    f.write(' & ')
    f.write(str(deverrvec[epi]))
    f.write(' & ')
    f.write(str(timetrain))
    f.write('\n')
    f.close()
    time.sleep(5)
    #
    modelparams = {}
    modelparams['Wcnn'] = Wcnn.get_value()
    modelparams['We'] = We.get_value()
    modelparams['Wxi'] = Wxi.get_value()
    modelparams['Whi'] = Whi.get_value()
    modelparams['bi'] = bi.get_value()
    modelparams['Wxf'] = Wxf.get_value()
    modelparams['Whf'] = Whf.get_value()
    modelparams['bf'] = bf.get_value()
    modelparams['Wxc'] = Wxc.get_value()
    modelparams['Whc'] = Whc.get_value()
    modelparams['bc'] = bc.get_value()
    modelparams['Wxo'] = Wxo.get_value()
    modelparams['Who'] = Who.get_value()
    modelparams['bo'] = bo.get_value()
    modelparams['Why'] = Why.get_value()
    modelparams['by'] = by.get_value()
#
    modelparams['train_errors'] = train_errs
    modelparams['ppl_vals'] = pplvec
    modelparams['dev_errs'] = deverrvec
    modelparams['epochnum'] = epi
        # put the variables into a dict
        #save the dict in the folder
    
    
    # get the BLEU score
    # this is sometimes turned off cuz we want to do some simple experiments and want speed up
    allcandidates = []
    for i,item in enumerate(valdata):
        if i%5 == 0:
            sentind = beamsearch2(item['imgvec'], modelparams, volmat, 1)
            candidate = ' '.join([ind2word[ix] for ix in sentind if ix>0])
            allcandidates.append(candidate)
            print "for BLEU, get ", (np.int(i/5))
    open('output','w').write('\n'.join(allcandidates))
    f = open('bleuscore','a')
    f.write(str(epi))
    f.write('\n')
    f.close()
    os.system('./multi-bleu.perl reference < output') 
    
       
    # save the model
    if ( epi%1 == 0 and epi > 0 ):
        sfname = './oriol1/epi'+str(epi)+'.pickle'
        sf = open(sfname,'w+')
        pickle.dump(modelparams, sf)
        sf.close()
    print 'finish this epoch'    
print "finish training"
#    end = time.time()
#    print end - start

#train_model(wholetrain, wholeval, max_epoch, volmat, ns)


# for testing things cuz valset is small
#train_model(valset, valset, max_epoch, volmat, ns)
## here we finish training




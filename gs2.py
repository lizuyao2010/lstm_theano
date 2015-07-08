# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:54:18 2015

this code gets the predicted action sequence

in this code, I want to output the whole inference sequence
and see whether it is reasonable

this is a revised version for drop-out

@author: hongyuan
"""

import numpy as np

import theano
from theano import sandbox
import theano.tensor as T

dtype = theano.config.floatX

def gs2(model, onedata, maps, vocabmat, word2ind):
    
    # for encoder
    #
    dropout = model['dropout']
    #
    Een = model['Een']
    Wxien = model['Wxien']
    Whien = model['Whien']
    bien = model['bien']
    Wxfen = model['Wxfen']
    Whfen = model['Whfen']
    bfen = model['bfen']
    Wxcen = model['Wxcen']
    Whcen = model['Whcen']
    bcen = model['bcen']
    Wxoen = model['Wxoen']
    Whoen = model['Whoen']
    boen = model['boen']

    c0en = model['c0en']
    h0en = model['h0en']

    # for decoder
    Wbeta = model['Wbeta']
    Ubeta = model['Ubeta']
    vbeta = model['vbeta']
    #
    Wzide = model['Wzide']
    Wzfde = model['Wzfde']
    Wzcde = model['Wzcde']
    Wzode = model['Wzode']
    #
    Ede = model['Ede']
    Wxide = model['Wxide']
    Wside = model['Wside']
    bide = model['bide']
    Wxfde = model['Wxfde']
    Wsfde = model['Wsfde']
    bfde = model['bfde']
    Wxcde = model['Wxcde']
    Wscde = model['Wscde']
    bcde = model['bcde']
    Wxode = model['Wxode']
    Wsode = model['Wsode']
    bode = model['bode']
    #Wsyde = model['Wsyde']
    #byde = model['byde']
    #
    L0 = model['L0']
    Ls = model['Ls'] * dropout
    Lz = model['Lz']
    #
    cs0de = model['cs0de']
    s0de = model['s0de']
    
    # prepare internal functions -- encoder and decoder
    def _sigmoid(x):
        return 1 / (1+np.exp(-x))
    
    def _encoder(wordt, htm1, ctm1, 
                 Een, Wxien, Whien, bien, Wxfen, Whfen, bfen, 
                 Wxcen, Whcen, bcen, Wxoen, Whoen, boen):
        xt = np.dot(wordt, Een)
        it = _sigmoid(np.dot(xt,Wxien)+np.dot(htm1,Whien)+bien)
        ft = _sigmoid(np.dot(xt,Wxfen)+np.dot(htm1,Whfen)+bfen)
        ct = ft * ctm1 + it*np.tanh(np.dot(xt,Wxcen)+np.dot(htm1,Whcen)+bcen)
        ot = _sigmoid(np.dot(xt,Wxoen)+np.dot(htm1,Whoen)+boen)
        ht = ot * np.tanh(ct)
        return ht, ct
        
    def _decoder(localt, stm1, cstm1, hmat,
                 Wbeta, Ubeta, vbeta,
                 Wzide, Wzfde, Wzcde, Wzode,
                 Ede, Wxide, Wside, bide, Wxfde, Wsfde, bfde, 
                 Wxcde, Wscde, bcde, Wxode, Wsode, bode,
                 L0, Ls, Lz):
        xt = np.dot(localt, Ede)
        beta = np.dot( np.tanh(np.dot(hmat,Ubeta) + np.dot(stm1,Wbeta) ) ,vbeta)
        alpha = np.exp(beta-np.amax(beta)) / np.sum(np.exp(beta-np.amax(beta)))
        zt = np.dot(alpha, hmat)
        
        it = _sigmoid(np.dot(xt,Wxide)+np.dot(stm1,Wside)+np.dot(zt,Wzide)+bide)
        ft = _sigmoid(np.dot(xt,Wxfde)+np.dot(stm1,Wsfde)+np.dot(zt,Wzfde)+bfde)
        cst = ft * cstm1 + it *np.tanh(np.dot(xt,Wxcde)+np.dot(stm1,Wscde)+np.dot(zt,Wzcde)+bcde)
        ot = _sigmoid(np.dot(xt,Wxode) + np.dot(stm1,Wsode)+np.dot(zt,Wzode)+bode)
        st = ot * np.tanh(cst)
        
        #yt0 = np.dot(st,Wsyde)
        yt0 = np.dot( (xt + np.dot(st, Ls) + np.dot(zt, Lz) ) , L0)
        yt0max = np.amax(yt0)
        yt = np.exp(yt0-yt0max) / np.sum(np.exp(yt0-yt0max))
        logyt = yt0-yt0max-np.log(np.sum(np.exp(yt0-yt0max)))
        
        return st, cst, yt, logyt
    
    # get the necessary things for usage below, like startlocal, correct map, etc
    volsize = len(vocabmat[:,0])
    instructlist = [w for w in onedata['instruction'] if w in word2ind]
    sent = np.zeros((len(instructlist),volsize),dtype=dtype)
    for i, w in enumerate(instructlist):
        sent[i,:] = vocabmat[:,word2ind[w]]
    #
    startpos0 = onedata['cleanpath'][0]
    endpos0 = onedata['cleanpath'][-1]
    
    startpos = np.ndarray((len(startpos0),),dtype=dtype)
    endpos = np.ndarray((len(endpos0),),dtype=dtype)
    
    for i in range( len(startpos) ):
        startpos[i] = startpos0[i]
        endpos[i] = endpos0[i]
    ##
    if onedata['map'] == 'Jelly':
        thismap = 1
    elif onedata['map'] == 'L':
        thismap = 2
    elif onedata['map'] == 'Grid':
        thismap = 0
    else:
        print "one point nowhere and it is in greedy search, must be wrong"
    #
    startx = startpos[0]
    starty = startpos[1]
    startdirec = startpos[2]
    # get local feat for start point
    def _getleftright(direc):
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
    #
    def _getfeat(startx, starty, startdirect, thismap):
        # we can definitely find this position since validation is already done
        # when we get this position
        thisfeat = np.zeros((78,),dtype=dtype)
        somelab = 0
        for j, node in enumerate(maps[thismap]['nodes']):
            if ( node['x']==startx and node['y']==starty ):
                thisleft, thisright, thisbehind = _getleftright(startdirec)
                #
                nodefeat = np.cast[dtype](node['objvec'])
                forwardfeat = node['capfeat'][startdirec]
                leftfeat = node['capfeat'][thisleft]
                rightfeat = node['capfeat'][thisright]
                behindfeat = node['capfeat'][thisbehind]
                #
                thisfeat = np.concatenate((nodefeat,forwardfeat,leftfeat,rightfeat, behindfeat),axis=0)
                #
                #thisfeat = \
                #np.concatenate((node['objvec'],node['nbfeats'][startdirec],node['nbfeats'][thisleft],node['nbfeats'][thisright]), 
                #               axis=0)
                somelab += 2
        local = np.cast[dtype](thisfeat)
        if somelab > 1:
            somelab += 0 # do thing, since we find it
        else:
            # it is wrong, since we did not find this position in this map
                # did not find it, that is wrong
            print "no such a postion in this map? -- while greedy search !!!"
        return local
    # choose the next position
    def _onestep(post, thisdirec):
        nextpos = np.zeros((len(post),), dtype=np.int)
        nextpos[0] = post[0]
        nextpos[1] = post[1]
        nextpos[2] = post[2]
        if thisdirec  == 0:
            nextpos[1] -= 1
        elif thisdirec  == 90:
            nextpos[0] += 1
        elif thisdirec  == 180:
            nextpos[1] += 1
        elif thisdirec  == 270:
            nextpos[0] -= 1
        else:
            print "no valid direction in greedy search?"
        return nextpos
    
    def _step(post, action, thismap):
        actid = np.argmax(action)
        nextpos = np.zeros((3,),dtype=np.int)
        if actid == 1: # turn left -- always possible
            nextpos[0] = post[0]
            nextpos[1] = post[1]
            turnleft, turnright, _ = _getleftright(post[2])
            nextpos[2] = turnleft
        elif actid == 2:
            nextpos[0] = post[0]
            nextpos[1] = post[1]
            turnleft, turnright, _ = _getleftright(post[2])
            nextpos[2] = turnright
        elif actid == 3:
            nextpos[0] = post[0]
            nextpos[1] = post[1]
            nextpos[2] = post[2]
        elif actid == 0:
            # move forward, need to check whether it can go
            nextpos = _onestep(post, post[2])
            canfind = 0
            for j, node in enumerate(maps[thismap]['nodes']):
                if ( node['x']==nextpos[0] and node['y']==nextpos[1]):
                    canfind += 2
            if canfind > 1:
                canfind += 0 # do nothing, since can find it, valid postion
            else:
                # can not find it, invalid position, so choose the second option -- turn or stop
                # change actid
                '''
                nextpos[0] = post[0] # try let it stop
                nextpos[1] = post[1]
                nextpos[2] = post[2]
                '''
                
                actid = np.argmax(action[1:])+1
                if actid == 1: # turn left -- always possible
                    nextpos[0] = post[0]
                    nextpos[1] = post[1]
                    turnleft, turnright, _ = _getleftright(post[2])
                    nextpos[2] = turnleft
                elif actid == 2:
                    nextpos[0] = post[0]
                    nextpos[1] = post[1]
                    turnleft, turnright, _ = _getleftright(post[2])
                    nextpos[2] = turnright
                elif actid == 3:
                    nextpos[0] = post[0] #let it stop
                    nextpos[1] = post[1]
                    nextpos[2] = post[2]
            #else:
                #print "there must be something wrong while stepping"
                
        else:
            print "no such action? wrong!"
        return nextpos
    
    # start inference
    # encode the sent
    sentlen = len(sent[:,0])
    nen = len(Een[0,:])
    hmat = np.zeros((sentlen,nen),dtype=dtype)
    for i in range(sentlen):
        h1en, c1en = _encoder(sent[i,:], h0en, c0en, 
                              Een, Wxien, Whien, bien, Wxfen, Whfen, bfen,
                              Wxcen, Whcen, bcen, Wxoen, Whoen, boen)
        hmat[i,:] = h1en
        h0en = h1en
        c0en = c1en
    # finish encoding, get the hmat
    # for decode, get a function to map action and current state to next state
    # take the first step
    startfeat = _getfeat(startx, starty, startdirec, thismap)
    s1, cs1, y1, logy1 = _decoder(startfeat, s0de, cs0de, hmat,
                                  Wbeta, Ubeta, vbeta,
                                  Wzide, Wzfde, Wzcde, Wzode,
                                  Ede, Wxide, Wside, bide, Wxfde, Wsfde, bfde, 
                                  Wxcde, Wscde, bcde, Wxode, Wsode, bode,
                                  L0, Ls, Lz)
    nextpos = _step(startpos, y1, thismap)
    deltapos = np.sum(np.absolute(nextpos-startpos) )
    # 
    outputseq = []
    
    # continue unitl stop or walk too much, say 30
    count = 0
    while (deltapos > 0.5 and count<30):
        # if this next is still really next step, we save it
        thisseqitem = np.zeros((3,),dtype=dtype)
        thisseqitem[0] = nextpos[0]
        thisseqitem[1] = nextpos[1]
        thisseqitem[2] = nextpos[2]
        outputseq.append(thisseqitem)
        #startpos = nextpos
        startpos[0] = nextpos[0]
        startpos[1] = nextpos[1]
        startpos[2] = nextpos[2]        
        #
        startx = startpos[0]
        starty = startpos[1]
        startdirec = startpos[2]
        startfeat = _getfeat(startx, starty, startdirec, thismap)
        s0de = np.copy(s1)
        cs0de = np.copy(cs1)
        s1, cs1, y1, logy1 = _decoder(startfeat, s0de, cs0de, hmat,
                                      Wbeta, Ubeta, vbeta,
                                      Wzide, Wzfde, Wzcde, Wzode,
                                      Ede, Wxide, Wside, bide, Wxfde, Wsfde, bfde, 
                                      Wxcde, Wscde, bcde, Wxode, Wsode, bode,
                                      L0, Ls, Lz)
        nextpos = _step(startpos, y1, thismap)
        deltapos = np.sum(np.absolute(nextpos-startpos) )
        count += 1
    # now , we stop the iteration and get the final state -- position
    if np.sum(np.absolute(nextpos-endpos)) < 0.5:
        strictlab = 1.
    else:
        strictlab = 0.
    if ( np.absolute(nextpos[0]-endpos[0]) + np.absolute(nextpos[1]-endpos[1]) ) < 0.5:
        looselab = 1.
    else:
        looselab = 0.
    
    return nextpos, endpos, strictlab, looselab, outputseq
    
    
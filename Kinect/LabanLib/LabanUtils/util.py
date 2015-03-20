import combinationsParser as cp
from pybrain.datasets import  SupervisedDataSet
import LabanLib.algorithm.generalExtractor as ge
import numpy as np
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure.modules import BiasUnit
import copy
from sklearn import metrics
from qualityToEmotion import q2e, emotions   
from LabanLib.LabanUtils.negetiveQualities import NQ , disjointQualities, emotionSeparetedQualities

def transform(X, selectedIndices):
    return X[:, selectedIndices]

def getPybrainDataSet(source):
    first = False#True
    qualities, combinations = cp.getCombinations()
    ds = None
    l=0
    i=0
    for emotion in emotions:
        for typeNum in range(1,30):
            for take in range(1,10):
                fileName = 'recordings/'+source+'/'+emotion+'/'+\
                str(typeNum)+'_'+str(take)+'.skl'
                try:
                    data, featuresNames = ge.getFeatureVec(fileName, first)
                    print fileName
                    first = False
                except IOError:
                    continue
                if ds is None:#initialization
                    ds = SupervisedDataSet( len(data), len(qualities) )
                output = np.zeros((len(qualities)))
                for q in combinations[emotion][typeNum]:
                    output[qualities.index(q)] = 1
                ds.appendLinked(data ,  output)
                print i, emotion
                i=i+1
                l+=sum(output)
    #print source, l, len(ds)
    return ds, featuresNames

def constructNet(inLayerSize, hiddenSize, outLayerSize):
    inLayer = LinearLayer(inLayerSize)
    hiddenLayer = SigmoidLayer(hiddenSize)
    outLayer = LinearLayer(outLayerSize)
    n = FeedForwardNetwork()
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    b = BiasUnit()
    n.addModule(b)
    n.addOutputModule(outLayer)
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    b_to_hidden = FullConnection(b, hiddenLayer)
    b_to_out = FullConnection(b, outLayer)
    
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)
    n.addConnection(b_to_hidden)
    n.addConnection(b_to_out)
    
    n.sortModules()
    
    return n, inLayer, hiddenLayer, b, in_to_hidden, b_to_hidden

def fromDStoXY(ds):
    outLayerSize = len(ds.getSample(0)[1])
    X=[]
    Y=[]
    for input, tag in ds:
        X.append(input)
        Y.append(tag)
    return X,Y


def getXYforMultiSet(source):
    ds, featuresNames = getPybrainDataSet(source)
    X, Y = fromDStoXY(ds)
    return X, Y, featuresNames

def getXYfromPybrainDS(ds):
    X=[]
    Y=[]
    for x,y in ds:
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)

def fromXY2DS(X, Y):
    ds=SupervisedDataSet(X.shape[1], Y.shape[1])
    for x,y in zip(X,Y):
        ds.addSample(x, y)
    return ds

def getSplitThreshold(x, y):
    bestSplit = None
    bestF1 = 0
    sortedX= copy.copy(x)
    sortedX.sort()
    splits = []
    for i in range(len(sortedX)-1):
        splits.append((sortedX[i]+sortedX[i+1])/2)
    yNew = [1 if e>0 else 0 for e in y]
    for split in splits:
        newX = [1 if e>=split else 0 for e in x ]
        f1 = metrics.f1_score(newX, yNew)
        if f1 > bestF1:
            bestSplit = split
            bestF1 = f1
    return bestSplit, bestF1

def getSplits(pred, Y):
    splits = []
    qualities, combinations = cp.getCombinations()
    for col in range(pred.shape[1]):
        p = pred[:, col]
        y = Y[:, col]
        bestSplit, bestF1 = getSplitThreshold(p, y)
        splits.append(bestSplit)
    return splits

import copy
def quantisizeBySplits(pred_p, splits):
    pred = copy.copy(pred_p)
    for col in range(pred.shape[1]):
        pred[:, col] = [1 if e>splits[col] else 0 for e in pred[:, col]]
    return np.array(pred)

def accumulatePybrainCMA(CMAs):
    trndatas = None
    for trainSource in CMAs:
        trndata, featuresNames = getPybrainDataSet(trainSource)  
        if trndatas is None:
            trndatas = trndata
        else:
            for s in trndata:
                trndatas.appendLinked(*s)
    return trndatas, featuresNames

def accumulateCMA(CMAs, accumaltor):
    X=[]
    y=[]
    for trainSource in CMAs:
        currX,currY,featuresNames = accumaltor(trainSource)
        X=X+currX
        y=y+currY
    return X, y, featuresNames
        
import os, os.path
def getNonCMAs(nonCMAs, qualities):
    counter = np.zeros((len(qualities)))
    X, Y = [], []
    for nc in nonCMAs:
        dirtocheck = './recordings/'+nc
        for root, _, files in os.walk(dirtocheck):
            for f in files:
                qs = f.split('.')[0]
                qs = qs.split('_')
                y = np.zeros((len(qualities)))
                for q in qs:
                    if q in qualities:
                        y[qualities.index(q)] = 1
                        counter[qualities.index(q)] +=1
                fileName = os.path.join(root, f)
                x, featuresNames = ge.getFeatureVec(fileName, False)
                X.append(x), Y.append(y)
    print counter
    return np.array(X), np.array(Y), counter

from sklearn.feature_selection import f_classif, SelectKBest, f_regression,RFECV
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskElasticNet
def getMultiTaskclassifier(X, Y):
    X, Y = np.array(X), np.array(Y)
    selectedFeaureNum=500
    accum = np.zeros((X.shape[1],))
    for y in np.transpose(Y):
        selector = SelectKBest(f_classif, selectedFeaureNum)
        selector = selector.fit(X, y)
        accum += selector.pvalues_
    selectedIndices = accum.argsort()[:selectedFeaureNum]
    def transform(X):
        return X[:, selectedIndices]     
    X_filtered = transform(X)
    clf = MultiTaskElasticNetCV(normalize=True)
    clf.fit(X_filtered, Y)
    return clf, selectedIndices

def getEmotionsDataset(source):
    X=[]
    y=[]
    i=0
    for emotion in emotions:
        for typeNum in range(1,30):
            for take in range(1,10):
                fileName = 'recordings/'+source+'/'+emotion+'/'+\
                str(typeNum)+'_'+str(take)+'.skl'
                try:
                    data, featuresNames = ge.getFeatureVec(fileName, False)
                    print fileName
                except IOError:
                    continue
                print i, emotion
                i=i+1
                X.append(data)
                y.append(emotions.index(emotion))
    return X, y, featuresNames 

  
def getMultiClassDataset(source):
    X=[]
    Y=[]
    i=0
    print source
    qualities, combinations = cp.getCombinations()
    for emotion in emotions:
        for typeNum in range(1,30):
            for take in range(1,10):
                fileName = 'recordings/'+source+'/'+emotion+'/'+\
                str(typeNum)+'_'+str(take)+'.skl'
                try:
                    data, featuresNames = ge.getFeatureVec(fileName, False)
                    print fileName
                except IOError:
                    continue
                print i, emotion
                i=i+1
                X.append(data)
                output = np.zeros((len(qualities)))
                for q in combinations[emotion][typeNum]:
                    for nq in NQ[q]:
                        output[qualities.index(nq)] = -1
                    output[qualities.index(q)] = 1
                print output
                Y.append(output)
    return X, Y, featuresNames 
    
def classifyMulti(labanClf,X_train_trasform, Y_train,X_test_trasform, Y_test_p):
    Y_test = copy.copy(Y_test_p)
    pred =  labanClf.predict(X_train_trasform)
    splits = getSplits(pred, Y_train)
    ps=[]
    rs=[]
    qualities, _ = cp.getCombinations()
    continuesPred = labanClf.predict(X_test_trasform)
    splits = np.array([v if v>0 else 0 for v in splits])
    print splits
    Pred = quantisizeBySplits(continuesPred, splits)
    Y_test = quantisizeBySplits(Y_test, np.zeros_like(splits))
    print np.array(Pred).shape
    
    
    for j,pred,cPred in zip(range(len(Pred)),Pred,continuesPred):
        for i, q2, v in zip(range(len(qualities)),qualities,pred):
            if v==0:
                continue
            nqs = disjointQualities[q2]
            c=0
            for nq in nqs:
                index = qualities.index(nq)
                if pred[index]==1:
                    c=c+cPred[index]
            if c>cPred[i]:
                Pred[j,i]=0
                print q2, nq
                print q2, cPred[i], c, Y_test[j,i]
            else:
                for nq in nqs:
                    index = qualities.index(nq)
                    if pred[index]==1:
                        Pred[j,index]=0
                        print q2, nq
                        print q2, cPred[i], c, Y_test[j,i]
                        print nq, cPred[index], Y_test[j,index] 
    
                
    for q, pred, y_test in zip(qualities, np.transpose(Pred), np.transpose(Y_test)):
        #y_test = [1 if e>0 else 0 for e in y_test] 
        f = metrics.f1_score(y_test, pred)
        p = metrics.precision_score(y_test, pred)
        r = metrics.recall_score(y_test, pred)
        ps.append(p)
        rs.append(r)
        print q, r, p, f
    print ps
    print np.mean(ps)
    print rs
    print np.mean(rs)
    print 2*np.mean(ps)*np.mean(rs)/(np.mean(rs)+np.mean(ps))
           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
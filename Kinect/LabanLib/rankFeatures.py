from sklearn import linear_model
import numpy as np
import LabanLib.LabanUtils.util as labanUtil
from sklearn import metrics, svm
import matplotlib.pyplot as plt
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.pipeline import Pipeline
from LabanLib.LabanUtils import informationGain as ig 
from sklearn import cross_validation
import copy
import LabanLib.LabanUtils.combinationsParser as cp
from sklearn.feature_selection import f_classif, SelectKBest, f_regression,RFECV
from sklearn.linear_model import MultiTaskLassoCV, \
    MultiTaskElasticNetCV, MultiTaskElasticNet, MultiTaskLasso
from collections import defaultdict
import pickle

#CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen','Michal','Tara']
#ds, featuresNames = labanUtil.accumulatePybrainCMA(CMAs) 
#X, Y = labanUtil.getXYfromPybrainDS(ds)

X = pickle.load( open( "X", "r" ) )
X= np.array(X)
Y = pickle.load( open( "Y_Laban", "r" ) )
Y = np.array(Y)
featuresNames =  pickle.load( open( "featuresNames2", "r" ) )
    
qualities, combinations = cp.getCombinations()


features = open('mixedFeatures.csv', 'w')
features.flush()
features.write('Quality, Feature Name, Operator, Information Gain, p-value\n')

selectedFeaureNum=100
accum = np.zeros((X.shape[1],))
for y in np.transpose(Y):
    selector = SelectKBest(f_classif, selectedFeaureNum)
    selector = selector.fit(X, y)
    accum += selector.pvalues_
selectedIndices = accum.argsort()[:selectedFeaureNum]
def transform(X):
    return X[:, selectedIndices]
X_filtered = transform(X)
featuresNames = [featuresNames[i] for i in selectedIndices]

selector = SelectKBest(ig.infoGain, 5)
for q, y in zip(qualities, np.transpose(Y)):
    selector = selector.fit(X_filtered, y)
    featureNums = np.where(selector.get_support() == True)[0]
    #featureNum = selector.get_support().tolist().index(True)
    print featureNums
    for featureNum in featureNums:
        pstr = str(selector.pvalues_[featureNum])
        pstr = pstr[:3] + pstr[-4:]
        scoreStr = str(round(selector.scores_[featureNum],2))
        features.write(q+', '+ featuresNames[featureNum]+', '+scoreStr+', ' +pstr+ '\n')
features.close()
    
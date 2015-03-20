import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pickle
from sklearn.multiclass import OneVsOneClassifier
from LabanLib.LabanUtils.qualityToEmotion import q2e, emotions
import LabanLib.LabanUtils.combinationsParser as cp
import LabanLib.LabanUtils.util as labanUtil

qualities, combinations = cp.getCombinations()

def transform(X, selectedIndices):
    return X[:, selectedIndices]

X =  np.array(pickle.load( open( "X", "r" ) ))
Y = np.array(pickle.load( open( "Y_Laban_multi", "r" ) ))

labanClfs = pickle.load( open( "labanClfs_multi", "r" ) )
n=len(labanClfs)
for labanClf, selectedIndices, train_index, test_index in labanClfs:
    X_trasform = transform(X, selectedIndices)
    Y_train, Y_test = Y[train_index], Y[test_index]
    X_train, X_test = X[train_index], X[test_index]
    X_train_trasform = transform(X_train, selectedIndices)
    X_test_trasform = transform(X_test, selectedIndices)
    labanUtil.classifyMulti(labanClf,X_train_trasform, Y_train,X_test_trasform, Y_test)

    
    



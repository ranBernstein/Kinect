import pickle
import LabanLib.LabanUtils.util as labanUtil
import numpy as np
import LabanLib.LabanUtils.combinationsParser as cp
from sklearn import metrics

def transform(X, selectedIndices):
    return X[:, selectedIndices]
#X = pickle.load( open( "X", "r" ) )
#y = pickle.load( open( "y", "r" ) )
#X, y = np.array(X), np.array(y)
#Y_Laban = pickle.load( open( "Y_Laban", "r" ) )
#Y_Laban = np.array(Y_Laban)

"""
ds, featuresNames = labanUtil.accumulatePybrainCMA(CMAs) 
X, Y = labanUtil.getXYfromPybrainDS(ds)

filename =  cma+'X'
f= open(filename, 'w')
f.flush()
pickle.dump(X,f)

filename =  cma+'Y'
f= open(filename, 'w')
f.flush()
pickle.dump(Y,f)
"""
qualities, combinations = cp.getCombinations()

labanClf = pickle.load(open( 'labanClf_multi', 'r'))
selectedIndices = pickle.load(open( 'selectedIndices_multi', 'r'))

X = np.array(pickle.load( open( "X", "r" ) ))
X_trasform = transform(X, selectedIndices)
Y_Laban = np.array(pickle.load( open( "Y_Laban_multi", "r" ) ))
pred =  labanClf.predict(X_trasform)
splits = labanUtil.getSplits(pred, Y_Laban)
print splits
CMAs = ['Nirit', 'Hila', 'Irena']
performance = open('Irena_Nirit_Hila.csv', 'w')
performance.flush()
performance.write('Quality, TestSubject, Precision, Recall, F1 score\n')
for cma in CMAs:
    print cma
    X_test = np.array(pickle.load( open( cma+"X", "r" ) ))
    print X_test.shape
    X_test_trasform =  transform(X_test, selectedIndices)
    Y_test = np.array(pickle.load( open( cma+"Y", "r" ) ))
    Pred = labanUtil.quantisizeBySplits(labanClf.predict(X_test_trasform), splits)
    print np.array(Pred).shape
    for q, pred, y_test in zip(qualities, np.transpose(Pred), np.transpose(Y_test)):
        #if q=='Free+Light':
        print cma
        print pred
        print y_test
        pass
        f = metrics.f1_score(y_test, pred)
        p = metrics.precision_score(y_test, pred)
        r = metrics.recall_score(y_test, pred)
        performance.write(q
                          +', '+cma
                          +', '+ str(round(p,3))\
                          +', '+ str(round(r,3))\
                          +', '+ str(round(f, 3))\
                          +'\n')
performance.close()




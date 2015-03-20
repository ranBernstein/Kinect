import pickle
import LabanLib.LabanUtils.util as labanUtil
import numpy as np
import LabanLib.LabanUtils.combinationsParser as cp
from sklearn import metrics
from LabanLib.LabanUtils.negetiveQualities import NQ   

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


ps=[]
rs=[]
X_test = X
print X_test.shape
X_test_trasform =  transform(X_test, selectedIndices)
Y_test = Y_Laban
continuesPred = labanClf.predict(X_test_trasform)
splits = np.array([v if v>0 else 0 for v in splits])
print splits
Pred = labanUtil.quantisizeBySplits(continuesPred, splits)
Y_test = labanUtil.quantisizeBySplits(Y_test, np.zeros_like(splits))
print np.array(Pred).shape

for j,pred,cPred in zip(range(len(Pred)),Pred,continuesPred):
    for i, q2, v in zip(range(len(qualities)),qualities,pred):
        if v==0:
            continue
        nqs = NQ[q2]
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
   






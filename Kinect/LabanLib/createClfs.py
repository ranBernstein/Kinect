import numpy as np
import LabanUtils.util as labanUtil
import pickle
import LabanUtils.combinationsParser as cp

X = pickle.load( open( "X", "r" ) )
Y_laban = np.array(pickle.load( open( "Y_Laban", "r" ) ))
X = np.array(X)
from sklearn import cross_validation
n=3
rs = cross_validation.ShuffleSplit(X.shape[0], n_iter=n, test_size=.1, random_state=0)
data = []
for i,(train_index, test_index) in enumerate(rs):
    print i
    X_train, X_test = X[train_index], X[test_index]
    Y_laban_train, Y_laban_test = Y_laban[train_index], Y_laban[test_index]   
    labanClf, selectedIndices = labanUtil.getMultiTaskclassifier(X_train, Y_laban_train)
    data.append((labanClf, selectedIndices, train_index, test_index))
f = open('labanClfs', 'w')
f.flush()
pickle.dump(data, f)
f.close()
    


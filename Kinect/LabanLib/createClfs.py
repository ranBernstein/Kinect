import numpy as np
import LabanUtils.util as labanUtil
import pickle
import LabanUtils.combinationsParser as cp

X = pickle.load( open( "X", "r" ) )
y = pickle.load( open( "y", "r" ) )
selectedIndices = pickle.load(open( 'selectedIndices', 'r'))
#labanClf = pickle.load(open( 'labanClf', 'r'))
def transform(X, selectedIndices):
    return X[:, selectedIndices]
Y_laban = pickle.load( open( "Y_Laban", "r" ) )
X, y = np.array(X), np.array(y)
from sklearn import cross_validation
n=7
rs = cross_validation.ShuffleSplit(X.shape[0], n_iter=n, test_size=.1, random_state=0)
data = []
for i,(train_index, test_index) in enumerate(rs):
    print i
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Y_laban_train, Y_laban_test = Y_laban[train_index], Y_laban[test_index]   
    labanClf, selectedIndices = labanUtil.getMultiTaskclassifier(X_train, Y_laban_train)
    data.append((labanClf, selectedIndices, train_index, test_index))
f = open('labanClfs', 'w')
f.flush()
pickle.dump(data, f)
f.close()
    


import numpy as np
#from sklearn.ensemble import AdaBoostClassifier
import pickle
#from sklearn.multiclass import OneVsOneClassifier
#from LabanLib.LabanUtils.qualityToEmotion import q2e, emotions
import LabanLib.LabanUtils.combinationsParser as cp
import LabanLib.LabanUtils.util as labanUtil
#from LabanLib.createClfs import Y_laban
qualities, combinations = cp.getCombinations()
CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen','Michal','Tara']




X,Y_Laban, featuresNames = labanUtil.accumulateCMA(CMAs, labanUtil.getMultiClassDataset) 
"""
fy = open('X', 'w')
fy.flush()
pickle.dump(X, fy)
fy.close()

fy = open('Y_Laban_multi2', 'w')
fy.flush()
pickle.dump(Y_Laban, fy)
fy.close()

fy = open('featuresNames', 'w')
fy.flush()
pickle.dump(Y_Laban, fy)
fy.close()
"""


"""
X = np.array(pickle.load( open( "X", "r" ) ))
print X.shape
Y_Laban = np.array(pickle.load( open( "Y_Laban_multi2", "r" ) ))
labanClf, selectedIndices = labanUtil.getMultiTaskclassifier(X, Y_Laban)
f = open('labanClf_multi2', 'w')
f.flush()
pickle.dump(labanClf, f)
f.close()

f = open('selectedIndices_multi2', 'w')
f.flush()
pickle.dump(selectedIndices, f)
f.close()
"""


labanClf = pickle.load(open( 'labanClf', 'r'))
selectedIndices = pickle.load(open( 'selectedIndices', 'r'))
X = np.array(pickle.load( open( "X", "r" ) ))
X_trasform = labanUtil.transform(X, selectedIndices)
Y = np.array(pickle.load( open( "Y_Laban", "r" ) ))
labanUtil.classifyMulti(labanClf,X_trasform, Y,X_trasform, Y)





   









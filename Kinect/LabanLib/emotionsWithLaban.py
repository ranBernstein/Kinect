import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pickle
from sklearn.multiclass import OneVsOneClassifier
from LabanLib.LabanUtils.qualityToEmotion import q2e, emotions
import LabanLib.LabanUtils.combinationsParser as cp
import LabanLib.LabanUtils.util as labanUtil
qualities, combinations = cp.getCombinations()
CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen','Michal','Tara']

def transform(X, selectedIndices):
    return X[:, selectedIndices]

def getEmotionFromLaban(labanClf, x):
    labans = labanClf.predict(x)
    accum = [[],[],[],[]]
    for  q,v in zip(qualities,labans):
        es = q2e[q][1]
        emotion = q2e[q][0]
        accum[emotions.index(emotion)].append(es*v)
    grades = []
    for l in accum:
        grades.append(np.mean(l))
    m  = np.argmax(grades)
    return m
"""
X,y, featuresNames = labanUtil.accumulateCMA(CMAs, labanUtil.getEmotionsDataset) 
fX = open('y', 'w')
fX.flush()
pickle.dump(y, fX)
fX.close()

fX = open('X', 'w')
fX.flush()
pickle.dump(X, fX)
fX.close()

ds, featuresNames = labanUtil.accumulatePybrainCMA(CMAs) 
X, Y_Laban = labanUtil.getXYfromPybrainDS(ds)
fy = open('Y_Laban', 'w')
fy.flush()
pickle.dump(Y_Laban, fy)
fy.close()
"""
X = pickle.load( open( "X", "r" ) )
y = pickle.load( open( "y", "r" ) )
X, y = np.array(X), np.array(y)
Y_Laban = pickle.load( open( "Y_Laban", "r" ) )
Y_Laban = np.array(Y_Laban)

"""
from sklearn import cross_validation
n=1
rs = cross_validation.ShuffleSplit(X.shape[0], n_iter=n, test_size=.1, random_state=0)
labanClfs = []
for train_index, test_index in rs:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Y_laban_train, Y_laban_test = Y_Laban[train_index], Y_Laban[test_index]   
    labanClf, selectedIndices = labanUtil.getMultiTaskclassifier(X_train, Y_laban_train)
    labanClfs.append((labanClf, selectedIndices, train_index, test_index))
f = open('labanClfs', 'w')
f.flush()
pickle.dump(labanClfs, f)
f.close()
"""


#selectedIndices = pickle.load(open( 'selectedIndices', 'r'))
#labanClf = pickle.load(open( 'labanClf', 'r'))
print 'after pickles'
baseClf = AdaBoostClassifier()
clf = OneVsOneClassifier(baseClf)
resRaw =[]
resRawTransform = []
resLaban = []
resEmotion =[]
resTable=[]
resLabanEmotion = []
resAll =[]
labanClfs = pickle.load( open( "labanClfs", "r" ) )
n=len(labanClfs)
for labanClf, selectedIndices, train_index, test_index in labanClfs:
    X_trasform = transform(X, selectedIndices)
    y_train, y_test = y[train_index], y[test_index]
    Y_laban_train, Y_laban_test = Y_Laban[train_index], Y_Laban[test_index] 
    y_table=[]
    X_test_low = []
    X_transform_laban = []
    X_transform_emotion = []
    X_transform_laban_emotion = []
    X_all=[]
    for i,x in enumerate(X_trasform):
        em = getEmotionFromLaban(labanClf, x)
        true_emotion = y[i]
        if em!=true_emotion:
            print 'y_laban', Y_Laban[i]
            print 'em', em
            print 'true_emotion', true_emotion
            print 'i', i
        
        y_table.append(em)
        xl = x.tolist()
        xl.append(em)
        X_transform_emotion.append(xl)
        labans = labanClf.predict(x)
        newVec = np.concatenate((x, labans))
        X_transform_laban.append(newVec)
        newVec = np.concatenate((newVec, [em]))
        X_transform_laban_emotion.append(newVec)
        rawX = X[i].tolist()
        rawX=rawX+labans.tolist()+[em]
        X_all.append(rawX)
    X_transform_laban = np.array(X_transform_laban)
    X_transform_laban_emotion = np.array(X_transform_laban_emotion)
    X_transform_emotion = np.array(X_transform_emotion)
    y_table=np.array(y_table)
    X_all=np.array(X_all)
    y_test_table = np.array(y_table[test_index])
    
    diff = np.nonzero(y_test-y_test_table)[0]
    diffSize = len(diff)
    resTable.append(1-float(diffSize)/len(y_test))
    
    X_train, X_test = X[train_index], X[test_index]
    clf.fit(X_train, y_train)
    r = clf.score(X_test, y_test)
    resRaw.append(r)

    X_train_trasform, X_test_trasform = X_trasform[train_index], X_trasform[test_index]
    clf.fit(X_train_trasform, y_train)
    r = clf.score(X_test_trasform, y_test)
    resRawTransform.append(r)
    
    X_train_transform_laban, X_test_transform_laban = X_transform_laban[train_index], X_transform_laban[test_index]
    clf.fit(X_train_transform_laban, y_train)
    r = clf.score(X_test_transform_laban, y_test)
    
    resLaban.append(r)
    clf.fit(X_transform_emotion[train_index], y_train)
    r = clf.score(X_transform_emotion[test_index], y_test)
    resEmotion.append(r)
    
    clf.fit(X_transform_laban_emotion[train_index], y_train)
    r = clf.score(X_transform_laban_emotion[test_index], y_test)
    resLabanEmotion.append(r)
    
    clf.fit(X_all[train_index], y_train)
    r = clf.score(X_all[test_index], y_test)
    resAll.append(r)
    
print 'clf', str(clf)
print 'n_iter', n

print 'resTable', np.mean(resTable)
print resTable

print 'resRaw', np.mean(resRaw)
print resRaw

print 'resRawTransform', np.mean(resRawTransform)
print resRawTransform

print 'resLaban', np.mean(resLaban)
print resLaban

print 'resEmotion', np.mean(resEmotion)
print resEmotion

print 'resLabanEmotion', np.mean(resLabanEmotion)
print resLabanEmotion

print 'resAll', np.mean(resAll)
print resAll


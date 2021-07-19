from numpy import *
import operator

def uspsclass(tr_fea,tr_label,ts_fea,ts_label):
    accuracycount = 0.0
    mtest = len(ts_fea)
    for i in range(mtest):
        classifierresult = classify(ts_fea[i],tr_fea,tr_label)
        # print("the kNN-classifier came back with:%d,the real is:%d" % (classifierresult,ts_label[i]))
        if(classifierresult==ts_label[i]):accuracycount+=1.0
    data = open("./result_python.txt", 'a')
    print("\nmost_NN's total number of accuracy is: %d" % accuracycount, file=data)
    print("\nmost_NN's total accuracy rate is: %f" % (accuracycount/float(mtest)), file=data)
    data.close()

def classify(inx,dataset,labels):
    datasetsize = dataset.shape[0]
    diffmat = tile(inx,(datasetsize,1))-dataset
    sqdiffmat = diffmat**2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances**0.5
    sorteddistindicies = distances.argsort()
    classcount = {}
    for i in range(200):
        voteilabel = labels[sorteddistindicies[i]]
        classcount[voteilabel[0]] = classcount.get(voteilabel[0],0)+1
    sortedclasscount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]
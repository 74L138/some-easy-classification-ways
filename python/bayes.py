from numpy import *
import numpy as np


def classify_bayes(tr_fea,tr_label,ts_fea,ts_label):
    prediction = []
    accur = 0.0
    mtest = len(ts_fea)
    for i in range(200):
       test = ts_fea[i]
       class_rate=[]
       for j in range(1,11):
           class_is_j_index = np.where(tr_label[:200]==j)[0]
           j_rate = len(class_is_j_index)/len(tr_label)
           class_is_j_x = np.array([tr_fea[x] for x in class_is_j_index])
           for k in range(256):
               j_rate *=len([item for item in class_is_j_x if np.fabs(item[k]-test[k]) <0.06])*1.0 / len(class_is_j_x)
           class_rate.append(j_rate)
       prediction.append(np.argmax(class_rate))
       # print('the bayes-classifier came back with:',prediction[-1]+1,'the real is:',ts_label[i])
       if prediction[-1]+1==ts_label[i]:
           accur+=1.0
    accurancy = accur / mtest
    data = open("./result_python.txt", 'a')
    print("\nbayes's total number of accuracy is: %d" % accur, file=data)
    print("\nbayes's total accuracy rate is: %f" % accurancy, file=data)
    data.close()


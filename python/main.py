import scipy.io as scio
import kNN
import bayes
import most_NN
file = open("./result_python.txt", 'w').close()

dataFile = './data/usps.mat'
data = scio.loadmat(dataFile)
# print(data.keys())
tr_fea = data['tr_fea'].T
tr_label = data['tr_label']
# print("Size of training data = {}".format(len(tr_fea)))
# print("Size of training data = {}".format(len(tr_label)))
ts_fea = data['ts_fea'].T
ts_label = data['ts_label']
# print("Size of test data = {}".format(len(ts_fea)))


kNN.uspsclass(tr_fea,tr_label,ts_fea,ts_label)
bayes.classify_bayes(tr_fea,tr_label,ts_fea,ts_label)
most_NN.uspsclass(tr_fea,tr_label,ts_fea,ts_label)

import sys
from scipy.misc import imsave, imread, imresize
import numpy as np
from numpy import genfromtxt
from math import isnan
import matplotlib.pyplot as plt
import os
import theano
import random


def get_lables(field, data):
	# get dignose result labels
	# para : field-attributes field, data-2darray
	# return : labels
	label_index = field.index('Glauocma_good_psn_2011$')
	labels = data[1:, label_index]
	return labels


def replace_lable_nan(labels):
	# replace the nan with 0
	# para: labels with nan
	# return: labels with nan replaced with 0
	for x in range(labels.shape[0]):
		if isnan(labels[x]) or labels[x] == -1:
			labels[x] = 0
	return labels


def get_pos_data(data):
	pos_data = []
	for x in range(len(data[:, 0])):
		if data[x, 0] == 1:
			pos_data.extend(list(data[x, 1:]))
	return pos_data


def get_neg_data(data):
	neg_data = []
	for x in range(len(data[:, 0])):
		if data[x, 0] == 0:
			neg_data.extend(list(data[x, 1:]))
	return neg_data


def visualizeFeat(data, lable):
	m = len(data[0])
	n = len(data[:, 0])
	for x in range(n):
		if lable[x] == 1:
			plt.plot(range(m), data[x], 'ro')
		elif lable[x] == 0:
			plt.plot(range(m), data[x], 'go')
	plt.show()


def ndarray2list(nda):
	# transform a n-d-array into a list
	# para: nda-n-dimension-array
	# return: a list from nda
	res = []
	for x in nda:
		res.append(list(x))
	print(len(res))
	return res


def training_file(lables, data):
	# create a training file
	fp = open('training_file', 'w')
	for (x, y) in zip(lables, data):
		ind = 1
		if x == 1:
			fp.write(str(-1) + " ")
			for w in y:
				fp.write(str(ind) + ':')
				fp.write(str(w.real))
				fp.write(' ')
				ind += 1
		else:
			fp.write(str(+1) + " ")
			for w in y:
				fp.write(str(ind) + ':')
				fp.write(str(w.real))
				fp.write(' ')
				ind += 1
		fp.write('\n')


def randomlize(data, lable):
	# randomlize the data and lables for training
	# para: data-data of rnflt, lable-original lables for data
	# return: randomlized data  and the corespond lables
	rand_list = random.sample(range(lable.shape[0]), lable.shape[0])
	res_data = np.zeros_like(data)
	res_lable = np.zeros_like(lable)
	index = 0
	for x in rand_list:
		res_data[index] = data[x]
		res_lable[index] = lable[x]
		index += 1
	return res_data, res_lable


def to_picture(data, lable):
	os.chdir('C:\\Users\\wwj\\Desktop\\FirstAttempt\\picture')
	for i in range(lable.shape[0]):
		x = data[i]
		y = lable[i]
		imsave(str(i) + '-' + str(y) + '.bmp', x.reshape(32, 24))
	os.chdir('C:\\Users\\wwj\\Desktop\\FirstAttempt')


def load_data(dataset='fine_tune_back_up.txt'):
	f = file("traindata.npy", "rb")
	train_datas = np.load(f)
	train_lables = np.load(f)
	valid_datas = np.load(f)
	valid_lables = np.load(f)
        train_datas = np.vstack((train_datas, valid_datas[1000:2000]))
        train_lables = np.array(train_lables.tolist() + valid_lables[1000:2000].tolist())
        print(train_lables.shape)
	test_datas = valid_datas[2000:5400]
	test_lables = valid_lables[2000:5400]
	valid_datas = valid_datas[:1000]
	valid_lables = valid_lables[:1000]
	f.close()
	print(train_lables[:10])

	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y), borrow=borrow)
		return shared_x, shared_y
	train_set_x, train_set_y = shared_dataset((train_datas, train_lables))
	valid_set_x, valid_set_y = shared_dataset((valid_datas, valid_lables))
	test_set_x, test_set_y = shared_dataset((test_datas, test_lables))
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
	return rval
	'''
	data = genfromtxt(dataset, delimiter=' ', dtype=int)
	pos_data = np.asarray(get_pos_data(data))
	neg_data = np.asarray(get_neg_data(data))
	pos_data = pos_data.reshape(pos_data.shape[0] / 768, 768)
	neg_data = neg_data.reshape(neg_data.shape[0] / 768, 768)
	lables = np.asarray(list(data[:, 0]), dtype=int)
	data = data[:, 1:]
	train_datas = np.vstack((neg_data[:2000], pos_data[:200]))
	train_datas = (train_datas - np.mean(train_datas, axis=0)) / np.var(train_datas, axis=0)
	train_lables = np.array(list(np.zeros(2000, dtype=int)) + list(np.ones(200, dtype=int)))
	train_datas, train_lables = randomlize(train_datas, train_lables)
	valid_datas = data[4000:6200]
	valid_lables = lables[4000:6200]
	valid_datas = (valid_datas - np.mean(valid_datas, axis=0)) / np.var(valid_datas, axis=0)
    #valid_datas = train_datas
    #valid_lables = train_lables
	test_datas = data[4000:6200]
	test_datas = (test_datas - np.mean(test_datas, axis=0)) / np.var(test_datas, axis=0)
	test_lables = lables[4000:6200]
	'''
	'''
	pos_data = genfromtxt('pos_samples.txt', delimiter=' ', dtype=int)
	neg_data = genfromtxt('neg_samples.txt', delimiter=' ', dtype=int)
	'''
	#plt.plot(range(768), np.mean(pos_data, axis=0), 'ro')
	#plt.plot(range(768), np.mean(neg_data, axis=0), 'go')

	'''
	train_datas = np.vstack((neg_data[:2000], pos_data[:300]))
	train_datas = (train_datas - np.mean(train_datas, axis=0)) / np.var(train_datas, axis=0)
	train_lables = np.array(list(np.zeros(2000, dtype=int)) + list(np.ones(300, dtype=int)))
	train_datas, train_lables = randomlize(train_datas, train_lables)
	'''
	'''
	valid_datas = np.vstack((neg_data[300:], pos_data[300:]))
	f = file("mean_var.npy", "wb")
	np.save(f, np.mean(valid_datas, axis=0))
	np.save(f, np.var(valid_datas, axis=0))
	f.close()
	'''
	'''
	valid_datas = (valid_datas - np.mean(valid_datas, axis=0)) / np.var(valid_datas, axis=0)
	valid_lables = np.array(list(np.zeros(neg_data[300:].shape[0], dtype=int)) + list(np.ones(pos_data[300:].shape[0], dtype=int)))
	valid_datas, valid_lables = randomlize(valid_datas, valid_lables)
	print(valid_datas.shape)
	test_datas = valid_datas
	test_lables = valid_lables
	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
		return shared_x, theano.tensor.cast(shared_y, 'int32')
	train_set_x, train_set_y = shared_dataset((train_datas, train_lables))
	valid_set_x, valid_set_y = shared_dataset((valid_datas, valid_lables))
	test_set_x, test_set_y = shared_dataset((test_datas, test_lables))
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
	return rval
	'''

'''
# get valid data with valid lables

field = None
data = genfromtxt('C:\\Users\\wwj\\Desktop\\FirstAttempt\\RNFLT.csv', delimiter=',', dtype=int)
reader = csv.reader(open('C:\\Users\\wwj\\Desktop\\FirstAttempt\\RNFLT.csv'))
for line in reader:
	field = line
	break

lables = get_lables(field, data)

f = open('fine_tune_back_up.txt', 'w')
valid = True
ind = 0
for x in range(len(data[:, 0])):
	if ind >= 6555:
		break
	tmp = data[ind + 1, 52:]
	if lables[ind] != 1 and lables[ind] != 0:
		ind += 1
		continue
	for e in tmp:
		if e < 0 or e > 255:
			valid = False
			break
	if valid is False:
		ind += 1
		valid = True
	else:
		f.write(str(lables[ind]) + ' ')
		for e in tmp:
			f.write(str(e))
			f.write(' ')
		f.write('\n')
		ind += 1
'''

'''
data = genfromtxt('C:\\Users\wwj\\Desktop\\FirstAttempt\\fine_tune_back_up.txt', delimiter=' ', dtype=float)
lables = data[:, 0]
pos_data = np.asarray(get_pos_data(data))
neg_data = np.asarray(get_neg_data(data))
pos_data = pos_data.reshape(pos_data.shape[0] / 768, 768)
neg_data = neg_data.reshape(neg_data.shape[0] / 768, 768)
print(neg_data.shape)
print(pos_data.shape)
plt.plot(range(768), np.mean(pos_data, axis=0), 'ro')
plt.plot(range(768), np.mean(neg_data, axis=0), 'g-')
plt.show()
data = data[:, 1:]
print(data)
train_datas = np.vstack((neg_data[:200], pos_data[:200]))
train_lables = np.vstack((np.zeros((200, 1), dtype=int), np.ones((200, 1), dtype=int)))
to_picture(train_datas, train_lables)

valid_datas = data[5000:6300]
valid_lables = lables[5000:6300]

cov = np.cov(train_datas, rowvar=0)
print(cov.shape)
eigvalue, eigvector = np.linalg.eig(cov)
index_list = np.argsort(eigvalue)[::-1]
redvectors = eigvector[:, index_list[0:6]]
print(redvectors.shape)

train_datas = np.dot(train_datas, redvectors)
training_file(train_lables, train_datas)

valid_datas = np.dot(valid_datas, redvectors)
visualizeFeat(np.vstack((train_datas[:50], train_datas[200:250])), np.vstack((train_lables[:50], train_lables[200:250])))

prob = svm_problem(list(train_lables), ndarray2list(train_datas))
# prob = svm_problem(pos_label, ndarray2list(positive_sample))
para = svm_parameter('-t 0')
m = svm_train(prob, para)
t = svm_predict(list(valid_lables), ndarray2list(valid_datas), m)
p_lable = t[0]
print(len(p_lable))
print(list(np.array(p_lable)))
'''
load_data()

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


def load_data():
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
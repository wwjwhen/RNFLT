# RNFLT
CNN for classification and localization

Raw data is not supplied for privacy. Just codes and processed and normalized data.

The model is modified from Theano teaching example-LeNet, in respect for Lecun, I didn't change the name of my model.

The model is mainly organized in LeNet.py. The format.py is used for reading data. And w4.npy is fine-tuned parameters(weights of each layer). The traindata.npy includes train, valid, test data and labels(do not be misled by its name).

The programme is designed for runing on GPU devices, yet cpu also does without.

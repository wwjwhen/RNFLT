from __future__ import print_function
import os
import sys
import timeit
import numpy
from numpy import genfromtxt
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from logistic_sgd import LogisticRegression
from format import load_data
from mlp import HiddenLayer
from sklearn.metrics import roc_curve, auc


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(1, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(15. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.maximum(0, pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input



def evaluate_lenet5(learning_rate=0.01, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 20, 20, 20, 50], batch_size=100):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient) 


    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(None)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print('the dataset type is', type(datasets[0][0][0]))
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.lvector('y')  # the labels are presented as 1D vector of
    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 1, 768))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 1, 768),
        filter_shape=(nkerns[0], 1, 1, 5),
        poolsize=(1, 2)
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 1, 382),
        filter_shape=(nkerns[1], nkerns[0], 1, 5),
        poolsize=(1, 2)
    )

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 1, 189),
        filter_shape=(nkerns[2], nkerns[1], 1, 4),
        poolsize=(1, 2))

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 1, 93),
        filter_shape=(nkerns[3], nkerns[2], 1, 4),
        poolsize=(1, 2))

    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], 1, 45),
        filter_shape=(nkerns[4], nkerns[3], 1, 4),
        poolsize=(1, 2))
    
    
    layer45 = LeNetConvPoolLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, 50, 1, 21),
        filter_shape=(50, 50, 1, 4),
        poolsize=(1, 2))
    '''
    layer452 = LeNetConvPoolLayer(
        rng,
        input=layer451.output,
        image_shape=(batch_size, 50, 1, 9),
        filter_shape=(50, 50, 1, 4),
        poolsize=(1, 2))
    '''
    # here we expect to reduce two conv-pool layers
    layer5_input = layer4.output.flatten(2)
    
    
    layer5 = HiddenLayer(
        rng,
        input=layer5_input,
        n_in=nkerns[4] * 1 * 21,
        n_out=50,
        activation=T.tanh)
    # for dropout
    srng = T.shared_randomstreams.RandomStreams(rng.randint(99999))

    mask = srng.binomial(n=1, p=0.5, size=layer5.output.shape)
   
    layer6 = LogisticRegression(input=layer5.output, n_in=50, n_out=2)
    #layer6a = LogisticRegression(input=layer5.output * 0.5, n_in=50, n_out=2)
    
    '''
    f = file('w1.npy', 'wb')
    numpy.save(f, layer0.W.get_value())
    numpy.save(f, layer1.W.get_value())
    numpy.save(f, layer2.W.get_value())
    numpy.save(f, layer3.W.get_value())
    numpy.save(f, layer4.W.get_value())
    numpy.save(f, layer5.W.get_value())
    numpy.save(f, layer6.W.get_value())
    print(layer0.W.get_value())
    f.close()
    '''
    
    f = file('w4.npy', 'rb')
    layer0.W.set_value(numpy.load(f))
    layer1.W.set_value(numpy.load(f))
    layer2.W.set_value(numpy.load(f))
    layer3.W.set_value(numpy.load(f))
    layer4.W.set_value(numpy.load(f))
    layer5.W.set_value(numpy.load(f))
    layer6.W.set_value(numpy.load(f))
    
    # the cost we minimize during training is the NLL of the model
    cost = layer6.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer6.params + layer5.params + layer4.params +  layer3.params + layer2.params + layer1.params + layer0.params
    # params = layer6.params + layer5.params + layer2.params + layer1.params + layer0.params
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        [cost, layer6.errors(y)],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_loss = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 25000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    best_paras = []
    test_score = 0.
    start_time = timeit.default_timer()

    lab_valid = theano.function(
        [index],
        layer6.y_pred,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    lab_test = theano.function(
        [index],
        layer6.y_pred,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    prob_test = theano.function(
        [index],
        layer6.p_y_given_x,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    num = T.lscalar("num")

    #gradX = T.grad(layer6.p_y_given_x[num][1], x)
    gradX = T.grad(layer6.p_sum, x)

    grad_x = theano.function(inputs=[num, index], outputs=gradX,
        givens={
            x: test_set_x[index * batch_size : (index + 1) * batch_size]
        },
        on_unused_input='ignore'
    )

    get_layer1_out = theano.function(inputs=[index], outputs=[layer1.output],
        givens={
            x : test_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )

    get_layer2_out = theano.function(inputs=[index], outputs=[layer2.output],
        givens={
            x : test_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )

    get_layer3_out = theano.function(inputs=[index], outputs=[layer3.output],
        givens={
            x : test_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )

    get_layer4_out = theano.function(inputs=[index], outputs=[layer4.output],
        givens={
            x : test_set_x[index * batch_size : (index + 1) * batch_size]
        }
    )

    ftv = file('train_valid5.txt', 'wb')
    epoch = 0
    done_looping = False
    print('the patience is ', patience)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            costij, gradient = train_model(minibatch_index)
            #print("the cost is " + str(costij))
            #print(gradient[0])
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                #layer6a.W.set_value(layer6.W.get_value())
                #layer6a.b.set_value(layer6.b.get_value())
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                train_losses = [train_loss(i) for i in range(n_train_batches)]
                this_train_loss = numpy.mean(train_losses)
                ftv.write(str(this_train_loss) + ' ' + str(this_validation_loss) + "\n")
                print('epoch %i, minibatch %i/%i, validation error %f %%, train error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100., this_train_loss * 100.))
                '''
		print(lab(valid_set_x.get_value()[0:batch_size].reshape((batch_size, 1, 1, 768))))
                '''
                print(lab_valid(0))
                print(type(valid_set_y))
	        #print(valid_set_y.get_value()[0:batch_size])
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough

                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
		    best_paras = [layer6.params[0].get_value(), layer6.params[1].get_value(), layer5.params[0].get_value(), layer5.params[1].get_value(), layer4.params[0].get_value(), layer4.params[1].get_value(), layer3.params[0].get_value(), layer3.params[1].get_value(), layer2.params[0].get_value(), layer2.params[1].get_value(), layer1.params[0].get_value(), layer1.params[1].get_value(), layer0.params[0].get_value(), layer0.params[1].get_value(), layer45.params[0].get_value(), layer45.params[1].get_value()]
                    #print(layer6.params[0].get_value()) 
                    # test it on the test set
                    #layer6a.W.set_value(layer6.W.get_value())
                    #layer6a.b.set_value(layer6.b.get_value())
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
            print('done_loop is ', done_looping)
            print('epoch and iter are: ' + str(epoch) + ' ' + str(iter))

    
    #print(layer6.params[0].get_value())
    #print(best_paras[0])
    
    layer6.params[0].set_value(best_paras[0])
    layer6.params[1].set_value(best_paras[1])
    layer5.params[0].set_value(best_paras[2])
    layer5.params[1].set_value(best_paras[3])
    layer4.params[0].set_value(best_paras[4])
    layer4.params[1].set_value(best_paras[5])
    layer3.params[0].set_value(best_paras[6])
    layer3.params[1].set_value(best_paras[7])
    layer2.params[0].set_value(best_paras[8])
    layer2.params[1].set_value(best_paras[9])
    layer1.params[0].set_value(best_paras[10])
    layer1.params[1].set_value(best_paras[11])
    layer0.params[0].set_value(best_paras[12])
    layer0.params[1].set_value(best_paras[13])
    layer45.params[0].set_value(best_paras[14])
    layer45.params[1].set_value(best_paras[15])
    
    f = file("traindata.npy", "rb")
    train_datas = numpy.load(f)
    train_labels = numpy.load(f)
    test_datas = numpy.load(f)[1000:5400]
    test_labels = numpy.load(f)[1000:5400]
    #test_datas = train_datas
    #test_labels = train_labels
    datas = test_datas[:100]
    print(datas.shape)
    

    def smooth(grad):
        grad = grad_x(0, 0)
        print(grad)
        print(grad.shape)
        for i in range(0, 768 - 3, 1):
            grad[:, i] = (grad[:, i] + grad[:, i + 1] + grad[:, i + 2] + grad[:, i + 3]) / 4
        print(grad)
        return grad
    
    X = theano.shared(value=datas, borrow=True)
    
    updates_x = [(X, X - 0.001 * gradX)]

    restore_x = theano.function(
        inputs=[x],
        outputs=layer6.p_y_given_x[61][1],
        #outputs=layer0.input,
        updates=updates_x,
        allow_input_downcast=True
    )
    
    fr = file("revised.txt", "wb")
    #grad_value = theano.function(inputs=[num, x], outputs=[gradX])

    for ind in range(n_test_batches):
        X.set_value(test_datas[ind * batch_size: (ind + 1) * batch_size])
        iter = 1000
        while iter > 0:
            p = restore_x(X.get_value())
            if iter % 100==0 and ind ==0:
                print(p)
                #print(grad_x(0, 0)[0])
            iter -= 1
        for i in range(batch_size):
            for x in (X.get_value()[i]):
                fr.write(str(x) + " ")
            fr.write('\n')
        print(X.get_value().shape)
        #break
    fr.close()

    # get the grad of one batch_size samples
    
    f = open("grad.txt", "wb")
    for x in range(batch_size):
        for y in grad_x(0, 0)[x]:
            f.write(str(y) + " ")
        f.write("\n")
   
    '''
    f1 = open("feat1.txt", "w")
    f2 = open("feat2.txt", "w")
    f3 = open("feat3.txt", "w")
    f4 = open("feat4.txt", "w")

    for ind in range(10):
        for x in range(batch_size):
            for y in get_layer1_out(ind)[0][x][0][0]:
                f1.write(str(y) + " ")
            f1.write("\n")
        for x in range(batch_size):
            for y in get_layer2_out(ind)[0][x][0][0]:
                f2.write(str(y) + " ")
            f2.write("\n")
        for x in range(batch_size):
            for y in get_layer3_out(ind)[0][x][0][0]:
                f3.write(str(y) + " ")
            f3.write("\n")
        for x in range(batch_size):
            for y in get_layer4_out(ind)[0][x][0][0]:
                f4.write(str(y) + " ")
            f4.write("\n")
    '''

    #print(grad_x(1, 0)[0][1])
    #print(layer6.params[0].get_value())
    #print(lab(test_set_x.get_value()[0:600].reshape((600, 1, 1, 768))))
    test_labels = [lab_test(i) for i in range(6)]
    print(numpy.array(test_labels).flatten())
    fp = open('test_result.txt', 'wb')
    test_labels = [lab_test(i) for i in range(n_test_batches)]
    for x in test_labels:
        fp.write(str(x) + ' ')
    print(test_set_y.get_value()[0:600])
    #prob = theano.function(inputs=[layer0.input], outputs=[layer6.p_y_given_x])
    print(test_set_y.get_value().shape)
    #pro = prob(test_set_x.get_value().reshape((5446, 1, 1, 768)))
    print("prob in index 0 is")
    print(prob_test(0)[:, 1])
    pro = [prob_test(i) for i in range(n_test_batches)]
    #print(numpy.array(pro[0]))
    pro = numpy.array(pro)
    pro = pro.reshape(4400, 2)
    print(pro.shape)

    fpr, tpr, thres = roc_curve(test_set_y.get_value()[:4400], pro[:, 1], pos_label=1)
    print("mean and std are:")
    print(numpy.mean(pro[:, 0]))
    print(numpy.var(pro[:, 0]))
    print(numpy.mean(pro[:, 1]))
    print(numpy.var(pro[:, 1]))
    fp = open('fpr_tpr.txt', 'w')
    for x in fpr:
        fp.write(str(x) + " ")
    fp.write('\n')
    for x in tpr:
        fp.write(str(x) + " ")
    fp.write('\n')
    for x in thres:
        fp.write(str(x) + ' ')
    print(auc(fpr, tpr))
    
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    print(gradX)
if __name__ == '__main__':
    evaluate_lenet5()

'''
Created on Mar 5, 2014

@author: sstober
'''
import os;
import numpy;
import theano;
import warnings;

import gzip;
import cPickle;
from sklearn import cross_validation;
import logging;

import argparse;
from config import Config;
import pylearn2.utils.logger;

import librosa;  # pip install librosa

def reset_pylearn2_logging(level=logging.WARN):
    logging.info('resetting pylearn2 logging');
    # this reset the whole pylearn2 logging system -> full control handed over    
    pylearn2.utils.logger.restore_defaults();
    logging.getLogger("pylearn2").setLevel(level)

def load_config(default_config='train_sda.cfg', reset_logging=True):
    parser = argparse.ArgumentParser(description='Train a Stacked Denoising Autoencoder');
    parser.add_argument('-c', '--config', help='specify a configuration file');
    parser.add_argument('-v', '--verbose', help='increase output verbosity", action="store_true');    
    args = parser.parse_args();
    
    if args.config == None:
        configfile = default_config; # load default config
    else:
        configfile = args.config;
    config = Config(file(configfile));  
    
    if args.verbose or config.logger.level=='debug':
        loglevel = logging.DEBUG;
    else:
        loglevel = logging.INFO;
    
    logging.basicConfig(format=config.logger.pattern, level=loglevel);    

    if reset_logging or config.get('reset_logging', True):
        reset_pylearn2_logging();
        
    logging.info('using config {0}'.format(configfile));
    
    # disable annoying deprecation warnings
    warnings.simplefilter('once', UserWarning)
    warnings.simplefilter('default')
    
    return config;

def splitdata(dataset, ptest = 0.1, pvalid = 0.1):
    
    data, labels = dataset;       
    ptrain = 1-ptest-pvalid;
    
    data_train, data_temp, labels_train, labels_temp = \
        cross_validation.train_test_split(data, labels, test_size=(1-ptrain), random_state=42);
    data_valid, data_test, labels_valid, labels_test = \
        cross_validation.train_test_split(data_temp, labels_temp, test_size=ptest/(pvalid+ptest), random_state=42);
        
    train_set = (data_train, labels_train);
    valid_set = (data_valid, labels_valid);
    test_set = (data_test, labels_test);
    
    logging.info('Split data into {0} train, {1} validation, {2} test: {3} {4} {5}'.format(ptrain, pvalid, ptest, data_train.shape, data_valid.shape, data_test.shape));
    
    return train_set, valid_set, test_set;

def load(filepath):
    
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return cPickle.load(f)
    elif filepath.endswith('.pkl.gz') or filepath.endswith('.pklz'):
        with gzip.open(filepath, 'rb') as f:
            return cPickle.load(f)
    else:
        raise 'File format not supported for {}'.format(filepath);


def save(filepath, data):
    
    if filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            cPickle.dump(data, f)
    elif filepath.endswith('.pkl.gz') or filepath.endswith('.pklz'):
        with gzip.open(filepath, 'wb') as f:
            cPickle.dump(data, f)
    else:
        raise 'File format not supported for {}'.format(filepath);
    
    
#     # Load the dataset
#     f = gzip.open(dataset, 'rb')
#     train_set, valid_set, test_set = cPickle.load(f)
#     f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    
    
def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
#         shared_y = theano.shared(numpy.asarray(data_y,
#                                                dtype=theano.config.floatX),
#                                  borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x #, T.cast(shared_y, 'int32')


    return shared_dataset(train_set)

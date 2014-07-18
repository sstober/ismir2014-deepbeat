'''
Created on Apr 3, 2014

@author: sstober
'''

import os;
import time

from pylearn2.utils.timing import log_timing


import logging;
log = logging.getLogger(__name__);

import numpy;
import theano;

from pylearn2.training_algorithms.sgd import SGD;
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError;
from pylearn2.termination_criteria import EpochCounter;
from pylearn2.datasets.transformer_dataset import TransformerDataset;
from pylearn2.blocks import StackedBlocks;
from pylearn2.train import Train;
from pylearn2.utils import serial;

# from pylearn2.training_algorithms.learning_rule import AdaDelta

from pylearn2.corruption import BinomialCorruptor;

from deepbeat.pylearn2ext.util import LoggingCorruptor, LoggingCallback;
from deepbeat.pylearn2ext import AdaptableDenoisingAutoencoder, StackedDenoisingAutoencoder;

    
def create_denoising_autoencoder(structure, corruption=0.1, act='tanh'):    
    n_input, n_output = structure;
    curruptor = LoggingCorruptor(
                                 BinomialCorruptor(corruption_level=corruption), 
                                 name='{}'.format(structure));
    
    irange = numpy.sqrt(6. / (n_input + n_output));            
    if act == theano.tensor.nnet.sigmoid or act == 'sigmoid':
        irange *= 4;
    
#     log.debug('initial weight range: {}'.format(irange));
    
    config = {
        'corruptor': curruptor,
        'nhid': n_output,
        'nvis': n_input,
        'tied_weights': True,
        'act_enc': act,
        'act_dec': act,
        'irange': irange, #0.001,
    }
    log.debug('creating denoising autoencoder {}'.format(config));
    
    da = AdaptableDenoisingAutoencoder(**config);
    return da;

def get_layer_trainer_sgd_autoencoder(
                                      layer, 
                                      trainset,
                                      batch_size = 10,
                                      monitoring_batches = 5,
                                      learning_rate=0.1, 
                                      max_epochs=100, 
                                      name=''):    
    # configs on sgd
    train_algo = SGD(
            learning_rate = learning_rate, # 0.1,
#             learning_rule = AdaDelta(),
              cost =  MeanSquaredReconstructionError(),
              batch_size =  10,
              monitoring_batches = 5,
              monitoring_dataset =  trainset,
              termination_criterion = EpochCounter(max_epochs=max_epochs),
              update_callbacks = None
              )
    
    log_callback = LoggingCallback(name);
    
    return Train(model = layer,
            algorithm = train_algo,
            extensions = [log_callback],
            dataset = trainset)
    


def train_SdA(config, dataset):
    ## load config
    hidden_layers_sizes = config.get('hidden_layers_sizes', [10, 10]);
    corruption_levels = config.get('corruption_levels', [0.1, 0.2]);
    stage2_corruption_levels = config.get('stage2_corruption_levels', [0.1, 0.1]);

    pretrain_epochs = config.get('pretrain_epochs', 10);
    pretrain_lr = config.get('pretrain_learning_rate', 0.001);

    finetune_epochs = config.get('finetune_epochs', 10);
    finetune_lr = config.get('finetune_learning_rate', 0.01);
    
    batch_size = config.get('batch_size', 10);
    monitoring_batches = config.get('monitoring_batches', 5);
    
    output_path = config.get('output_path', './');

    
    input_trainset = dataset;
    design_matrix = input_trainset.get_design_matrix();
#     print design_matrix.shape;
    n_input = design_matrix.shape[1];
    log.info('done');
    
    log.debug('input dimensions : {0}'.format(n_input));
    log.debug('training examples: {0}'.format(design_matrix.shape[0]));

    # numpy random generator
#     numpy_rng = numpy.random.RandomState(89677)

    log.info('... building the model');

    # build layers
    layer_dims = [n_input];
    layer_dims.extend(hidden_layers_sizes);
        
    layers = [];
    for i in xrange(1, len(layer_dims)):
        structure = [layer_dims[i-1], layer_dims[i]];
        layers.append(create_denoising_autoencoder(structure, corruption=corruption_levels[i-1]));
    
    # unsupervised pre-training
    log.info('... pre-training the model');
    start_time = time.clock();    
    
    for i in xrange(len(layers)):
        # reset corruption to make sure input is not corrupted
        for layer in layers:
            layer.set_corruption_level(0);
            
        if i == 0:
            trainset = input_trainset;
        elif i == 1:
            trainset = TransformerDataset( raw = input_trainset, transformer = layers[0] );
        else:
            trainset = TransformerDataset( raw = input_trainset, transformer = StackedBlocks( layers[0:i] ));
            
        # set corruption for layer to train
        layers[i].set_corruption_level(corruption_levels[i]);
            
        trainer = get_layer_trainer_sgd_autoencoder(
                        layers[i], 
                        trainset,
                        learning_rate=pretrain_lr,
                        max_epochs=pretrain_epochs,
                        batch_size=batch_size,
                        monitoring_batches=monitoring_batches,
                        name='pre-train'+str(i));
        
        log.info('unsupervised training layer %d, %s '%(i, layers[i].__class__));
        trainer.main_loop();
        
#         theano.printing.pydotprint_variables(
#                                      layer_trainer.algorithm.sgd_update.maker.fgraph.outputs[0],
#                                      outfile='pylearn2-sgd_update.png',
#                                      var_with_name_simple=True);
        
    end_time = time.clock();
    log.info('pre-training code ran for {0:.2f}m'.format((end_time - start_time) / 60.));
        
    # now untie the decoder weights
    log.info('untying decoder weights');
    for layer in layers:
        layer.untie_weights();
    
    # construct multi-layer training fuctions
    
    # unsupervised training
    log.info('... training the model');
    
    sdae = None;
    for depth in xrange(1, len(layers)+1):
        first_layer_i = len(layers)-depth;
        log.debug('training layers {}..{}'.format(first_layer_i,len(layers)-1));

        group = layers[first_layer_i:len(layers)];
#         log.debug(group);
        
        # reset corruption 
        for layer in layers:
            layer.set_corruption_level(0);
                
        if first_layer_i == 0:
            trainset = input_trainset;
        elif first_layer_i == 1:
            trainset = TransformerDataset( raw = input_trainset, transformer = layers[0] );
        else:
            trainset = TransformerDataset( raw = input_trainset, transformer = StackedBlocks( layers[0:first_layer_i] ));
            
        # set corruption for input layer of stack to train
#         layers[first_layer_i].set_corruption_level(stage2_corruption_levels[first_layer_i]);

        corruptor = LoggingCorruptor(
                                     BinomialCorruptor(corruption_level=stage2_corruption_levels[first_layer_i]),
                                     name='depth {}'.format(depth));
        sdae = StackedDenoisingAutoencoder(group, corruptor);      
             
        trainer = get_layer_trainer_sgd_autoencoder(
                                    sdae,
                                    trainset, 
                                    learning_rate=finetune_lr,
                                    max_epochs=finetune_epochs,
                                    batch_size=batch_size,
                                    monitoring_batches=monitoring_batches,
                                    name='multi-train'+str(depth)
                                    );
                                    
        log.info('unsupervised multi-layer training %d'%(i));        
        trainer.main_loop()
    
    end_time = time.clock()
    log.info('full training code ran for {0:.2f}m'.format((end_time - start_time) / 60.));        
    
    # save the model
    model_file = os.path.join(output_path, 'sdae-model.pkl'); 
    with log_timing(log, 'saving SDA model to {}'.format(model_file)):
        serial.save(model_file, sdae);

    # TODO: pylearn2.train_extensions.best_params.KeepBestParams(model, cost, monitoring_dataset, batch_size)
    # pylearn2.train_extensions.best_params.MonitorBasedSaveBest

    log.info('done');
    
    return sdae;

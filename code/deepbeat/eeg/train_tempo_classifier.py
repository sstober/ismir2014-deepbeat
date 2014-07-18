'''
Created on Apr 2, 2014

@author: sstober
'''
import os;

import logging;
log = logging.getLogger(__name__);

from EEGDatasetLoader import EEGDatasetLoader;
from deepbeat.util import load_config;
from deepbeat.pylearn2ext import train_SdA;
from deepbeat.pylearn2ext.util import LoggingCallback, ClassificationLoggingCallback;

from pylearn2.utils import serial;
from pylearn2.utils.timing import log_timing


from pylearn2.models.mlp import PretrainedLayer, Softmax, MLP;
from pylearn2.costs.mlp import Default;

from pylearn2.training_algorithms.sgd import SGD, ExponentialDecay, MomentumAdjustor;
from pylearn2.termination_criteria import EpochCounter, And, MonitorBased;
from pylearn2.train import Train;

import theano;
import theano.tensor as T;

from sklearn.metrics import confusion_matrix,classification_report;

def train_mlp(trainset, testset, config):
#     output_path = config.get('output_path');
#     sda_file = os.path.join(output_path, 'sdae-model.pkl');
    
    sda_file = config.get('sda_file'); 
    mlp_file = config.get('mlp_file'); 
    
    if not os.path.isfile(sda_file):
        log.info('no SDA model found at {}. re-computing'.format(sda_file));
        train_SdA(config, trainset);

    # load model    
    with log_timing(log, 'loading SDA model from {}'.format(sda_file)):
        sda = serial.load(sda_file);

    ## construct MLP
    mlp_layers = [];
    for i, ae in enumerate(sda.autoencoders):
        mlp_layers.append(PretrainedLayer(
                                   layer_name='ae{}'.format(i),
                                   layer_content=ae
                                   ));
    mlp_layers.append(Softmax(
                             max_col_norm=1.9365,
                             layer_name='y',
                             n_classes=2,
                             irange= .005
                             ));                    
    mlp = MLP(layers=mlp_layers, input_space=sda.input_space);
                
    max_epochs = 1000;
    mlp_train_algo = SGD(
                    learning_rate = 0.001,
#                     init_momentum = .5,
#                    learning_rule = AdaDelta(),
#                     cost = Default,
                    batch_size =  1000,
                    monitoring_batches = 10,
                    monitoring_dataset = {'train': trainset, 'test' : testset},
                    termination_criterion = And(
                            criteria=[
                                    EpochCounter(max_epochs=max_epochs),
#                                     MonitorBased(
#                                         channel_name = "valid_y_misclass",
#                                         prop_decrease = 0.,
#                                         N = 100
#                                    )
                                      ]),                    
#                     update_callbacks = ExponentialDecay(
#                                                         decay_factor = 1.00004,
#                                                         min_lr = .000001
#                                                         )
                    );
    
    mlp_trainer = Train(
                        model = mlp,
                        algorithm = mlp_train_algo,
#                         extensions = [],
                        extensions = [
                                    LoggingCallback('mlp', obj_channel='train_objective'),
                                    ClassificationLoggingCallback(trainset, mlp, header='train'),
                                    ClassificationLoggingCallback(testset, mlp, header='test'),
#                                       MomentumAdjustor(
#                                                        start = 1,
#                                                        saturate = 250,
#                                                        final_momentum = .7
#                                                     )
                                    ],
                        dataset = trainset);

    with log_timing(log, 'training MLP'):    
        mlp_trainer.main_loop();
    
    # save the model
    
    with log_timing(log, 'saving SDA model to {}'.format(mlp_file)):
        serial.save(mlp_file, mlp);
    
    log.info('done');


def train(config):
    
    
    subject_groups = [[0,  9],  # 180 + 240
                      [1, 10],
                      [2, 11],
                      [6,  3],
                      [7,  4],
                      [8,  5],
                      [   12]   # no 180 for this one                   
                      ];
    
    train_subjects = sum(subject_groups[0:4], []); # uses overloaded + operator for list
    log.info('train subjects: {}'.format(train_subjects));
    test_subjects = subject_groups[5];
    log.info('test subjects: {}'.format(test_subjects));
    
    config = config.eeg;
    dataset_root = config.get('dataset_root', './');
    dataset_suffix = config.get('dataset_suffix', '');
    
    datasets = EEGDatasetLoader(dataset_root, suffix=dataset_suffix, label_mode='rhythm_type')
#     datasets.load_data(mask=[0,1]); # FIXME: remove mask
#     datasets.load_data();

    config = config.tempo;
    frame_size = config.get('frame_size', 100);
    hop_size = config.get('hop_size', 10);        
    
    trainset, testset, validset = datasets.split_dataset(
                                    #subjects=xrange(12), # all but last subject
                                    subjects=[0], # only 1st subject
				    frame_size=frame_size, 
                                    hop_size=hop_size, 
                                    p_test=0.2, p_valid=0);
    
#     trainset = datasets.get_train_dataset(subjects=train_subjects, frame_size=frame_size, hop_size=hop_size);
    log.info('train dataset loaded. X={} y={}'.format(trainset.X.shape, trainset.y.shape));
#     testset = datasets.get_train_dataset(subjects=test_subjects, frame_size=frame_size, hop_size=hop_size);
    log.info('test dataset loaded. X={} y={}'.format(testset.X.shape, testset.y.shape));
    
    
#     output_path = config.get('output_path');
#     mlp_file = os.path.join(output_path, 'mlp-model.pkl');
    mlp_file = config.get('mlp_file'); 
    
    if not os.path.isfile(mlp_file):
        train_mlp(trainset, testset, config);
    
    # load model    
    with log_timing(log, 'loading MLP model from {}'.format(mlp_file)):
        mlp = serial.load(mlp_file);
    
    
    
    
    
    y_true = trainset.labels;
#     y_pred = mlp.fprop(trainset.X);
    
    
    X = mlp.get_input_space().make_theano_batch()
    Y = mlp.fprop( X )
    Y = T.argmax( Y, axis = 1 )
    f = theano.function( [X], Y )
    y_pred = f( trainset.X );
    
    # Compute confusion matrix
    print classification_report(y_true, y_pred);
    print confusion_matrix(y_true, y_pred);

    return mlp;

if __name__ == '__main__':
    config = load_config(default_config='../train_sda.cfg', reset_logging=False);

    train(config);

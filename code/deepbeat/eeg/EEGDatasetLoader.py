'''
Created on Apr 2, 2014

@author: sstober
'''
import os;
import glob;
import math;

import logging;
log = logging.getLogger(__name__);

import numpy as np;

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix;
from pylearn2.utils.timing import log_timing
from pylearn2.utils.one_hot import one_hot;

import librosa;  # pip install librosa

from deepbeat.util import load;
from deepbeat.eeg.LabelConverter import LabelConverter;


class EEGDatasetLoader(object):
    '''
    classdocs
    
    TODO: use label_mode
    '''

    def __init__(self, path, suffix='', label_mode='tempo'):
        '''
        Constructor
        '''
        
        self.datafiles = [];
        subject_paths = glob.glob(os.path.join(path, 'Sub*'));
        for path in subject_paths:
            dataset_filename = os.path.join(path, 'dataset'+suffix+'.pklz');
            if os.path.isfile(dataset_filename):   
                log.debug('addding {}'.format(dataset_filename));
                self.datafiles.append(dataset_filename);
            else:
                log.warn('file does not exists {}'.format(dataset_filename));                
        self.datafiles.sort();
        
        self.label_mode = label_mode;
        
        self.label_converter = LabelConverter();
        
#     def load_data(self, mask=None):
#         if mask is None:
#             mask = np.arange(0, len(self.datafiles));
#             
#         log.debug('selected datasets: {}'.format(mask));
#             
#         self.datasets = [];
#         with log_timing(log, 'loading data from {} files'.format(len(self.datafiles))):    
#             for i in mask:
#                 self.datasets.append(load(self.datafiles[i]));
#                 log.debug('dataset {}: {}'.format(i,self.datasets[i][0].shape));
                
                
    def _split_sequences(self, sequences, sequence_labels, frame_length, hop_length):
        log.debug('splitting sequences {} with labels {} into {}-frames with hop={}'.format(
                            len(sequences), sequence_labels.shape, frame_length, hop_length));
        data = [];
        labels = [];
        for i in xrange(len(sequences)):
            sequence = sequences[i];
            label = sequence_labels[i];
        
            frames = librosa.util.frame(sequence, frame_length=frame_length, hop_length=hop_length);
            frames = np.transpose(frames);
        
            frame_labels = [];
            for i in range(0, frames.shape[0]):
                frame_labels.append(label);
#             frame_labels = np.vstack(frame_labels); # causes factor 5 slowdown
            
            data.append(frames);
            labels.append(frame_labels);
            
        data = np.vstack(data);
        labels = np.vstack(labels);
        
        log.debug('generated frames {} with labels {}'.format(data.shape, labels.shape));
        return data, labels;
        
    def get_train_dataset(self, subjects, frame_size=72, hop_size=6):    
        return self.split_dataset(self, subjects, frame_size, hop_size, 0, 0)[0];
        
    def split_dataset(self, subjects, frame_size, hop_size, p_test=0, p_valid=0):
        with log_timing(log, 'generating dataset for subjects {} frame_legth={} hop_size={}'.format(subjects,frame_size,hop_size)):
            X_train = [];
            X_test  = []; 
            X_valid = [];
            y_train = [];
            y_test  = []; 
            y_valid = [];
            
            assert p_test + p_valid < 1;
            
            p_train = 1.0 - p_test - p_valid;
            log.info('spliting into {:.2f}% train, {:.2f}% test, {:.2f}% validation'.format(p_train*100, p_test*100, p_valid*100));
            
            for i in xrange(len(self.datafiles)):
                if i in subjects:
                    with log_timing(log, 'loading data from {}'.format(self.datafiles[i])): 
                        sequences, labels = load(self.datafiles[i]); 
            
                        train_sequences = [];
                        test_sequences = [];
                        valid_sequences = [];
                                    
                        # Note: assuming all sequences have the same length
                        seq_len = len(sequences[0]);
                        train_len = int(math.ceil(seq_len * p_train));
                        valid_len = int(math.ceil(seq_len * p_valid));
                        valid_len = min(valid_len, seq_len - train_len);
                        
                        # split sequences only - no need to split labels
                        for seq in sequences:
                            train_sequences.append(seq[0:train_len]);
                            if p_test > 0: test_sequences.append(seq[train_len:seq_len-valid_len]);
                            if p_valid > 0: valid_sequences.append(seq[seq_len-valid_len:seq_len]); 
                                            
                        train_sequences, train_labels = self._split_sequences(train_sequences, labels, frame_size, hop_size);                    
                        X_train.append(train_sequences);
                        y_train.append(train_labels);
                        
                        if p_test > 0:
                            test_sequences, test_labels = self._split_sequences(test_sequences, labels, frame_size, hop_size);                    
                            X_test.append(test_sequences);
                            y_test.append(test_labels);
                        
                        if p_valid > 0:
                            valid_sequences, valid_labels = self._split_sequences(valid_sequences, labels, frame_size, hop_size);                    
                            X_valid.append(valid_sequences);
                            y_valid.append(valid_labels);                
            
            def convert_to_dataset(X,y):            
                X = np.vstack(X);
                y = np.vstack(y);
                
                # convert labels
                y = self.label_converter.get_labels(y, self.label_mode);
                y = np.hstack(y);
                
                one_hot_y = one_hot(y);
                
                dataset = DenseDesignMatrix(X=X, y=one_hot_y);
                dataset.labels = y; # for confusion matrix
        
                return dataset;
            
            sets = [convert_to_dataset(X_train, y_train)];
            if p_test > 0:
                sets.append(convert_to_dataset(X_test, y_test));
            else:
                sets.append(None);
            if p_valid > 0:
                sets.append(convert_to_dataset(X_valid, y_valid));
            else:
                sets.append(None);
                
            return sets;
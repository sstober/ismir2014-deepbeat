'''
Created on Apr 2, 2014

@author: sstober
'''
import os;

import logging;
log = logging.getLogger(__name__);

from pylearn2.train_extensions import TrainExtension;
from pylearn2.corruption import Corruptor;

import numpy as np;

import theano;
import theano.tensor as T;
from sklearn.metrics import confusion_matrix,classification_report,precision_recall_fscore_support;

from pylearn2.config import yaml_parse;
from pylearn2.utils import serial;
from pylearn2.utils.timing import log_timing
from pylearn2.space import CompositeSpace
from pylearn2.space import NullSpace

from config import Config, ConfigMerger, overwriteMergeResolve;

def merge_params(default_params, override_params):

    merger = ConfigMerger(resolver=overwriteMergeResolve);
    params = Config();
    merger.merge(params, default_params);
    merger.merge(params, override_params);
    return params;

def load_yaml_file(yaml_file_path, params=None):
    
    with open(yaml_file_path, 'r') as f:
        yaml_template = f.read();
    f.close();
    
    return load_yaml(yaml_template, params);

def load_yaml(yaml_template, params=None):    
    print params;
    
    if params is not None:
        yaml_str = yaml_template % params;
    else:
        yaml_str = yaml_template;
    print yaml_str;

    with log_timing(log, 'parsing yaml'):    
        obj = yaml_parse.load(yaml_str);
    
    return obj, yaml_str;

def save_yaml_file(yaml_str, yaml_file_path):
    if save_yaml_file is not None:
        with log_timing(log, 'saving yaml to {}'.format(yaml_file_path)):
            save_dir = os.path.dirname(yaml_file_path);
            if save_dir == '':
                save_dir = '.'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with  open(yaml_file_path, 'w') as yaml_file:
                yaml_file.write(yaml_str) 
            yaml_file.close();
            
def process_dataset(model, dataset, data_specs=None, output_fn=None):
    
    if data_specs is None:
        data_specs = (CompositeSpace((
                                model.get_input_space(), 
                                model.get_output_space())), 
                           ("features", "targets"));
    
    if output_fn is None:                
        with log_timing(log, 'compiling output_fn'):         
            minibatch = model.get_input_space().make_theano_batch();
            output_fn = theano.function(inputs=[minibatch], 
                                        outputs=model.fprop(minibatch));
    
    it = dataset.iterator('sequential',
                          batch_size=100,
                          data_specs=data_specs);
    y_pred = [];
    y_real = [];                
    output = [];
    for minibatch, target in it:
        out = output_fn(minibatch); # this hangs for convnet on Jeep2
        output.append(out);
        y_pred.append(np.argmax(out, axis = 1));
        y_real.append(np.argmax(target, axis = 1));
    y_pred = np.hstack(y_pred);
    y_real = np.hstack(y_real);  
    output = np.vstack(output);
    
    return y_real, y_pred, output;

def aggregate_classification(seq_starts, y_real, y_pred, output):
    s_real = [];
    s_pred = [];
    s_predf = [];
    s_predfsq = [];
    for i in xrange(len(seq_starts)):        
        start = seq_starts[i];
        if i < len(seq_starts) - 1:
            stop = seq_starts[i+1];
        else:
            stop = None;
        
        s_real.append(y_real[start]);
        s_pred.append(np.argmax(np.bincount(y_pred[start:stop])));
        s_predf.append(np.argmax(np.sum(output[start:stop], axis=0)));      # sum of scores, then max
        s_predfsq.append(np.argmax(np.sum(np.log(output[start:stop]), axis=0)));   # experimental: sum of -log scores, then max
    
    s_real = np.hstack(s_real);
    s_pred = np.hstack(s_pred);  
    s_predf = np.hstack(s_predf); 
    s_predfsq = np.hstack(s_predfsq);   
    
    return s_real, s_pred, s_predf, s_predfsq;

class SaveEveryEpoch(TrainExtension):
    """
    A callback that saves a copy of the model every time 

    Parameters
    ----------
    save_path : str
        Path to save the model to
    """
    def __init__(self, save_path, save_prefix='cnn_epoch'):
        self.__dict__.update(locals())        

    def on_monitor(self, model, dataset, algorithm):
        
        epoch = algorithm.monitor._epochs_seen;
        model_file = self.save_path + self.save_prefix + str(epoch) + '.pkl'; 
        
        with log_timing(log, 'saving model to {}'.format(model_file)):
            serial.save(model_file, model, on_overwrite = 'backup')


class ClassificationLoggingCallback(TrainExtension):
    def __init__(self, dataset, model, header=None, 
                 class_prf1_channels=True, confusion_channels=True, 
#                  seq_prf1_channel=True, seq_confusion_channel=True
                 ):
        self.dataset = dataset;
        self.header = header;
        
        self.class_prf1_channels = class_prf1_channels;
        self.confusion_channels = confusion_channels;
                
        minibatch = model.get_input_space().make_theano_batch();
        self.output_fn = theano.function(inputs=[minibatch], 
                                        outputs=model.fprop(minibatch));
        
        self.data_specs = (CompositeSpace((
                                model.get_input_space(), 
                                model.get_output_space())), 
                           ("features", "targets"));
        
        if self.header is not None:
            self.channel_prefix = self.header;
        else:
            if hasattr(self.dataset, 'name'): #s  elf.dataset.name is not None:
                self.channel_prefix = self.dataset.name;
            else:
                self.channel_prefix = '';
                           
    def setup(self, model, dataset, algorithm):
        
#         print 'setup for dataset: {}\t {} '.format(dataset.name, dataset);
#         print 'self.dataset: {}\t {} '.format(self.dataset.name, self.dataset);
        
        if hasattr(self.dataset, 'get_class_labels'): 
            class_labels = self.dataset.get_class_labels();
        else:
            class_labels = ['0', '1'];
        
        # helper function
        def add_channel(name, val):
            model.monitor.add_channel(
                            name=self.channel_prefix+name,
                            ipt=None,                       # no input
                            data_specs = (NullSpace(), ''), # -> no input specs
                            val=val,
                            dataset=self.dataset,
                            );

        if self.class_prf1_channels:                                
            for class_label in class_labels:
                add_channel('_precision_'+str(class_label), 0.);
                add_channel('_recall_'+str(class_label), 0.);
                add_channel('_f1_'+str(class_label), 0.);
        
        add_channel('_f1_mean', 0.);
        
        # add channels for confusion matrix
        if self.confusion_channels:
            for c1 in class_labels:
                for c2 in class_labels:
                    add_channel('_confusion_'+c1+'_as_'+c2, 0.);

        add_channel('_seq_misclass_rate', 0.);
        add_channel('_wseq_misclass_rate', 0.);
        add_channel('_pseq_misclass_rate', 0.);
        
        add_channel('_trial_misclass_rate', 0.);
        add_channel('_wtrial_misclass_rate', 0.);
        add_channel('_ptrial_misclass_rate', 0.);
        
        add_channel('_trial_mean_f1', 0.);
        add_channel('_wtrial_mean_f1', 0.);
        add_channel('_ptrial_mean_f1', 0.);
        
        add_channel('_seq_mean_f1', 0.);
        add_channel('_wseq_mean_f1', 0.);
        add_channel('_pseq_mean_f1', 0.);
        
        
    def on_monitor(self, model, dataset, algorithm):
        
#         print 'self.dataset: {}\t {} '.format(self.dataset.name, self.dataset);
        
#         print self.dataset.X[0,0:5];
                
        y_real, y_pred, output = process_dataset(model, 
                                                 self.dataset, 
                                                 data_specs=self.data_specs, 
                                                 output_fn=self.output_fn)
        
        if self.header is not None:
            print self.header;                            

        # Compute confusion matrix
#         print classification_report(y_real, y_pred);
        conf_matrix = confusion_matrix(y_real, y_pred);
        
#         if self.dataset.name == 'test':
#             print conf_matrix;
                
        # log values in monitoring channels
        channels = model.monitor.channels;
        
        
        if hasattr(self.dataset, 'get_class_labels'): 
            class_labels = self.dataset.get_class_labels();
        else:
            class_labels = ['0', '1']; # FIXME: more flexible fallback required
        
        p, r, f1, s = precision_recall_fscore_support(y_real, y_pred, average=None);
        
        mean_f1 = np.mean(f1);
        misclass = (y_real != y_pred).mean();
        report = [['frames', mean_f1, misclass]];
        
        channels[self.channel_prefix+'_f1_mean'].val_record[-1] = mean_f1;
        
        if self.class_prf1_channels:
            for i, class_label in enumerate(class_labels):
                channels[self.channel_prefix+'_precision_'+str(class_label)].val_record[-1] = p[i];
                channels[self.channel_prefix+'_recall_'+str(class_label)].val_record[-1] = r[i];
                channels[self.channel_prefix+'_f1_'+str(class_label)].val_record[-1] = f1[i];
        
        if self.confusion_channels:
            # add channels for confusion matrix
            for i, c1 in enumerate(class_labels):
                for j, c2 in enumerate(class_labels):
                    channels[self.channel_prefix+'_confusion_'+c1+'_as_'+c2].val_record[-1] = conf_matrix[i][j];
                    
        if self.dataset.name == 'test':
            print confusion_matrix(y_real, y_pred);

        if hasattr(self.dataset, 'sequence_partitions'):
#             print 'sequence-aggregated performance';
            
            s_real, s_pred, s_predf, s_predp = aggregate_classification(
                                                                          self.dataset.sequence_partitions,
                                                                          y_real, y_pred, output);            
            # NOTE: uses weighted version for printout
            # both, weighted and un-weighted are logged in the monitor for plotting
            
            p, r, f1, s = precision_recall_fscore_support(s_real, s_pred, average=None);
            s_mean_f1 = np.mean(f1);
            
            p, r, f1, s = precision_recall_fscore_support(s_real, s_predf, average=None);
            ws_mean_f1 = np.mean(f1);
            
            p, r, f1, s = precision_recall_fscore_support(s_real, s_predp, average=None);
            ps_mean_f1 = np.mean(f1);
            
#             print classification_report(s_real, s_predf);
#             print confusion_matrix(s_real, s_predf);
            
            s_misclass = (s_real != s_pred).mean();
            ws_misclass = (s_real != s_predf).mean();
            ps_misclass = (s_real != s_predp).mean();
            
            report.append(['sequences', s_mean_f1, s_misclass]);
            report.append(['w. sequences', ws_mean_f1, ws_misclass]);
            report.append(['p. sequences', ps_mean_f1, ps_misclass]);
            
#             print 'seq misclass {:.4f}'.format(s_misclass);
#             print 'weighted seq misclass {:.4f}'.format(ws_misclass);
                        
            channels[self.channel_prefix+'_seq_misclass_rate'].val_record[-1] = s_misclass;
            channels[self.channel_prefix+'_wseq_misclass_rate'].val_record[-1] = ws_misclass;
            channels[self.channel_prefix+'_pseq_misclass_rate'].val_record[-1] = ps_misclass;
            
            channels[self.channel_prefix+'_seq_mean_f1'].val_record[-1] = s_mean_f1;
            channels[self.channel_prefix+'_wseq_mean_f1'].val_record[-1] = ws_mean_f1;
            channels[self.channel_prefix+'_pseq_mean_f1'].val_record[-1] = ps_mean_f1;
        
        if hasattr(self.dataset, 'trial_partitions'):
#             print 'trial-aggregated performance';
                        
            t_real, t_pred, t_predf, t_predp = aggregate_classification(
                                                                          self.dataset.trial_partitions,
                                                                          y_real, y_pred, output);            
            # NOTE: uses un-weighted version
            # both, weighted and un-weighted are logged in the monitor for plotting
            
            p, r, f1, s = precision_recall_fscore_support(t_real, t_pred, average=None);
            t_mean_f1 = np.mean(f1);
            
            p, r, f1, s = precision_recall_fscore_support(t_real, t_predf, average=None);
            wt_mean_f1 = np.mean(f1);
            
            p, r, f1, s = precision_recall_fscore_support(t_real, t_predp, average=None);
            pt_mean_f1 = np.mean(f1);
            
#             print classification_report(t_real, t_pred);

#             if self.dataset.name == 'test':
#                 print confusion_matrix(t_real, t_predp);
            
            t_misclass = (t_real != t_pred).mean();
            wt_misclass = (t_real != t_predf).mean();
            pt_misclass = (t_real != t_predp).mean();
            
            report.append(['trials', t_mean_f1, t_misclass]);
            report.append(['w. trials', wt_mean_f1, wt_misclass]);
            report.append(['p. trials', pt_mean_f1, pt_misclass]);

#             print 'trial misclass {:.4f}'.format(t_misclass);
#             print 'weighted trial misclass {:.4f}'.format(wt_misclass);
            
            channels[self.channel_prefix+'_trial_misclass_rate'].val_record[-1] = t_misclass;
            channels[self.channel_prefix+'_wtrial_misclass_rate'].val_record[-1] = wt_misclass;
            channels[self.channel_prefix+'_ptrial_misclass_rate'].val_record[-1] = pt_misclass;
            
            channels[self.channel_prefix+'_trial_mean_f1'].val_record[-1] = t_mean_f1;
            channels[self.channel_prefix+'_wtrial_mean_f1'].val_record[-1] = wt_mean_f1;
            channels[self.channel_prefix+'_ptrial_mean_f1'].val_record[-1] = pt_mean_f1;
        
        for label, f1, misclass in report:
            print '{:>15}:  f1 = {:.3f}  mc = {:.3f}'.format(label, f1, misclass);

class LoggingCallback(TrainExtension):
    def __init__(self, name='', obj_channel='objective'):
        self.name = name;
        self.obj_channel = obj_channel;

    def on_monitor(self, model, dataset, algorithm):
                
        epoch = algorithm.monitor._epochs_seen;
        lr = algorithm.monitor.channels['learning_rate'].val_shared.get_value();
        obj = algorithm.monitor.channels[self.obj_channel].val_shared.get_value();
        t_epoch = algorithm.monitor.channels['training_seconds_this_epoch'].val_shared.get_value();
#         max_norms = algorithm.monitor.channels['training_seconds_this_epoch'].val_shared.get_value();
        
        log.debug('running {0} (lr={1:.7f}), epoch {2}, cost {3:.4f}, t_epoch={4}'.format(
                         self.name,
                         float(lr), #algorithm.learning_rate.get_value(),
                         epoch, 
                         float(obj),
                         t_epoch));


class LoggingCorruptor(Corruptor):
    '''
    decorator for an actual corruptor
    logs whenever the corruptor is called 
    '''
    
    def __init__(self, corruptor, name=''):
        self.name = name;
        self._corruptor = corruptor;
        self.corruption_level = self._corruptor.corruption_level; # copy value so that it can be read and changed
    
    def __call__(self, inputs):
        self._corruptor.corruption_level = self.corruption_level; # make sure of correct value before call
        log.debug('corruptor.call {} called, corruption_level={}'.format(self.name, self._corruptor.corruption_level));        
        return self._corruptor.__call__(inputs);
    
    def _corrupt(self, x):
        self._corruptor.corruption_level = self.corruption_level; # make sure of correct value before call
        log.debug('corruptor.corrupt {} called, corruption_level={}'.format(self.name, self._corruptor.corruption_level));
        return self._corruptor._corrupt(x);
    

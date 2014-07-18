'''
Created on Apr 7, 2014

@author: sstober
'''

import numpy as np;

class LabelConverter(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        audio_files = [
               '180C_10.wav180F_8_180afr.wav',
               '180C_10.wav180F_8_180west.wav',
               '180C_10.wav180F_9_180afr.wav',
               '180C_10.wav180F_9_180west.wav',
               '180C_11.wav180F_12_180west.wav',
               '180C_11.wav180F_13_180west.wav',
               '180C_12.wav180F_11_180west.wav',
               '180C_12.wav180F_13_180west.wav',
               '180C_13.wav180F_11_180west.wav',
               '180C_13.wav180F_12_180west.wav',
               '180C_3.wav180F_4_180afr.wav',
               '180C_3.wav180F_5_180afr.wav',
               '180C_4.wav180F_3_180afr.wav',
               '180C_4.wav180F_5_180afr.wav',
               '180C_5.wav180F_3_180afr.wav',
               '180C_5.wav180F_4_180afr.wav',
               '180C_8.wav180F_10_180afr.wav',
               '180C_8.wav180F_10_180west.wav',
               '180C_8.wav180F_9_180afr.wav',
               '180C_8.wav180F_9_180west.wav',
               '180C_9.wav180F_10_180afr.wav',
               '180C_9.wav180F_10_180west.wav',
               '180C_9.wav180F_8_180afr.wav',
               '180C_9.wav180F_8_180west.wav',
               '240C_10.wav240F_8_240afr.wav',
               '240C_10.wav240F_8_240west.wav',
               '240C_10.wav240F_9_240afr.wav',
               '240C_10.wav240F_9_240west.wav',
               '240C_11.wav240F_12_240west.wav',
               '240C_11.wav240F_13_240west.wav',
               '240C_12.wav240F_11_240west.wav',
               '240C_12.wav240F_13_240west.wav',
               '240C_13.wav240F_11_240west.wav',
               '240C_13.wav240F_12_240west.wav',
               '240C_3.wav240F_4_240afr.wav',
               '240C_3.wav240F_5_240afr.wav',
               '240C_4.wav240F_3_240afr.wav',
               '240C_4.wav240F_5_240afr.wav',
               '240C_5.wav240F_3_240afr.wav',
               '240C_5.wav240F_4_240afr.wav',
               '240C_8.wav240F_10_240afr.wav',
               '240C_8.wav240F_10_240west.wav',
               '240C_8.wav240F_9_240afr.wav',
               '240C_8.wav240F_9_240west.wav',
               '240C_9.wav240F_10_240afr.wav',
               '240C_9.wav240F_10_240west.wav',
               '240C_9.wav240F_8_240afr.wav',
               '240C_9.wav240F_8_240west.wav'
               ]
        
        self.class_labels = {};
        self.class_labels['rhythm'] = audio_files;
        self.class_labels['rhythm_type'] = ['African', 'Western'];
        self.class_labels['tempo'] = ['180', '240'];
        
        self.stimulus_id_map = {};
        self.label_map = {}; #np.empty(len(audio_files));
        for i, audio_file in enumerate(audio_files):
            self.stimulus_id_map[audio_file] = i;
            
            labels = {};
            labels['audio_file'] = audio_file;
            
            if '180' in audio_file:
#                 labels['tempo'] = 180;
                labels['tempo'] = 0;
            else:
#                 labels['tempo'] = 240;
                labels['tempo'] = 1;
        
            if 'west' in audio_file:
#                 labels['rhythm_type'] = 'W';
#                 labels['rhythm_type'] = 1;
                labels['rhythm_type'] = self.class_labels['rhythm_type'].index('Western');
            else:
#                 labels['rhythm_type'] = 'A';
#                 labels['rhythm_type'] = 0;
                labels['rhythm_type'] = self.class_labels['rhythm_type'].index('African');
                
            labels['rhythm'] = i % 24; # map down to 24 classes
        
            self.label_map[i] = labels;
        
    def get_class_labels(self, label_mode):
        return self.class_labels[label_mode];
        
    def get_stimulus_id(self, stimulus):
        return self.stimulus_id_map[stimulus];
    
    def get_tempo_label(self, stimulus_id):
        return self.label_map[stimulus_id]['tempo'];
        
    def get_rhythm_type_label(self, stimulus_id):
        return self.label_map[stimulus_id]['rhythm_type'];
    
    def get_audio_file(self, stimulus_id):
        return self.label_map[stimulus_id]['audio_file'];
    
    def get_label(self, stimulus_id, label_mode):
        return self.label_map[stimulus_id][label_mode];
    
    def get_labels(self, stimulus_ids, label_mode):
        labels = [];
#         counts = np.zeros(50);
        for i in xrange(len(stimulus_ids)):
            stimulus_id = stimulus_ids[i][0]; # FIXME 
#             counts[stimulus_id] += 1;
            labels.append(self.label_map[stimulus_id][label_mode]);
#         print labels.count(180);
#         print labels.count(240);
#         print counts;
#         return np.vstack(labels);
        return labels;
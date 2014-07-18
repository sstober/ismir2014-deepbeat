'''
Created on Apr 30, 2014

@author: sstober
'''

from pylearn2.models.mlp import Softmax;
from pylearn2.utils import wraps

import theano.tensor as T;
from theano.printing import Print;
import theano;

class SoftmaxClassificationLayer(Softmax):
    '''
    classdocs
    '''
#     def __init__(self, n_classes, layer_name, irange=None,
#                  istdev=None,
#                  sparse_init=None, W_lr_scale=None,
#                  b_lr_scale=None, max_row_norm=None,
#                  no_affine=False,
#                  max_col_norm=None, init_bias_target_marginals=None):
#         
#         super(SoftmaxClassificationLayer, self).__init__(n_classes, layer_name);
     
    def __init__(self,  *args, **kwargs):
        '''
        Constructor
        '''
        super(SoftmaxClassificationLayer, self).__init__(*args, **kwargs);
    
    def _print_Y(self, op, xin):
        for attr in op.attrs:
#             temp = getattr(np.argmax(xin[0:1]), attr)
            temp = getattr((xin), attr)
            if callable(temp):
                pmsg = temp()
            else:
                pmsg = temp
            print op.message, ' = ', pmsg
            
    
    @wraps(Softmax.cost)
    def cost(self, Y, Y_hat):
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        
#         # misclass zero-one loss -> does not work
#         Y = T.argmax(Y, axis=1);
#         Y_hat = T.argmax(Y_hat, axis=1);              
#         
#         misclass = T.neq(Y, Y_hat).mean();
#         misclass = T.cast(misclass, theano.config.floatX);
#         return misclass;
    
        # MeanSquaredReconstructionError : ((a - b) ** 2).sum(axis=1).mean()
#         return T.sum((Y - Y_hat) ** 2, axis=1).mean();
    
        # custom hinge loss -> does not work
        # hinge loss  = \max(0, 1 + \max_{y \ne t} \mathbf{w}_y \mathbf{x} - \mathbf{w}_t \mathbf{x})
        
#         Y = theano.printing.Print('Y', global_fn=self._print_Y)(Y)
#         Y_hat = theano.printing.Print('Y_hat', global_fn=self._print_Y)(Y_hat)

        Y_t = T.max(Y * Y_hat, axis=1); # activation of the desired output
        
#         Y_t = theano.printing.Print('Y_t', global_fn=self._print_Y)(Y_t)
                
        loss = 0.1 + T.max(Y_hat, axis=1) - Y_t;

#         loss = theano.printing.Print('loss', global_fn=self._print_Y)(loss)
                
        loss = T.cast(loss, theano.config.floatX);
        return T.mean(loss)
    
    @wraps(Softmax.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        raise NotImplementedError();
    
    @wraps(Softmax.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        
        rval = super(SoftmaxClassificationLayer, self).get_monitoring_channels_from_state(state, target);

        rval['nll'] = super(SoftmaxClassificationLayer, self).cost(Y_hat=state, Y=target);
        rval['cost'] = self.cost(Y_hat=state, Y=target);
        
        return rval;
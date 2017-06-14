import numpy as _np
from pyemma.coordinates.data._base.transformer import StreamingTransformer as _StreamingTransformer
from itertools import combinations as _combinations

class ProductMaker(_StreamingTransformer):
    def __init__(self, chunksize=1000, take_max_dim=float('inf')):
        super(ProductMaker, self).__init__(chunksize=chunksize)
        if hasattr(self, '_estimated'):
            self._estimated = True # work around bug in Pyemma
        self._take_max_dim = take_max_dim
    def describe(self):
        return '[ buy my products! ]'
    def dimension(self):
        d = min(self.data_producer.dimension(), self._take_max_dim)
        extra_d = d + d*(d-1)//2 # squares + all different products
        return d + extra_d
    def _transform_array(self, X):
        d = min(X.shape[1], self._take_max_dim)
        Y = _np.zeros((X.shape[0], self.dimension()), dtype=X.dtype)
        Y[:, 0:d] = X[:, 0:d] # orig. data = products with the implicit 1 eigenfunctions
        # TODO: optimize
        for i in range(d):
            Y[:, i+d] = X[:, i]*X[:, i] # squares
        for i, (k, l) in enumerate(_combinations(range(d), 2)):
            Y[:, i+2*d] = X[:, k]*X[:, l] # products
        return Y


def product_maker(input_stage, chunksize=1000, take_max_dim=float('inf')):
    product_maker = ProductMaker(chunksize=chunksize, take_max_dim=take_max_dim)
    product_maker.data_producer = input_stage
    return product_maker


from pyemma.coordinates import tica as _tica

def product_tica(input_stage, lag, tica_args_stage_1={}, tica_args_stage_2={}, return_stage_1=False, take_max_dim=float('inf')):
    r'''Hierarchical TICA algorithm that consits of two stages of TICA.
    The first stage is a conventional TICA run. In the second TICA stage, new basis functions
    are added that are the products of all tICs computed by the first TICA run.

    parameters
    ----------
    input_stage : data or Pyemma transformer
    lag : int
        TICA lag time for both stages. Can be overwritten by `tica_args_stage_1`
        and `tica_args_stage_2`.
    tica_args_stage_1 : dictionary
        parameters for the first TICA stage
    tica_args_stage_2 : dictionary
        parameters for the second TICA stage (TICA with product basis functions)
    return_stage_1 : boolean, default = False
        whether to return TICA stage 1
    take_max_dim : integer, default = inf
        only keep the first `take_max_dim` dimensions of the first TICA stage.

    returns
    -------
    depending on the value of return_stage_1:
    * stage_1
    * (stage_1, stage_2)

    stage_1 :  a Pyemma TICA object
    stage_2 :  a Pyemma TICA object
    '''
    if not 'lag' in tica_args_stage_1:
        tica_args_stage_1['lag'] = lag
    if not 'lag' in tica_args_stage_2:
        tica_args_stage_2['lag'] = lag
    tica_1st_level = _tica(data = input_stage, **tica_args_stage_1)
    p = product_maker(tica_1st_level, take_max_dim=take_max_dim)
    tica_2nd_level = _tica(p, **tica_args_stage_2)
    if return_stage_1:
        return tica_1st_level, tica_2nd_level
    else:
        return tica_2nd_level

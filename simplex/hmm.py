# TODO: think about HMM with multiple emissions
import numpy as np
import msmtools

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1)[:, np.newaxis]

class NullLogger(object):
    def log(self, **kwargs):
        pass

class MSM(object):
    # wrap Pyemma MSM or msmtools MSM
    def __init__(self, n_states):
        self.n_states = n_states
        self.estimated = False
        # gap: its > tau and large relative gap...
        # TODO: automatic identification of spectral gap with RMA?

    def estimate(self, observations, lag):
        C = msmtools.estimation.count_matrix(observations, lag=lag)

        # compute memberships
        lcs = msmtools.estimation.largest_connected_set(C, directed=False)
        T = np.zeros(C.shape)
        T[lcs, lcs[:, np.newaxis]] = msmtools.estimation.tmatrix(C[lcs, :][:, lcs], revsersible=False).todense()

        C_back = msmtools.estimation.count_matrix([o[::-1] for o in observations], lag=lag)
        lcs_back = msmtools.estimation.largest_connected_set(C_back, directed=False)
        T_back = np.zeros(C.shape)
        T_back[lcs_back, lcs_back[:, np.newaxis]] = msmtools.estimation.tmatrix(C_back[lcs_back, :][:, lcs_back], reversible=False).todense()
        
        T_forward_backward = T.dot(T_back)
        lcs_forward_backward = msmtools.estimation.largest_connected_set(T_forward_backward, directed=True)
        chi = np.zeros((C.shape[0], self.n_states))
        chi[lcs_forward_backward, :] = msmtools.analysis.pcca(T_forward_backward[lcs_forward_backward, :][:, lcs_forward_backward], self.n_states)

        # compute pi
        # TODO: compute individual pi for all the disconnected blocks and weight by frequency???
        lcs_pi = msmtools.estimation.largest_connected_set(C, directed=True)
        T_pi = np.zeros(C.shape)
        T_pi[lcs, lcs[:, np.newaxis]] = msmtools.estimation.tmatrix(C[lcs, :][:, lcs]).todense()
        pi = np.zeros(C.shape[0])
        pi[lcs_pi] = msmtools.analysis.statdist(T_pi[lcs_pi, :][:, lcs_pi])
        self.PI = np.diag(pi)
        # TODO: we don't really don't need any connectivity when doing PCCA...
        # can we work with an ill-defined pi? 
        # ideally we should just do pcca on all the disconnected parts... then there is always a defined pi
        
        # membership will be defined on the full observed space (but might be zero for some states in the observed space)
        self.b = chi.dot(PI)
        # coarse-grained T matrix
        self.T = np.linag.inv(chi.dot(PI).dot(chi.T)).dot(chi.dot(T).dot(chi.T))
        self.estimated = True

    @property
    def b(self):
        return self.b

    @property
    def p0(self, use_all=False):
        if not use_all:
            # use initial memberships
            p0 = np.array([ chi[o[0], :] for o in observations ]).mean(axis=0)
            return p0
        else:
            # use empirical counts
            vset, counts = np.unique(np.concatenate(observations), return_counts=True)
            p0 = np.zeros()
            p0[vset] = counts
            return p0 / p0.sum()

    @property
    def T(self):
        'coarse-grained transition matrix'
        return self.T

class DiscreteEmissionModel(object):
    def __init__(self):
        pass

    def initialize(self, model, observations):
        self.observations = observations
        assert isinstance(model, MSM)
        self._b = model.b

    # TODO: make estimate a function
    def estimate(self, metastable_memberships):
        'estimate b_i(k) from the fixed observed space trajectories and the metastable memberships'
        self._unique_observations = np.unique([np.unique(o) for o in self.observations])
        self._b = np.zeros(np.max(self._unique_observations) + 1)
        gamma = metastable_memberships
        norm = gamma.sum(axis=0) # sum over all frames (time steps)
        for o, g in zip(observations, gamma):
            for i in self._unique_observations:
                self._b[i, :] += (g[o==i, :]).sum(axis=0) / norm
        # TODO: rewrite such that we return a model; don't change this object's state!

    def emission_probability(self): # TODO better name
        'return [ b_i(O_(t)) ]_t,i'
        # TODO: make b trajectory-depdent (list)
        return [ self._b[o] for o in self.observations ]

    # alternative implementation of what???
    def b_alt(self):
        observations = dense(self.observations)
        # TODO: check index order!
        #b = (metastable_memberships[:, np.newaxis, :]*observations[:, :, np.newaxis]).sum(axis=0) / metastable_memberships.sum(axis=0)
        #return observations.dot(b)
        return observations.dot((metastable_memberships[:, np.newaxis, :]*observations[:, :, np.newaxis]).sum(axis=0) / metastable_memberships.sum(axis=0))
        

class SimplexCoreMSM(object):
    # simplex and TBA MSM
    def estimate(self, observations, lag):
        from . import milestoning_count_matrix, find_vertices_inner_simplex, core_assignments, memberships
        vertices = find_vertices_inner_simplex(observations)
        mems = memberships(vertices, observations)
        ctrajs = core_assignments_given_memberships(mems, observations)
        C = milestoning_count_matrix(ctrajs)
        T = C / C.sum(axis=1)[:, np.newaxis]
        self.T = T
        self.vertices = vertices
        self.p0 = np.array([ m[0, :] for m in mems ]).mean(axis=0)
        #return SimplexCoreMSMModel(self.T, self.vertices, ctrajs)
        
    @property
    def p0(self):
        pass
        
    @property
    def T(self):
        return self.T

class SimplexEmissionModel(object):
    def __init__(self):
        self.W = None
        
    def initialize(self, model, observations):
        assert isinstance(model, SimplexCoreMSM)
        # get W from simplex
        self.W = model.vertices # TODO: take inverse! somehow, see lyx
        self.observations = observations # these are the TICs

    # scipy special softmax?
    def estimate(self, metastable_memberships):
        import scipy as sp
        self.observations = observations  # ??
        # optimize W to match metastable_memberships
        def function_and_gradient(x):
            W = np.unravel(x, self.W.shape)
            function = 0.0
            gradient = np.zeros_like(self.W)
            for obs, meta_mem in zip(self.observations, metastable_memberships):
                obs = np.hstack((np.ones(obs.shape[0], dtype=obs.dtype), obs))  # add back the constant eigenfunction
                logistic_mem = softmax(obs.dot(W))
                function += -(meta_mem*np.log(logistic_mem)).sum()  # optimized
                gradient += np.dot(obs.T, logistic_mem-meta_mem)  # TODO: test if einsum is faster?
            return (function, gradient)
        # scipy-optimize W
        res = sp.optimize.minimize(function_and_gradient, np.reshape(self.W, -1), method='L-BFGS-B' , jac=True) # tol=None?
        if res.success:
            self.W = np.reshape(res.x, self.W.shape)
        else:
            raise Exception(res.message)

    @property
    def emission_probability(self):
        b = []
        for obs in self.observations:
            np.hstack((np.ones(obs.shape[0], dtype=obs.dtype), obs))
            logistic_mem = softmax(obs.dot(W))
            b.append(logistic_mem / logistic_mem.sum(axis=0)[np.newaxis, :])
        return b

    #@property
    #def vertices(self):
    #    return self.W # TODO: correct me!

# to the wiring on the api function level
def hmm():
    pass


class HMM(object):
    def __init__(self, observations, lag, initial_model=MSM(n_states=None), emission_model=DiscreteEmissionModel(), logger=NullLogger(), max_iter=1000000, tol=1.E-6):
        self.emission_model = emission_model
        self.initial_model = initial_model
        # can we also pass an estimated model?
        if not initial_model.estimated:
            initial_model.estimate(observations=observations, lag=lag)
        T = len(observations)
        alpha = np.zeros((T, n_states))
        beta = np.zeros((T, n_states))
        self.logger = logger
        self.emission_model.intialize(initial_model, observations)
        self.max_iter = max_iter
        self.tol = tol

    def estimate(self):
        # initialization step
        # pi = counts or initial memberships?? since we use gamma later...
        #self.emission_model.intialize(initial_model) # or so variant of parameter update?

        # TODO: make alpha and beta trajectory-dependent -> list of nparray
        # TODO: think about pi (keep list of pi or make one global one?) How is Frank doing it? local pi makes more sense
        
        pi = self.initial_model.p0  # attention: pi means p0 here!
        a = self.intial_model.T
        last_L = float('inf')
        for iteration in range(self.max_iter):
            # expectation step
            b = self.emission_model.emission_probability()
        
            alpha[0, :] = b[0, :]*pi
            for t in range(1, T): # TODO: scaling
                alpha[t, :] =  alpha[t-1, :].dot(self.a)*b[t, :]
            beta[-1, :] = 1
            for t in range(T-2, 0, -1):
                b[t, :] = a.dot(b[t+1, :]*beta[t+1, :])

            temp = alpha*beta
            gamma = temp / temp.sum(axis=1)[:, np.newaxis]

            temp = alpha[0:-1, :, np.newaxis]*a[np.newaxis, :, :]*b[1:, np.newaxis, :]*beta[1:, np.newaxis, :]
            xi = temp / temp.sum(axis=2).sum(axis=1)[:, np.newaxis, np.newaxis]

            # compute likelihood
            L = alpha[-1, :].sum() # P(O|lambda) in the paper # TODO: make trajectory-dependent (see one of the appendices in the paper)
            delta_L = abs(last_L - L) 

            # maximization step
            self.emission_model.estimate(alpha*beta)
            # continue with which memberships???? Read paper again!
             
            # update pi (p0 and the hidden transition matrix)
            pi = gamma[0, :]
            a = xi.sum(axis=0) / gamma.sum(axis=0) # CHECK
             
            if delta_L < self.tol:
                break

        self.T = a
        self.p0 = pi
        self.gamma = gamma
        self.pi = msmtools.analysis.statdist(self.T)

        self.logger.log(self)
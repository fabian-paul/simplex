# TODO: think about HMM with multiple emissions

def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1)[:, np.newaxis]

class NullLogger(object):
    def log(self, *+kwargs)
        pass

class DiscreteEmissionModel(object):
    def __init__(self):
       # TODO: multiple trajectories ...
       pass

    def estimate(self, observations, metastable_memberships):
        self.observations = observations # TODO: optimize
        self._unique_observations = np.unique([np.unique(o) for o in observations])
        self._b = np.zeros(np.max(self._unique_observations)+1)
        gamma = metastable_memberships
        norm = gamma.sum(axis=0)
        for o,g in zip(observations, gamma):
            for i in self._unique_observations:
                self._b[i, :] += (g[o==i, :]).sum(axis=0) / norm

    def emission_probability(self): # TODO better name
        # TODO: make b trajectory-depdent (list)
        return [self._b[o] for o in self.observations]
        
    def b_alt(self):
        observations = dense(self.observations)
        # TODO: check index order!
        b = (metastable_memberships[:, np.newaxis, :]*observations[:, :, np.newaxis]).sum(axis=0) / metastable_memberships.sum(axis=0)
        return observations.dot(b)
        return observations.dot((metastable_memberships[:, np.newaxis, :]*observations[:, :, np.newaxis]).sum(axis=0) / metastable_memberships.sum(axis=0))
        

class SimplexCoreMSM(object):
    # simplex and TBA MSM
    pass

class SimplexEmissionModel(object):
    def __init__(self):
        self.W = None
        
    def initialize(self, model):
        # get W from simplex
        self.W = model.vertices # TODO: take inverse! somehow, see lyx
        self.observations = model.tics # TODO: name?

    # scipy special softmax?
    def estimate(self, observations, metastable_memberships):
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
        res = sp.optimize.minimize(function_and_gradient, np.reshape(self.W, -1), method=‘L-BFGS-B’ , jac=True) # tol=None?
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

    @property
    def vertices(self):
        return self.W # TODO: correct me!


class MSM(object):
    # wrap Pyemma MSM or msmtools MSM
    def __init__(self):
        pass
    
    def estimate(self, observations):
        C = msmtools.estimation.count_matrix(observations)
        self.T = msmtools.estimation.tmatrix(C).todense()

    @property
    def b(self):
        pass

    @property
    def p0(self):
        pass

    @property
    def a(self):
        pass
        # coarse-grained? matrix
        
    T = a

class HMM(object):
    def __init__(self, n_states, initial_model=MSM(), emission_model=DiscreteEmissionModel(), observations, logger=ZeroLogger()):
        self.emission_model = emission_model
        T = len(observations)
        alpha = np.zeros((T, n_states))
        beta = np.zeros((T, n_states))
        self.logger = logger

    def estimate(self):
        # initialization step
        # pi = counts or initial memberships?? since we use gamma later...
        self.emission_model.intialize(initial_model) # or so variant of parameter update?

        # TODO: make alpha and beta trajectory-dependent -> list of nparray
        # TODO: think about pi (keep list of pi or make one global one?) How is Frank doing it? local pi makes more sense
        while True:
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
             L = alpha[-1, :].sum() # P(O|lambda) in the paper # TODO: make trajectory-dependent

             # maximization step
             self.emission_model.estimate(alpha*beta)
             pi = gamma[0, :]
             a = xi.sum(axis=0) / gamma.sum(axis=0) # CHECK

        self.logger.log(self)
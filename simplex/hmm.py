# TODO: think about HMM with multiple emissions
import numpy as np
import scipy as sp
import scipy.optimize
import msmtools
from .simplex import core_assignment_given_memberships, milestoning_count_matrix


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1)[:, np.newaxis]


def dense(dtrajs, n_states=None, dtype=np.float32):  # TODO: also implement sparse?
    if n_states is None:
        n_states = max(np.max(d) for d in dtrajs) + 1
    dense_trajs = []
    for dtraj in dtrajs:
        dense_traj = np.zeros((len(dtraj), n_states), dtype=dtype)
        dense_traj[np.arange(len(dtraj)), dtraj] = 1.0  # TODO: check me
        dense_trajs.append(dense_traj)
    return dense_trajs


class NullLogger(object):
    def log(self, **kwargs):
        pass

# TODO: distinguish this MSM from Pyemma's or make compatible
class MSM(object):
    # wrap Pyemma MSM or msmtools MSM
    def __init__(self, n_states, reversible=True):
        # TODO: make n_states automatic (min-Chi)!
        self.n_states = n_states
        self.estimated = False
        self.reversible = reversible
        # gap: its > tau and large relative gap...
        # TODO: automatic identification of spectral gap with simplex method or RMA?

    def estimate(self, observations, lag):
        from functools import reduce
        C = msmtools.estimation.count_matrix(observations, lag=lag).toarray()

        csets = msmtools.estimation.connected_sets(C, directed=self.reversible)
        lcs = sorted(csets, key=len)[-1]
        T = np.zeros(C.shape, dtype=np.float64)
        for cs in csets:
            if C[cs, :][:, cs].sum() == 0:
                T[cs, cs] = 1.0
            else:
                T[cs[:, np.newaxis], cs] = msmtools.estimation.tmatrix(C[cs, :][:, cs], reversible=self.reversible)

        # TODO: disconnected sets for the purpose of coarse-graining should be found using non-reversible connectivity
        # TODO: then do G-PCCA inside each of the sets
        # compute memberships
        if self.reversible:
            chi = np.zeros((C.shape[0], self.n_states + len(csets) - 1))
            k = self.n_states
            for cs in csets:
                if list(cs) == list(lcs):
                    n_states = self.n_states
                    chi[cs, 0:n_states] = msmtools.analysis.pcca(T[cs, :][:, cs], n_states)
                else:
                    # TODO: find number of states based on the min-Chi criterion
                    chi[cs, k] = 1.0
                    k += 1
        else:
            raise NotImplementedError('non-reversible case not yet implemented')
            # TODO: implement using Schur decomposition and G-PCCA


        rho = np.diag(C.sum(axis=1))

        self.chi = chi
        
        # membership will be defined on the full observed space (but might be zero for some states in the observed space)
        self._b = np.dot(rho, self.chi)

        # coarse-grained T matrix
        C0_cg = chi.T.dot(rho).dot(chi)
        empty = np.where(C0_cg.sum(axis=1) == 0)[0]
        C0_cg[empty, empty] = 1
        Ct_cg = chi.T.dot(C).dot(chi)
        empty = np.where(Ct_cg.sum(axis=1) == 0)[0]
        Ct_cg[empty, empty] = 1
        self._T = np.linalg.inv(C0_cg).dot(Ct_cg)
        # TODO: do we need to make elements positive?
        self.estimated = True

    #@property
    #def pcca_score(self):
    #    pass

    @property
    def b(self):
        return self._b

    def p0(self, observations, frames=slice(0, 1)):
        # use initial memberships (first frame of every dtraj)
        p0 = np.hstack([ self.chi[o[frames], :] for o in observations ]).mean(axis=0)
        return p0

    @property
    def T(self):
        'coarse-grained transition matrix'
        return self._T

class DiscreteEmissionModel(object):
    r'''Classical discrete-state emission model for HMMs.
        q_t : hidden state at time t
        S_i : symbol for hidden state number i
        O_t : observed state at time t
        v_i : symbol for observed state number i

        b_i(O_t=v_k) = P(q_t = S_i | O_t = v_k), saved internally as _b[k, i]
    '''
    def __init__(self):
        pass

    def initialize(self, model, observations):
        self.observations = observations
        assert isinstance(model, MSM)
        self._b = model.b
        self._unique_observations = np.unique([np.unique(o) for o in self.observations])
        self.n_obs = max(self._unique_observations) + 1  # number of distinct observed states
        self.n_states = model.n_states  # number of hidden states

    def estimate(self, joint_probabilities, trajectory_weights=None):
        r'''estimate the internally saved distributions b_i(k) from the fixed observed space trajectories and the joint probabilities

        parameters
        ----------
        joint_probabilities : list of np.array(())
            for each trajectory with number n
            joint_probabilities[n][i, k] = P(q_t = S_i and O_t = v_k | model)
        trajectory_weights : list of floats
            probability of each trajectory
        '''
        if trajectory_weights is None:
            trajectory_weights = [1.0]*len(joint_probabilities)
        self._b = np.zeros(self.n_obs, self.n_states)
        norm = np.zeros(self.n_states, dtype=np.float64)
        for ab, w in zip(joint_probabilities, trajectory_weights):
            norm += w*ab.sum(axis=0)
        for o, ab, w in zip(self.observations, joint_probabilities, trajectory_weights):
            for i in self._unique_observations:
                self._b[i, :] += w*(ab[o==i, :]).sum(axis=0) / norm

    @property
    def emission_likelihood(self):
        r'''likelihood of the fixed sequence of observed visible (observable) states

        :returns list of np.array(T_n, S_i)
            [ P(O_t|q_t=S_i) t for 0 ... T_n ] where O_t are fixed and q_t are free
        '''
        return [ self._b[o, :] for o in self.observations ]


class SimplexCoreMSM(object):
    # simplex and TBA MSM
    def estimate(self, observations, lag):
        from . import milestoning_count_matrix, find_vertices_inner_simplex, core_assignments, memberships
        vertices = find_vertices_inner_simplex(observations)
        mems = memberships(vertices, observations)
        ctrajs = core_assignment_given_memberships(mems, observations)
        C = milestoning_count_matrix(ctrajs)
        T = C / C.sum(axis=1)[:, np.newaxis]
        self._T = T
        self.vertices = vertices
        self._p0 = np.array([ m[0, :] for m in mems ]).mean(axis=0)
        #return SimplexCoreMSMModel(self.T, self.vertices, ctrajs)
        self.estimated = True
        
    @property
    def p0(self):
        return self._p0
        
    @property
    def T(self):
        return self._T

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
        self.observations = self.observations  # ??
        # optimize W to match metastable_memberships
        def function_and_gradient(x):
            W = np.unravel(x, self.W.shape)  # fixme!
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
        r'''[ P(v_k at t|q_t = S_j) t for 0 ... T ]
        '''
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

#def sum(l, weights=None, axis=None):
#    if weights is None:
#        weights = [1.0]*len(l)
#    sum = l[0].sum(axis=axis)
#    for x,w in zip(l[1:], weights[1:]):
#        sum += w*x.sum(axis=axis)
#    return sum

class HMM(object):
    def __init__(self, observations, lag, initial_model=MSM(n_states=None), emission_model=DiscreteEmissionModel(), logger=NullLogger(), max_iter=1000000, tol=1.E-6):
        self.emission_model = emission_model
        self.initial_model = initial_model
        self.observations = observations
        # can we also pass an estimated model?
        if not initial_model.estimated:
            initial_model.estimate(observations=observations, lag=lag)
        self.logger = logger
        self.emission_model.initialize(initial_model, observations)
        self.max_iter = max_iter
        self.tol = tol

    def estimate(self):
        last_logL = np.inf
        n_states = self.initial_model.n_states
        # general convention for indices: zero'th index is time, if possible

        # TODO: should we have one initial distribution per trajectory?
        alpha = [ np.zeros((len(o_traj), n_states)) for o_traj in self.observations]   # Rabiner notation
        beta = [ np.zeros((len(o_traj), n_states)) for o_traj in self.observations]  # Rabiner notation

        pi = self.initial_model.p0  # attention: pi means p0 here!
        a = self.initial_model.T  # a is the transition matrix
        for iteration in range(self.max_iter):
            # expectation step
            bl = self.emission_model.emission_likelihood

            # forward procedure:
            for alpha_traj, bl_traj in zip(alpha, bl):
                alpha_traj[0, :] = bl_traj[0, :]*pi
                for t in range(1, len(alpha_traj)): # TODO: scaling
                    alpha_traj[t, :] =  alpha_traj[t-1, :].dot(a)*bl_traj[t, :]
            # backward procedure:
            for beta_traj, bl_traj in zip(beta, bl):
                beta_traj[-1, :] = 1
                for t in range(len(beta_traj)-2, 0, -1):
                    beta_traj[t, :] = a.dot(bl_traj[t+1, :]*beta_traj[t+1, :])
            alpha_times_beta = [alpha_traj*beta_traj for alpha_traj, beta_traj in zip(alpha, beta)]

            # likelihoods of the individual trajectories
            P = [alpha_traj[-1, :].sum() for alpha_traj in alpha]
            # compute likelihood
            logL = np.sum(np.log(P)) # P(O|lambda) in the paper
            delta_logL = abs(last_logL - logL)

            # update step:
            # update pi (i. e. p(t=0) of the hidden transition matrix)
            pi = np.sum([1.0/p_traj*alpha_traj[0, :]*beta_traj[0, :] for alpha_traj, beta_traj, p_traj in zip(alpha, beta, P)])
            pi = pi/pi.sum()
            # update transition matrix
            counts = np.zeros((n_states, n_states), dtype=np.float64)
            visits = np.zeros(n_states, dtype=np.float64)
            for i, (alpha_traj, beta_traj, b_traj, p_traj) in enumerate(zip(alpha, beta, b, P)):
                counts += (1.0 / p_traj) * np.einsum('ti,ij,tj,tj->ij', alpha_traj[:-1], a, b_traj[1:], beta_traj[1:])
                visits += (1.0 / p_traj) * np.einsum('ti,ti->i', alpha_traj[:-1], beta_traj[:-1])
            a = counts / visits[:, np.newaxis]
            # update emissions
            self.emission_model.estimate(alpha_times_beta, trajectory_weights=1./P)  # update b
             
            if delta_logL < self.tol:
                break

        self.T = a
        self.p0 = pi
        #self.gamma = gamma
        #self.pi = msmtools.analysis.stationary_distribution(self.T)
        self.estimated = True  # should we return a Model object instead?
        #self.logger.log(self)



if __name__ == '__main__':
    n = 10
    N = 100
    C = np.random.rand(n, n)
    C = C +  C.T
    C = C / C.sum()
    T = C / C.sum(axis=1)[:, np.newaxis]
    T_cdf = T.cumsum(axis=1)
    assert msmtools.analysis.is_transition_matrix(T)
    assert msmtools.analysis.is_reversible(T)
    b = np.random.rand(n, N)
    b = b / b.sum(axis=1)[:, np.newaxis]
    b_cdf = b.cumsum(axis=1)
    p0 = np.random.rand(n)
    p0 = p0 / p0.sum()

    n_trajs = 10
    n_steps = 10
    dtrajs = []
    otrajs = []
    for _ in range(n_trajs):
        dtraj = np.zeros(n_steps, dtype=int)
        otraj = np.zeros(n_steps, dtype=int)
        s = np.random.choice(np.arange(n), size=None, p=p0)
        for t in range(n_steps):
            dtraj[t] = s
            o = np.searchsorted(b_cdf[s, :], np.random.rand())
            otraj[t] = o
            s = np.searchsorted(T_cdf[s, :], np.random.rand())
        dtrajs.append(dtraj)
        otrajs.append(otraj)


    msm = MSM(n_states=n)
    msm.estimate(otrajs, 1)
    emm = DiscreteEmissionModel()
    emm.initialize(msm, otrajs)
    emm.estimate(msm.chi)

    hmm =  HMM(otrajs, 1)
    hmm.estimate()

    # compare b to hmm.b
    # compare

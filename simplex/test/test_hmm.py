import simplex.hmm as hmm
import numpy as np
import unittest

def T_matrix(energy):
    n = energy.shape[0]
    metropolis = energy[np.newaxis, :] - energy[:, np.newaxis]
    metropolis[(metropolis < 0.0)] = 0.0
    selection = np.zeros((n,n))
    selection += np.diag(np.ones(n-1)*0.5,k=1)
    selection += np.diag(np.ones(n-1)*0.5,k=-1)
    selection[0,0] = 0.5
    selection[-1,-1] = 0.5
    metr_hast = selection * np.exp(-metropolis)
    for i in range(metr_hast.shape[0]):
        metr_hast[i, i] = 0.0
        metr_hast[i, i] = 1.0 - metr_hast[i, :].sum()
    return metr_hast

def tower_sample(cdf):
    return np.searchsorted(cdf, np.random.rand() * cdf[-1])

def generate(T, N_steps, s0=0):
    dtraj = np.zeros(N_steps, dtype=int)
    s = s0
    T_cdf = T.cumsum(axis=1)
    for t in range(N_steps):
        dtraj[t] = s
        s = np.searchsorted(T_cdf[s, :], np.random.rand())
    return dtraj

class TestHmm(unittest.TestCase):
    def setup(cls):
        pass
    
    #def test_hmm(self):
    #    N = 10
    #    ref_vertices = np.random.randn((N+1, N)) #  these define the "random transformation"; make sure non-degenerate
    #    # sample from 
        # u are uniform random numbers
        # c is the globsal steepness of the core 
        # m_self = np.log((np.exp(-c) + 1)*np.exp(u*c) - 1)/(2*c) + 0.5 #  this leads to very fuzzy states
        # m_other = np.random.rand(N)
        # m_other = m_other / (m_other.sum() - m_self)
        # m[i] = m_self
        # m[other] = m_other
        # now apply random non-singular linear transform (fixed for all the data) ideally use orthogonal transform

    def test_discrete_hmm(self):
        n_hidden_states = 5
        n_observed_states = 50
        energy = np.random.rand(n_hidden_states)
        reordering= np.argsort(energy)[::-1]
        T = T_matrix(energy)[reordering, :][:, reordering]

        dtrajs = [ generate(T, 10000) for _ in range(10) ]

        emission_matrix = np.zeros((n_hidden_states, n_observed_states))
        cdf_emission_matrix = emission_matrix.cumsum(axis=1)
        otrajs = []
        for dtraj in dtrajs:
            otraj = np.zeros_like(dtraj)
            for i, state in enumerate(dtraj):
                otraj[i] = tower_sample(cdf_emission_matrix[state, :])
            otrajs.append(otraj)

        model = hmm.HMM(otrajs, 1, initial_model=hmm.MSM(n_states=5)).estimate()
        reordering = np.argsort(model.pi)
        np.testing.assert_allclose(model.T[reordering, :][:, reordeirng], T)
        np.testing.assert_allclose(model.emission_model.b[reordering, :], emission_matrix)


if __name__ == '__main__':
    unittest.main()

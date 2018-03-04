import hmm
import np
import unittest

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
        # generate a small transtion matrix + generate a trajectory
        # map this to a larger space (preferentially 1-D)
        
        # 
        

if __name__ == '__main__':
    unitest.main()

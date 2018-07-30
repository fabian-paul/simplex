import unittest
import simplex
import numpy as np

class TestMemberships(unittest.TestCase):
    def test_assignement(self):
        data = [ np.random.randn(100, 3) for _ in range(2) ]
        vertices = simplex.find_vertices_inner_simplex(data)
        assert vertices.shape[1]==3
        ctrajs1 = simplex.core_assignments(data, vertices)
        memberships = simplex.memberships(data, vertices)
        ctrajs2 = simplex.core_assignment_given_memberships(memberships)
        np.testing.assert_allclose(ctrajs1, ctrajs2)

    def test_projection(self):
        data = [ np.random.randn(100, 3) for _ in range(2) ]
        vertices = simplex.find_vertices_inner_simplex(data)
        memberships = simplex.memberships(data, vertices)
        memberships_corrected = simplex.membership_simplex_projection(memberships)
        eps = 1.E-14
        for m in memberships_corrected:
            assert np.all(m >= -eps)
            assert np.all(m <= 1.0 + eps)
            np.testing.assert_allclose(m.sum(axis=1), 1)

class TestUsingPerfectSimplex(unittest.TestCase):
    def setUp(self):
        dim = 3
        N_pts = 10000
        vol = 0
        real_vertices = None
        for _ in range(100):  # generate vertices of random and non-degenerate simplex
            trial_vertices = np.random.rand(dim + 1, dim)
            trial_vol = np.abs(np.linalg.det(trial_vertices[1:, :] - trial_vertices[0, :]))
            if trial_vol > vol:
                vol = trial_vol
                real_vertices = trial_vertices

        X = np.random.rand(N_pts, dim + 1)
        X = X / X.sum(axis=1)[:, np.newaxis]
        X = X.dot(real_vertices)

        self.real_vertices = real_vertices
        self.X = np.concatenate((X, real_vertices))

    def test_score(self):
        # implicit membership computation
        score, score_by_state = simplex.simplex_misfit(self.X, self.real_vertices, per_state=True)
        tol = -10*np.finfo(np.array(score).dtype).eps
        np.testing.assert_array_less(tol, score)
        np.testing.assert_array_less(tol, score_by_state)

        # explicit membership computation
        mems = simplex.memberships(self.X, self.real_vertices)
        score, score_by_state = simplex.simplex_misfit(mems, None, per_state=True)
        np.testing.assert_array_less(tol, score)
        np.testing.assert_array_less(tol, score_by_state)


if __name__ == '__main__':
    unittest.main()
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


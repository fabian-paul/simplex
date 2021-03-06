# Copyright (c) 2017, Fabian Paul, Computational Molecular Biology Group,
# Freie Universitaet Berlin and Max Planck Institute of Colloids and Interfaces
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""PCCA for the TICA space and the splash projection.

.. moduleauthor:: Fabian Paul <fab@zedat.fu-berlin.de>, Tim Hempel <tim.hempel@fu-berlin.de>

"""

from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.spatial
import scipy.optimize
import warnings


def _othonormalize(vertices):
    # pick vertex that is closest to zero as the origin
    i0 = np.argmin(np.linalg.norm(vertices, axis=1))
    v0 = vertices[i0, :]

    # for the remaining vertices, subtract v0 and othonormalize
    a = np.concatenate((vertices[0:i0, :], vertices[i0+1:, :])) - v0[np.newaxis, :]
    q, _ = np.linalg.qr(a.T)
    return v0, q.T


def _projector(vectors, min_1=True):
    dim = vectors.shape[1]
    X = np.zeros((dim, dim))
    for i in range(vectors.shape[0]):
        X += np.outer(vectors[i, :], vectors[i, :])
    if min_1:
        X -= np.eye(dim)
    return X


def _log_simplex_volume(vertices):
    if vertices.shape[1] > vertices.shape[0] - 1: # more dimensions than vertices
        #print('putting in plane')
        # Replace vertices by the coordiantes of the vertices in the low-dimensional
        # space that they span.
        o, W = _othonormalize(vertices)
        vertices = (vertices-o).dot(W.T)
        assert vertices.shape[1] == vertices.shape[0] - 1

    dim = vertices.shape[1]

    i0 = np.argmin(np.linalg.norm(vertices, axis=1))
    v0 = vertices[i0, :]
    A = np.concatenate((vertices[0:i0, :], vertices[i0+1:, :])) - v0[np.newaxis, :]

    direct_1 = np.log(abs(np.linalg.det(A))) - sp.special.gammaln(dim+1)
    direct_2 = np.log(abs(np.linalg.det(A)) / sp.misc.factorial(dim))

    # http://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
    G = np.linalg.cholesky(A.dot(A.T))
    #print('diagonal', np.diag(G))
    log_vol = np.log(np.diag(G)).sum() - sp.special.gammaln(dim+1)
    if not np.allclose(log_vol, direct_1):
        warnings.warn('optimized volume computation does not agree with direct implementation 1')
    if not np.allclose(log_vol, direct_2):
        warnings.warn('optimized volume computation does not agree with direct implementation 2')
    #print('stable', log_vol)
    return log_vol


def _order_from_rank(rank):
    # TODO: start from the absolute maximum?
    N = rank.shape[0]
    assert rank.shape[1]==N
    rank = rank.copy()
    order = np.zeros(N, dtype=int)
    for i in range(N):
        j = np.argmax(rank[:, i])
        order[i] = j
        rank[j, :] = -np.inf

    assert len(np.unique(order)) == len(order)
    return order


def _vertex_order(vertices):
    # vertex #0 is the one that is closest to the origin of the coordinate system
    i0 = np.argmin(np.linalg.norm(vertices, axis=1))
    v0 = vertices[i0, :]
    others = np.concatenate((np.arange(0, i0), np.arange(i0+1, vertices.shape[0])))
    other_vertices = vertices[others, :]

    # order the rest by closest canonical (Cartesian) axis
    N = other_vertices.shape[0]

    rank = np.zeros((N, N))
    for i in range(N):
        for j in range(N): 
            rank[i, j] = abs(other_vertices[i, j]) #-(sum(vertices[i, :]**2)-vertices[i, j]**2)

    order = _order_from_rank(rank)

    return np.concatenate(([i0], others[order]))


def _source(input_):
    # for integration with pyemma
    if isinstance(input_, (list, tuple, np.ndarray)):
        return _MyDataInMemory(input_)
    else: # assume that we have a source
        return input_


class _MyDataInMemory(object):
    # for integration with pyemma
    def __init__(self, input_):
        if not isinstance(input_, (list, tuple)) and not (isinstance(input_, np.ndarray) and input_.dtype==np.dtype('O') and input_.dim==1):
            input_ = [ input_ ]
        input_  = [ traj.reshape((-1, 1)) if traj.ndim==1 else traj for traj in input_ ]
        for traj in input_:
            if traj.shape[1] != input_[0].shape[1]:
                raise ValueError('input trajectories must have same number of dimensions')
        self._trajs = input_
        self.dtype = input_[0].dtype

    def dimension(self):
        return self._trajs[0].shape[1]

    def iterator(self, return_trajindex=False, stride=1):
        return _MyDataInMemoryIterator(self._trajs, return_trajindex, stride)

    def trajectory_lengths(self, stride=1):
        return [t[::stride, :].shape[0] for t in self._trajs]


class _MyDataInMemoryIterator(object):
    # for integration with pyemma
    def __init__(self, data, return_trajindex, stride):
        self._data = data
        self._pos = 0
        self._stride = stride
        self._return_trajindex = return_trajindex
        self.pos = 0 # we always start at the beginning of a trajectory
    def __iter__(self):
        return self
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass
    def __next__(self):
        if self._pos < len(self._data):
            i = self._pos
            self._pos += 1
            if self._return_trajindex:
                return i, self._data[i][::self._stride, :]
            else:
                return self._data[i][::self._stride, :]
        else:
            raise StopIteration
    next = __next__


def core_assignments(input_, vertices, f=0.6, d=0.0, return_n_inside=False):
    r"""Assign every row of input_ to that vertex to which is has the highest membership.

        parameters
        ----------
        input_ : list of np.ndarray((n_time_steps, n_dims))
            the input data
        vertices : np.ndarray((n_dims+1, n_dims))
            coordiantes of the vertices
        f : float, default = 0.5
            Cut-off for the PCCA membership. Frames with a membership lower than f are left unassigned.
            f typically takes a value between 0 and 1.
        d : float, default = 0.0
            Cut-off for the PCCA membership. Let m be the largest membership of a frame and let m'
            be the second-largest membership of the same frame (to a different vertex). Then the
            frame is only assigned if m >= m' + d. (Typical choice of d could be 0.1)
            This leaves unassigned stripes between cores. Useful for points far outside the simplex,
            where cores defined solely by the f cut-off touch.
        return_n_inside : bool, default=False
            return the number of points in input_ that are located inside
            the simplex spanned by the vertices

        returns
        -------
        depending on the value of `return_n_inside`:
        *  dtrajs
        *  (dtrajs, n_inside)

        dtrajs : list of np.ndarray(n_time_steps, dtype=int)
            For every assigned frame, the index of the vertex with highest membership.
            For frames that are unassigned, the value -1.
        n_inside : int
            number of points in input_ that are located inside the simplex
    """
    # if integrated into pyemma, will become a streamingestimatortransformer
    data = _source(input_)

    ndim = vertices.shape[1]

    M = np.vstack((vertices.T, np.ones(vertices.shape[0])))
    lu_and_piv = sp.linalg.lu_factor(M)

    dtrajs = [ np.zeros(l, dtype=int)-1 for l in data.trajectory_lengths() ]

    n_inside = 0

    it = data.iterator(return_trajindex=True)
    with it:
        for itraj, chunk in it:
            for i, x in enumerate(chunk[:, 0:ndim]):
                #l = np.linalg.solve(M, np.concatenate((x, [1])))
                l = sp.linalg.lu_solve(lu_and_piv, np.concatenate((x, [1]))) # these are the memberships
                # see https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates
                if np.all(l>=0): n_inside += 1
                k, j = np.argsort(l)[-2:]
                if l[j] > f and l[j] >= l[k] + d:
                    dtrajs[itraj][it.pos + i] = j

    if return_n_inside:
        return dtrajs, n_inside
    else:
        return dtrajs


def core_assignment_given_memberships(memberships_, f=0.6, d=0.0):
    data = _source(memberships_)
    dtrajs = [ np.zeros(l, dtype=int) - 1 for l in data.trajectory_lengths() ]

    it = data.iterator(return_trajindex=True)
    with it:
        for itraj, chunk in it:
            L = chunk.shape[0]
            ks, js = np.argsort(chunk, axis=1)[:, -2:].T
            if d != 0: # case decision gives speedup if no logical and operation is needed
                assigned_idx = np.logical_and(chunk[np.arange(L), js] > f,
                                              chunk[np.arange(L), js] > chunk[np.arange(L), ks] + d)
            else:
                assigned_idx = chunk[np.arange(L), js] > f

            dtrajs[itraj][it.pos:it.pos + L][assigned_idx] = js[assigned_idx]

    return dtrajs


def memberships(input_, vertices, dtype=None):
    r"""Computes the membership to all vertices for all frames in input trajectories.

        parameters
        ----------
        input_ : list of np.ndarray((n_time_steps, n_dims))
            the input data
        vertices : np.ndarray((n_dims+1, n_dims))
            coordiantes of the vertices
        dtype : numpy data type, optional
            select the data type for the memberships, allows to set
            the precision, e. g. `np.float32` (to save memory)
            default is the data type of the input data

        returns
        -------
        memberships

        dtrajs : list of np.ndarray((n_time_steps, n_dims+1), dtype=int)
            For every frame, memberships to all vertices.
            In this algorithm (Weber & Galliat 2002), memberships can be negative.
    """
    data = _source(input_)
    if dtype is None:
        dtype = data.dtype

    ndim = vertices.shape[1]

    M = np.vstack((vertices.T, np.ones(vertices.shape[0])))
    lu_and_piv = sp.linalg.lu_factor(M)

    memberships = [ np.zeros((l, ndim+1), dtype=dtype) for l in data.trajectory_lengths() ]

    it = data.iterator(return_trajindex=True)
    with it:
        for itraj, chunk in it:
            for i, x in enumerate(chunk[:, 0:ndim]):
                #l = np.linalg.solve(M, np.concatenate((x, [1])))
                m = sp.linalg.lu_solve(lu_and_piv, np.concatenate((x, [1]))) # these are the memberships
                # see https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates
                memberships[itraj][it.pos + i] = m

    return memberships


def find_vertices_inner_simplex(input_, return_means=False, f_centers=float('-inf'), return_log_volume=False, order_by_axis=True, take_max_dims=float('inf')):
    r'''Find vertices of the "inner simplex". This is the old PCCA algorithm from Weber & Galliat 2002.

    parameters
    ----------
    input_ : list of np.ndarray((n_time_steps, n_dims)) or pyemma iterable
        The input data.
    return_means : bool, default = False
        assign frames to cores and return the mean of each core
    f_centers : float, default = -inf
        f parameter used in the assigment, if return_means=True. 
        See `core_assignments` for more information.
    return_log_volume : bool, default = False
        whether to compute the volume of the simplex
    order_by_axis : bool, default = True
        order vertices (and means) by the IC they maximize
    take_max_dims : integer, default = infinity
        limit the number of dimenions in input_ to take_max_dim. By default use all dimensions.

    returns
    -------
    Depending on the value of return_means, return_log_volume
    *  vertices
    *  (vertices, means)
    *  (vertices, log_volumes)
    *  (vertices, means, log_volumes)    
    
    vertices : np.ndarray((n_dims+1, n_dims))
        coordinates of the n_dims+1 vertices
    means : np.ndarray((n_dims+1, n_dims))
        means of the n_dims+1 cores
    log_volumes : list
        log of volume of the simplex after adding the n'th vertex. Starts with 3 vertices.
    '''
    # inner simplex algorithm (a.k.a. old PCCA, Weber & Galliat 2002) for large number of data points
    data = _source(input_)

    dim = min(data.dimension(), take_max_dims)

    # First find the two most distant vertices. We use the following heuristic:
    # The two points with the largest separation in a simplex should be among those that 
    # lie on the (axes-parallel, Cartesian) bounding box of the points. E.g. in 2-D 
    # a simplex is a triangle. Even if the triangle has an obtuse angle, two of its
    # vertices will lie on the bounding box. In 3-D two (or more) vertices of a 
    # tetrahedron will lie on the bounding box while up to two vertices will dangle 
    # in midair, etc.
    maxima = np.zeros(dim) - np.inf
    minima = np.zeros(dim) + np.inf
    min_pts = np.empty((dim, dim))
    max_pts = np.empty((dim, dim))

    # first pass
    print('pass 1')
    it = data.iterator(return_trajindex=False)
    with it:
        for chunk in it:
            for x in chunk[:, 0:dim]:
                wh = x < minima
                minima[wh] = x[wh]
                min_pts[wh, :] = x
                wh = x > maxima
                maxima[wh] = x[wh]
                max_pts[wh, :] = x

    # Among all the points on the bounding box, pick the ones with largest separation.
    ext_pts = np.concatenate((max_pts, min_pts))
    d = sp.spatial.distance.squareform(sp.spatial.distance.pdist(ext_pts))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    vertices = np.empty((2, dim))
    vertices[0, :] = ext_pts[i]
    vertices[1, :] = ext_pts[j]

    # further passes, follow the algorithm form Weber & Galliat
    for k in range(2, dim+1): # find dim+1 vertices
        print('pass', k)
        v0, w = _othonormalize(vertices)
        P = _projector(w, min_1=True)
        candidate = vertices[-1, :]
        d = 0.0
        it = data.iterator(return_trajindex=False)
        with it:
            for chunk in it:
                for frame in chunk[:, 0:dim]:
                    d_candidate = np.linalg.norm(P.dot(frame-v0))
                    if d_candidate > d:
                        candidate = frame
                        d = d_candidate
        vertices = np.vstack((vertices, candidate))

    if order_by_axis:
        order = _vertex_order(vertices)
    else:
        order = np.arange(len(vertices))

    if return_means:
        centers = np.zeros((dim+1, dim))
        counts = np.zeros(dim+1, dtype=int)
        dtrajs = core_assignments(input_, vertices, f=f_centers)
        it = data.iterator(return_trajindex=True)
        with it:
            for itraj, chunk in it:
                for x, d in zip(chunk[:, 0:dim], dtrajs[itraj][it.pos:]):
                    counts[d] += 1
                    centers[d, :] += x

        centers /= counts[:, np.newaxis]

    if return_log_volume:
        ordered_vertices = vertices[order, :]
        log_volumes = [ ]
        for i in range(3, dim+1):
            log_volumes.append(_log_simplex_volume(ordered_vertices[0:i, :]))

    if return_means and return_log_volume:
        return vertices[order, :], centers[order, :], log_volumes
    elif return_means:
        return vertices[order, :], centers[order, :]
    elif return_log_volume:
        return vertices[order, :], log_volumes
    else:
        return vertices[order, :]

    #from ipywidgets import FloatProgress
    #from IPython.display import display
    #f = FloatProgress(min=0, max=10)
    #display(f)
    #f.value += 1


def _membership_simplex_projection_singlevec(vec):
    r"""
    Implementation of canonical simplex projection by Chen & Ye, 2011, ArXiv:1101.6081v2
    Projects n-dimensional data onto the canonical n-simplex
    Parameters
    ----------
    vec: input vector
    Returns projected vector
    -------
    """
    idx_order = [np.where(np.sort(vec) == _elem)[0][0] for _elem in vec]  # TODO: np.arssort?
    vec = np.sort(vec)

    n = vec.shape[0]
    i = n - 1

    while i > 0:
        t_i = (vec[i:].sum() - 1) / (n - i)
        if t_i >= vec[i - 1]:
            t = t_i
            break

        elif i >= 1:
            i -= 1
            continue

    if i == 0:
        t = (vec.sum() - 1) / n

    x = (vec - t) * (vec - t > 0).astype(float)

    return x[idx_order]


def membership_simplex_projection(membership_trajs):
    r"""
    Project memberships onto canonical simplex to compute positive
    membership trajs
    Parameters
    ----------
    input_: membership trajectories
    Returns positive defined membership trajectories
    -------
    """
    data = _source(membership_trajs)
    projection = [ np.zeros((l, data.dimension()), dtype=data.dtype) for l in data.trajectory_lengths() ]

    it = data.iterator(return_trajindex=True)
    with it:
        for itraj, chunk in it:
                for idx, y in enumerate(chunk):
                    x = _membership_simplex_projection_singlevec(y)
                    projection[itraj][it.pos + idx] = x

    return projection


def splash_corner_projection(vertices, center=0, n_dim_target=2, max_iter=100):
    r"""Compute a projection matrix that represents an even embedding of the vertices into a target space with dimension n_dim_target.

        parameters
        ----------
        vertices : np.ndarray((n_dims_source+1, n_dims_source))
            coordiantes of the vertices
        center : int, optional, default = None
            index of the vertex to put at the coordinate origin in the target space
            By default, the vertex closest ot the coodinate origin in the source space is selected.
        n_dim_target : int, default = 2
            dimension of the target space
        max_iter : int, default = 100
            For n_dim_target >= 3, the projection is found by searching for
            a local optimum of the Thomson problem. max_iter limits the
            number of iterations of the minimizer.

        returns
        -------
        (P, o)
        P : np.ndarray((n_dims_source, n_dim_target))
            the projection matrix
        o : np.ndarray((n_dims_source))
            the shift vector

        To apply the projection to your data `d`, compute `(d-o).dot(P)`
    """

    N = vertices.shape[0] - 1
    if center is None:
        center = np.argmin(np.linalg.norm(vertices, axis=1))
    elif center < 0:
        center = vertices.shape[0] + center
    o = vertices[center, :]
    W = np.concatenate((vertices[0:center, :], vertices[center+1:, :])) - o
    if n_dim_target == 2:
        L = np.empty((N, 2))
        for i in range(N):
            L[i, 0] = np.sin(2.0*np.pi*i/float(N))
            L[i, 1] = np.cos(2.0*np.pi*i/float(N))
    elif n_dim_target>2:
        raise Exception('n_dim_target > 2 is no longer supported')
    else:
        raise Exception('n_dim must be an integer > 1')
    return np.linalg.inv(W).dot(L), o


def membership_ring_projection(n_vertices):
    r"""Similar to splash_corner_projection but for memberships.
    """
    L = np.empty((n_vertices, 2))
    for i in range(n_vertices):
        L[i, 0] = np.sin(2.0*np.pi*i/float(n_vertices))
        L[i, 1] = np.cos(2.0*np.pi*i/float(n_vertices))
    return L


def softmax(membership_trajs, beta=1):
    r"""Softmax function exp(beta*m_i)/[sum_j exp(beta*m_j)]

    Parameters
    ----------
    membership_trajs: list of np.ndarray((n_time_steps_i, n_dims)) or pyemma iterable
        membership trajectories
    beta : float or np.ndarray(n_dims)
        the factor beta in the softmax function exp(beta*m_i)/[sum_j exp(beta*m_j)]

    Returns
    -------
    list of np.ndarray((n_time_steps_i, n_dims))
    """
    data = _source(membership_trajs)

    beta = np.array(beta)
    if beta.ndim==0:
        beta = np.ones(data.dimension())*beta

    transformed = [ np.zeros((l, data.dimension()), dtype=data.dtype) for l in data.trajectory_lengths() ]

    it = data.iterator(return_trajindex=True)
    with it:
        for itraj, chunk in it:
            smax = np.exp(beta[np.newaxis, :]*chunk)
            norm = smax.sum(axis=1)
            transformed[itraj][it.pos:it.pos+chunk.shape[0]] = smax / norm[:, np.newaxis]

    return transformed


def _past(ctraj):
    past = np.zeros(len(ctraj), dtype=int)
    last_s = ctraj[0]
    for i, s in enumerate(ctraj):
        if s>=0:
            last_s = s
        past[i] = last_s
    return past

def _future(ctraj):
    future = np.zeros(len(ctraj), dtype=int)
    next_s = ctraj[-1]
    for i, s in zip(np.arange(len(ctraj))[::-1], ctraj[::-1]):
        if s>=0:
            next_s = s
        future[i] = next_s
    return future

def milestoning_count_matrix(dtrajs, lag=1, n_states=None, return_mass_matrix=False, return_scrapped=False):
    r"""Computed the Milestoning covariance matrix and mass matrix as described in [1]_

        parameters
        ----------
        dtrajs : list of np.ndarray((n_time_steps, ), dtype=int)
            Core label trajectories. Frames that are not assigned to any
            core, should take a strictly neagtive value, e. g. -1.
        lag : int, default = 1
            the lag time
        n_states : int, optional
            determines the shape of the returned matrices. If None, use
            the maximum core index in dtrajs + 1.
        return_mass_matrix : bool, optional, default = False
            whether to compute and return the mass matrix (a. k. a. the
            instantaneous covariance matrix of the core committors)
        return_scrapped : bool, optional, default = False
            whether to return the number of frames that were scrapped
            in the computation of the matrices. Frames at the beginning
            or the end of a trajectory that are not assigned to any
            core, are scrapped.

        returns
        -------
        Depending on the value of return_mass_matrix, return_scrapped
        *  c
        *  (c, m)
        *  (c, n)
        *  (c, m, n)

        c : np.ndarray((n_states, n_states), dtype=int)
            the time-lagged covartiance matrix of the core committors
        m : np.ndarray((n_states, n_states), dtype=int)
            the mass matrix
        n : int
            number of frames that weren't used in estimating the matrices

    References
    ----------
    .. [1] Ch. Schuette & M. Sarich, Eur. Phys. J. Spec. Top. 244, 2445 (2015)
    """
    import warnings
    warnings.warn('Milestoning code is not thoroughly tested, to be safe, please use pyemma.')
    if n_states is None:
        n_states = max([np.max(d) for d in dtrajs]) + 1
        assert n_states >= 1

    c = np.zeros((n_states, n_states), dtype=int)
    if return_mass_matrix:
        m = np.zeros((n_states, n_states), dtype=int)
    n_scrapped = 0

    for d in dtrajs:
        if np.any(d>=0):
            # cut off transition state pieces at the end and beginning
            first_idx = next(i for i, s in enumerate(d) if s>=0)
            last_idx = len(d) - next(i for i, s in enumerate(d[::-1]) if s>=0)
            if last_idx - first_idx <= lag:
                n_scrapped += len(d)
                continue
            n_scrapped += first_idx
            n_scrapped += len(d)-last_idx
            d = d[first_idx:last_idx]
            # generate past and future
            past = np.zeros(len(d), dtype=int)
            last_s = d[0]
            for i, s in enumerate(d):
                if s>=0:
                    last_s = s
                past[i] = last_s
            future = np.zeros(len(d), dtype=int)
            next_s = d[-1]
            for i, s in zip(np.arange(len(d))[::-1], d[::-1]):
                if s>=0:
                    next_s = s
                future[i] = next_s
            # fill count matrix
            for p, f in zip(past[0:-lag], future[lag:]):
                c[p, f] += 1
            if return_mass_matrix:
                for p, f in zip(past[0:-lag], future[0:-lag]):
                    m[p, f] += 1
        else:
            n_scrapped += len(d)

    if return_mass_matrix and return_scrapped:
        return c, m, n_scrapped
    elif return_mass_matrix:
        return c, m
    elif return_scrapped:
        return c, n_scrapped
    else:
        return c


def connected_set_assignments(dtrajs, cset, return_dtrajs=False):
    r"""For every frame in dtrajs, indicate whether the frame is in the connected set.

        For a frame to belong to the connected set, neither the last core
        hit nor the next core hit must not occur in `cset`.

        parameters
        ----------
        dtrajs : list of np.ndarray((n_time_steps,), dtype=int)
            Markov or Milestoning trajejectories. Frames not assigned
            to any cory should take the value -1.
        cset : list-like of integers
            list of Markov states or cores that are to be considered
            as forming the connected set

        returns
        -------
        *  assingments if return_dtrajs is false
        *  (assignements, new_dtrajs) if return_dtrajs is true

        assingments : list of np.ndarray((n_time_steps,), dtype=bool)
            list of arrays of the same shapes as dtrajs. An element is
            True if the frame belongs to the connected set.
        new_dtrajs :  list of np.ndarray((n_time_steps,), dtype=int)
            like the input parameter dtrajs, but with every inactive
            time step set to -1
    """
    # mark all time steps as belonging to the connected set
    # if the past and the future of the time step both belong
    # to cores that are in `cset`. If either of past or future is
    # undefined, ignore it, and decide on the time direction that
    # is defined.
    cset_trajs = [ np.ones(len(dtraj), dtype=bool) for dtraj in dtrajs ]

    cset = np.concatenate((cset, [-1])) # TODO: generalize to other negative numbers

    for d, cset_traj in zip(dtrajs, cset_trajs):
        # compute past and future
        past = _past(d)
        future = _future(d)
        # do the assignment
        for i, (p, f) in enumerate(zip(past, future)):
            if p not in cset or f not in cset:
                cset_traj[i] = False
            else:
                cset_traj[i] = True

        # rules:
        # core in cset -> core in cset :  True
        # core not in cset -> core in cset : False
        # core not in cset -> core not in cset : False
        # core in cset -> core not in cset : False
        # no core -> core in cset : True
        # no core -> core not in cset : False
        # core in cset -> no core : True
        # core not in cset -> no core : False
        # no core -> no core : True (why? likely not absorbing?, state will stay -1 anyway)

    if return_dtrajs:
        dtrajs_deactivated = [ np.where(c, d, -1) for d, c in zip(dtrajs, cset_trajs)]

    if return_dtrajs:
        return cset_trajs, dtrajs_deactivated
    else:
        return cset_trajs

def _lifetimes(ctraj):
    lifetimes = np.zeros(len(ctraj), dtype=int) # number of repeats passed in the current block
    last_s = ctraj[0]
    l = 0
    for i, s in enumerate(ctraj):
        if s!=last_s:
            lifetimes[i] = 0
            l = 1
            last_s = s
        else:
            lifetimes[i] = l
            l += 1
    return lifetimes

def assign_transitions_to_cores(core_to_core_dtrajs, keep_ends_unassigned=True, weak_cores=[]):
    r"""Convert core-to-core trajecotories to conventional discrete
    trajectories that formally mimic a full state space partitioning [1]_.

    Parameters
    ----------
    core_trajs : list of np.ndarray((n_time_steps,), dtype=int)
        Milestoning (core-to-core) trajectories. Unassigned time steps
        take negative values.
    keep_ends_unassigned : boolean, default = True
        Whether to leave the beginnings and the ends of trajectories
        that don't belong to any core unassigned.
    weak_cores : list of integers, optional, default = []
        The core whose indices are given in this list are not considered
        when assigning the transitions regions. That is the whole
        transition is assigned to the other core involved in the transition.

    Returns
    -------
    dtrajs : list of np.ndarray((n_time_steps,), dtype=int)
        fully assigned discrete trajectories (except for beginnings/ends,
        depending on the value of keep_ends_unassigned)

    References
    ----------
    .. [1] Buchete & Hummer J. Phys. Chem. B, 112, 6057 (2008)
    """
    dtrajs = [ np.zeros(len(ctraj), dtype=int)-1 for ctraj in core_to_core_dtrajs ]
    if not keep_ends_unassigned:
        weak_cores.append(-1)

    for ctraj, dtraj in zip(core_to_core_dtrajs, dtrajs):
        # compute past and future
        past = _past(ctraj)
        future = _future(ctraj)
        future_length = _lifetimes(ctraj[::-1])[::-1]
        past_length = _lifetimes(ctraj)

        # do the assignment
        # rules:
        # strong core -> strong core: assign half-half
        # weak core -> strong core: assign to strong core
        # strong core -> weak core: assign to strong core
        # weak core -> weak core: assign half-half
        # no core -> strong core: don't assign
        # no core -> weak core:  don't assign
        for i, (c, p, f, pl, fl) in enumerate(zip(ctraj, past, future, past_length, future_length)):
            if c<0:
                if (p<0 or f<0) and keep_ends_unassigned:
                    dtraj[i] = c # don't assign end/beginning
                elif (p in weak_cores and f in weak_cores) or (p not in weak_cores and f not in weak_cores):
                    if pl<fl: # past and future of same kind + i in first half
                        dtraj[i] = p
                    else:
                        dtraj[i] = f # both defined + i in second half
                elif p in weak_cores and f not in weak_cores:
                    dtraj[i] = f
                elif p not in weak_cores and f in weak_cores:
                    dtraj[i] = p
                else:
                    raise Exception('unanticipated case')
            else:
                assert p==f==c
                dtraj[i] = c

    return dtrajs


def _rle(ia):
    # adapted from Thomas Browne's answer on http://stackoverflow.com/a/32681075
    n = len(ia)
    y = np.array(ia[1:] != ia[:-1])
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return z, p, ia[i]

def filter_out_short_core_visits(ctrajs, cutoff=1, fill=True, weak_cores=[]):
    r'''Relabel short visits to cores in a trajectory as unassigned.

    Parameters
    ----------
    ctrajs : list of np.ndarray((n_time_steps,), dtype=int)
        Milestoning (core-to-core) trajectories. Unassigned time steps
        take negative values.
    cutoff : integer, default = 1
        Visits of this length (or shorter) are relabeled.
    fill: boolean, default=True
        If true, assign excursions from a core back to the same core
        before removing short core visits. An excursion is a trajectory
        piece [..., c, -1, -1,  -1, c, ...] with the same core c
        at the beginning and the end.
    weak_cores : list of integers
        Short visits to weak cores are never removed.

    Retuns:
    -------
    list of np.ndarray((n_time_steps,), dtype=int)
    same content as ctrajs but with short visits relabeled
    '''
    filtered = []
    for ctraj in ctrajs:
        if fill:
            temp = ctraj.copy()
            past = _past(ctraj)
            future = _future(ctraj)
            for i, (c, p, f) in enumerate(zip(ctraj, past, future)):
                if c<0 and p==f:
                    temp[i] = p
        ctraj = temp

        filtered_ctraj = np.zeros(len(ctraj), dtype=int)
        for length, position, value in zip(*_rle(ctraj)):
            if length<=cutoff and value>=0 and value not in weak_cores:
                filtered_ctraj[position:position+length] = -1
            else:
                filtered_ctraj[position:position+length] = value
        filtered.append(filtered_ctraj)

    return filtered

def life_time_distributions(ctrajs, fill=True):
    r'''Compute life times for all cores.

    Parameters
    ----------
    ctrajs : list of np.ndarray((n_time_steps,), dtype=int)
        Milestoning (core-to-core) trajectories. Unassigned time steps
        take negative values.
    fill: boolean, default=True
        If true, assign excursions from a core to the core before
        computing life times. An excursion is a trajecotory
        piece [..., c, -1, -1,  -1, c, ...] with the same core c
        at the beginning and the end.

    Returns
    -------
    d : dictonary where the keys are core ids
    d[i] is a list of life times of core i
    '''
    import collections
    hist = collections.defaultdict(list)

    for ctraj in ctrajs:
        if fill:
            temp = ctraj.copy()
            past = _past(ctraj)
            future = _future(ctraj)
            for i, (c, p, f) in enumerate(zip(ctraj, past, future)):
                if c<0 and p==f:
                    temp[i] = p
            ctraj = temp

        for length, _, value in zip(*_rle(ctraj)):
            hist[value].append(length)
    return hist


def scatter_mem(memberships, ctrajs=None, selection=range(0, 4), center=True, ax=None, max_plot=10000, scatter_kwargs={'s':5}, decoration=False):
    r'''Show 2-D scatter plot of memberships

    Parameters
    ----------
    memberships : list of ndarray((T_i, N))
        memberships off all frames to all N vertices

    ctrajs : list of ndarray((T_i,), dtype=int)
        discrete trajectories, unassigned frames should take negative values

    selection : list or range
        selection of cores to show with colors

    center : boolean, default = True
        If true, plot frames that belong to the non-selected cores
        around the coordiante origin in the scatter plot.
        If false, allocate a sector on the unit disc to plot
        all frames that belong to non-selected cores.

    ax : pyplot.axes object or None

    max_plot : integer
         limit number of points plotted for each core to max_plot
         (prevents pyplot from crashing due to memory errors)

    scatter_kwargs : dict
         dictionary of keyword arguments that are passed to plt.scatter
    '''
    import matplotlib.pyplot as plt
    #if ctrajs is None:
    #    import simplex
    #    ctrajs = simplex.core_assignment_given_memberships(memberships, f=0.6, d=0.1)

    N = memberships[0].shape[1]
    X = np.zeros(N)
    Y = np.zeros(N)
    n = len(selection)
    if not center and n<N:
        n+=1
    assert n<=N
    for i, j in enumerate(selection):
        X[j] = np.sin(i*(2.0*np.pi/n))
        Y[j] = np.cos(i*(2.0*np.pi/n))
    if not center:
        for i in range(N):
            if i not in selection:
                X[i] = np.sin((n-1)*(2.0*np.pi/n))
                Y[i] = np.cos((n-1)*(2.0*np.pi/n))

    colors = ['darkcyan', 'orange', 'darkgreen', 'red', 'black', 'purple', 'lime']

    if ax is None:
        ax = plt.gca()

    if decoration:
        ax.add_artist(plt.Circle((0, 0), 1.0, color='gray', fill=False))
        for i in range(n):
            x = np.sin((i+0.5)*(2.0*np.pi/n))
            y = np.cos((i+0.5)*(2.0*np.pi/n))
            ax.add_line(plt.Line2D([0.0, x], [0.0, y], color='gray'))
            ax.set_aspect('equal')

    # scatter selected
    for i, col in zip(selection, colors):
        n_frames = sum([np.count_nonzero(c==i) for c in ctrajs])
        stride = max(n_frames // max_plot, 1)
        wh = [ np.where(c==i)[0][::stride] for c in ctrajs ]
        x = np.concatenate([ m[w, :].dot(X) for m, w in zip(memberships, wh) ])
        y = np.concatenate([ m[w, :].dot(Y) for m, w in zip(memberships, wh) ])
        ax.scatter(x, y, c=col, **scatter_kwargs)
        #ax.scatter(np.arctan2(x, y), (x*x+y*y)**0.5, c=col, **scatter_kwargs)

    # scatter the rest
    for i in range(N):
        if i not in selection:
            n_frames = sum([np.count_nonzero(c==i) for c in ctrajs])
            stride = max(n_frames // max_plot, 1)
            wh = [ np.where(c==i)[0][::stride] for c in ctrajs ]
            x = np.concatenate([ m[w, :].dot(X) for m, w in zip(memberships, wh) ])
            y = np.concatenate([ m[w, :].dot(Y) for m, w in zip(memberships, wh) ])
            ax.scatter(x, y, c='gray', **scatter_kwargs)
            ##ax.scatter(np.arctan2(x, y), (x*x+y*y)**0.5, c='gray', **scatter_kwargs)

    # add labels
    for s in selection:
        plt.text(X[s], Y[s], str(s))
    if len(selection) < N:
        if not center:
            plt.text(np.sin((n-1)*(2.0*np.pi/n)), np.cos((n-1)*(2.0*np.pi/n)), 'rest')
        else:
            plt.text(0, 0, 'rest')

    return ax


def pcca_score(memberships, autocorrect=True):
    r'''See Roeblitz, Weber, Adv. Data. Anal. Classif. 7, 147 (2013)
    '''
    # TODO: think about doing this with ctrajs + direct assigment + milestoning
    # ctrajs -> cov, direct -> diagonal pi (alternative: use past)
    if autocorrect:
        if np.any(m<0 for m in memberships) or np.any(m>1 for m in memberships):
            memberships = membership_simplex_projection(memberships)
    nc = memberships[0].shape[1]
    D2c_inv = np.diag(1./sum(m.sum(axis=0) for m in memberships))
    chit_D2_chi = sum(m.T.dot(m) for m in memberships)
    return nc - np.trace(D2c_inv.dot(chit_D2_chi))


def _membership_computer(vertices):
    r'''Return a function that computes memberships from a chunk of ICs.
    '''
    M = np.vstack((vertices.T, np.ones(vertices.shape[0])))
    del vertices
    lu_and_piv = sp.linalg.lu_factor(M)
    n_memberships = M.shape[0]
    del M
    def function(chunk):
        out = np.zeros((chunk.shape[0], n_memberships), chunk.dtype)
        for i, frame in enumerate(chunk):
            out[i, :] = sp.linalg.lu_solve(lu_and_piv, np.concatenate((frame, [1])))
        return out
    return function

# compute simple SSE/SST score
def sse_sst_score(input_, vertices=None):
    data = _source(input_)

    if vertices is not None:
        membership_computer = _membership_computer(vertices)
        n_states = vertices.shape[0]
        n_dim = vertices.shape[1]
    else:
        membership_computer = lambda x: x  # identity
        n_states = data.dimension()
        n_dim = data.dimension()

    means = np.zeros(n_states, dtype=data.dtype)
    counts = np.zeros(n_states, dtype=int)

    # compute means
    it = data.iterator(return_trajindex=False)
    with it:
        for chunk in it:
            mems = membership_computer(chunk[:, 0:n_dim])
            state = np.argmax(mems, axis=1)
            means[state] += mems[state, :]
            counts[state] += 1

    nonzero = counts > 0
    means[nonzero] = means[nonzero] / counts[nonzero]

    sse = np.zeros(n_states, dtype=data.dtype)
    sst = np.sum(np.var(means))

    # compute errors
    it = data.iterator(return_trajindex=False)
    with it:
        for chunk in it:
            mems = membership_computer(chunk[:, 0:n_dim])
            state = np.argmax(mems, axis=1)
            max_mems = np.max(mems, axis=1)
            sse[state] += (max_mems - means[state])**2

    sse[nonzero] = sse[nonzero] / counts[nonzero]

    return sse, sst


def simplex_misfit(input_, vertices, minChi=True, extrema=False, per_state=False):
    r'''Compute score that measures the misfit of the memberships to the simplex-structure of the data.

    Essentially this functions computes an averaged version of the minChi error indicator.

    Parameters
    ----------
    input_ : list of ndarray((T_i, N))
        times series of the independent components
        (or time series of the metastable memberships, depending of the value of `vertices`)

    vertices : np.ndarray((n_dim + 1, n_dim)) or None
        coordiantes of the vertices or None.
        If this argument is None, `input_` is treated as metastable memberships,
        else input_ is treated as independent components.

    minChi : bool, optional, default=True
        If true, compute the minChi score (either the averaged or extremal version,
        depending on the value of `extrema`). If false, also include violations
        of the simplex structure, where memberships are larger than one into the
        score.

    extrema : bool, optional, default=False
        If set to True, search for the single strongest violation of the simplex
        structure, instead of computing an average misfit (extrema=False). This
        is very close to the original minChi criterion. However this is sensitive
        to outliers.

    per_state : bool, optional, defeault=False
        return misfit measure resolved by metastable state

    Returns
    -------
    * `(maximal_misfit, per_state_misfit)` if `per_state` is true
    * `maximal_misfit` else

    maximal_misfit : float
        the smaller, the worse the fit (values > 0 indicate perfect fit)
    per_state_misfit : np.ndarray(n_states, dtype=float)
        misfit resolved per metastable state
    '''
    data = _source(input_)

    if vertices is not None:
        membership_computer = _membership_computer(vertices)
        n_states = vertices.shape[0]
        n_dim = vertices.shape[1]
    else:
        membership_computer = lambda x: x  # identity
        n_states = data.dimension()
        n_dim = data.dimension()

    max_misfit = np.zeros(n_states, dtype=data.dtype)
    total_misfit = np.zeros(n_states, dtype=data.dtype)
    n_misfits = np.zeros(n_states, dtype=int)

    # internally we work with the negative of the score (the larger, the worse)
    it = data.iterator(return_trajindex=False)
    with it:
        for chunk in it:
            mems = membership_computer(chunk[:, 0:n_dim])
            if minChi:
                delta = -np.minimum(mems, 0.0)
            else:
                delta = np.maximum(-np.minimum(mems, 0.0), np.maximum(mems - 1.0, 0.0))
            if extrema:
                max_misfit = np.maximum(np.max(delta, axis=0), max_misfit)
            else:
                is_misfit = (delta > 0).astype(int)
                n_misfits_chunk = np.sum(is_misfit, axis=0)
                total_misfit_chunk = np.sum(is_misfit*delta, axis=0)
                n_misfits += n_misfits_chunk
                total_misfit += total_misfit_chunk

    if extrema:
        typical_misfit = max_misfit
    else:
        typical_misfit = np.zeros(n_states, dtype=float) - 1.  # default to perfect fit
        gt_zero = n_misfits > 0
        typical_misfit[gt_zero] = total_misfit[gt_zero] / n_misfits[gt_zero]

    if per_state:
        return -np.max(typical_misfit), -typical_misfit
    else:
        return -np.max(typical_misfit)


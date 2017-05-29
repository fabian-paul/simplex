import numpy as np
import scipy as sp
import scipy.sparse
import msmtools


def _dtraj_correlation(a, b, lag=0, weights=None, sliding=True, sparse=False, nstates=None, skip=0, rskip=0, normalize=False, future_weights=False):
    # Adapted from msmtools.estimation.count_matrix
    assert sliding, 'sliding=False not tested'
    rows = []
    cols = []
    values = []
    assert len(a)==len(b)
    if weights is None:
        weights = [ np.ones(len(x)) for x in a ]
    else:
        assert len(weights)==len(a)
    # collect transition index pairs
    for x, y, z in zip(a, b, weights):
        assert len(x)==len(y)==len(z)
        if x.size > lag+skip+rskip: 
            if sliding:
                rows.append(x[skip:len(x)-lag-rskip])
                cols.append(y[lag+skip:len(y)-rskip])
                if not future_weights:
                    values.append(z[skip:len(z)-lag-rskip])
                else:
                    values.append(z[lag+skip:len(y)-rskip])
            else: # TODO: check order of lag and skip
                rows.append(x[skip:len(x)-lag-rskip:lag])
                cols.append(y[lag+skip:len(y)-rskip:lag])
                values.append(z[skip:len(z)-lag-rskip:lag])
    # is there anything?
    if len(rows) == 0:
        raise ValueError('No counts found - lag ' + str(lag) + ' may exceed all trajectory lengths.')
    # feed into one COO matrix
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    values = np.concatenate(values)
    if nstates is None:
        nstates = max(np.max(row), np.max(col)) + 1
    if normalize:
        values /= np.sum(values)
    C = sp.sparse.coo_matrix((values, (row, col)), shape=(nstates, nstates))
    # export to output format
    if sparse:
        return C.tocsr()
    else:
        return C.toarray()


def _past(ctrajs):
    pasts = []
    for ctraj in ctrajs:
        past = np.zeros(len(ctraj), dtype=int)
        if len(ctraj)>0:
            last_s = ctraj[0]
            for i, s in enumerate(ctraj):
                if s>=0:
                    last_s = s
                past[i] = last_s
        pasts.append(past)
    return pasts


def _future(ctrajs):
    futures = []
    for ctraj in ctrajs:
        future = np.zeros(len(ctraj), dtype=int)
        if len(ctraj)>0:
            next_s = ctraj[-1]
            for i, s in zip(np.arange(len(ctraj))[::-1], ctraj[::-1]):
                if s>=0:
                    next_s = s
                future[i] = next_s
        futures.append(future)
    return futures


def _is_valid(ctrajs, lag=0):
    # label all time steps that have properly defined future and past
    #v = []
    #for a, b in zip(p, f):
    #     v.append(np.array([ x>=0 and y>=0 for x, y in zip(a, b) ]))
    #return v
    slices = []

    for ctraj in ctrajs:
        if np.any(ctraj>=0):
            # cut off transition state pieces at the end and beginning
            first_idx = next(i for i, s in enumerate(ctraj) if s>=0)
            last_idx = len(ctraj) - next(i for i, s in enumerate(ctraj[::-1]) if s>=0)
            if last_idx - first_idx <= lag:
                #n_scrapped += len(d)
                slices.append(slice(0)) # empty slice
                continue
            #n_scrapped += first_idx
            #n_scrapped += len(d)-last_idx
            slices.append(slice(first_idx, last_idx))
        else:
            slices.append(slice(0)) # empty slice

    return slices


def _restrict_to_valid(v, x):
    return [ y[w] for w, y in zip(v, x) ]

# Meine Idee: immer das oertlich naechste core zum umgewichten nehmen
# Heuristik: Ausdrucke der Art \sum_i u_i*q_i(x) werden durch den aktuell groessten committor
# dominiert. Welcher committor der groeste ist, leasst sich durch die Naehe abschaetzen (greedy). 
# Evtl. koennte man auch mit den (positiven, auf die neachste simplex-Flaeche projezierten) memberships arbeiten.

# Frank's Idee: immer das letzte core zum umgewichten nehmen.
# Heuristik: meistens ist das letzte auch das neachste core, weil die Trajektorie meistens in der Naehe des letzten cores bleibt
# Ist nicht klar, ob das eine guter Ansatz zum umgewichten ist, weil das evtl. stark davon abhanget, wie viel gesampelt wurde.
# Es scheint mir noch heuristischer zu sein, als mein Ansatz.

# Frank's Idee 2: MSM aus past traj. scheatzen


def simple_reversible_milestoning(ctrajs, lag=1):
    p = _past(ctrajs)
    p_valid = []
    for x in p:
        if np.any(x>=0):
            first = next(i for i,s in enumerate(x) if s>=0)
            p_valid.append(x[first:])
    C = msmtools.estimation.count_matrix(p_valid, lag=lag)
    C = msmtools.estimation.largest_connected_submatrix(C, directed=True).toarray()
    return msmtools.estimation.tmatrix(C, reversible=True)

# TODO: offer some option for computing the eigendecomposition

def reversible_milestoning(ctrajs, dtrajs=None, lag=1, counting_mode='forward', return_Ct=False):
    v = _is_valid(ctrajs, lag=lag)
    ctrajs = _restrict_to_valid(v, ctrajs)
    p = _past(ctrajs)
    f = _future(ctrajs)
    assert np.all(np.concatenate(p)>=0)
    assert np.all(np.concatenate(f)>=0)
    if dtrajs is not None:
        have_dtrajs = True
        dtrajs = _restrict_to_valid(v, dtrajs)
    else:
        have_dtrajs = False
        dtrajs = p
    assert np.all(np.concatenate(dtrajs)>=0)

    C0 = _dtraj_correlation(p, f, rskip=lag)
    Ct = _dtraj_correlation(p, f, lag=lag)
    assert Ct.sum() == C0.sum()
    #C0 = _dtraj_correlation(p, f, normalize=True)
    #Ct = _dtraj_correlation(p, f, lag=lag, normalize=True)

    cset = msmtools.estimation.largest_connected_set(Ct, directed=True)
    C0 = C0[cset, :][:, cset]
    Ct = Ct[cset, :][:, cset]

    K = np.linalg.inv(C0).dot(Ct)
    l, V = np.linalg.eig(K.T)
    b = V[:, np.argmax(l)]
    b = b / b.sum() 

    ok = [  np.logical_and(np.in1d(x, cset), np.in1d(y, cset)).astype(np.float) for x, y in zip(p, f) ]
    G = _dtraj_correlation(p, dtrajs, weights=ok, normalize=True, rskip=lag)
    G = G[cset, :][:, cset]
    if not have_dtrajs:
        # for Frank's algorithm G should really be C0
        G = C0 / C0.sum()
    #u_cset = np.linalg.inv(G).dot(b) # TODO: don't have to invert, just SOLVE
    u_cset = np.linalg.solve(G, b)
    u_full = np.zeros(max(np.max(d) for d in dtrajs) + 1)
    u_full[cset] = u_cset
    print u_full

    w = [ u_full[d] for d in dtrajs ]
    print np.concatenate(w).sum()/len(np.concatenate(w)),
    if not np.allclose(np.concatenate(w).sum()/len(np.concatenate(w)), 1):
        print 'warning'
    else:
        print
    if counting_mode=='forward':
        # TODO: derive some way for symmetrizing and keeping the eigenvalue 1
        C0_forw = _dtraj_correlation(p, f, weights=w, normalize=True, rskip=lag)
        Ct_forw = _dtraj_correlation(p, f, lag=lag, weights=w, normalize=True)
        C0_sym = C0_forw #+ C0_forw.T
        Ct_sym = Ct_forw #+ Ct_forw.T
    elif counting_mode=='forward-backward':
        # this really not working!
        C0_forw_1 = _dtraj_correlation(p, f, weights=w, rskip=lag, normalize=True)
        C0_back_1 = _dtraj_correlation(f, p, weights=w, rskip=lag, normalize=True)
        C0_forw_2 = _dtraj_correlation(p, f, weights=w, skip=lag, normalize=True)
        C0_back_2 = _dtraj_correlation(f, p, weights=w, skip=lag, normalize=True)
        Ct_forw = _dtraj_correlation(p, f, lag=lag, weights=w, normalize=True)
        Ct_back = _dtraj_correlation(f, p, lag=lag, weights=w, normalize=True)
        #C0_sym = C0_forw + C0_forw.T + C0_back + C0_back.T
        C0_sym = C0_forw_1 + C0_forw_2 + C0_back_1 + C0_back_2
        C0_sym = 0.5*(C0_sym + C0_sym.T)
        Ct_sym = Ct_forw + Ct_forw.T + Ct_back + Ct_back.T

    C0_sym = C0_sym[cset, :][:, cset]
    Ct_sym = Ct_sym[cset, :][:, cset]

    K_rev = np.linalg.inv(C0_sym).dot(Ct_sym)

    if return_Ct:
        return K, K_rev, Ct
    else:
        return K, K_rev


def cross_validation_score(ctrajs, lag, k):
    scores = []
    for indices in []:
         K_train = reversible_milestoning(ctrajs[train_indices], lag)
         K_test = reversible_milestoning(ctrajs[test_indices], lag)
         #V = reversible_decomposition(C0, Cts)
         #scores.append(V.T.dot())
         
    return np.mean(scores)
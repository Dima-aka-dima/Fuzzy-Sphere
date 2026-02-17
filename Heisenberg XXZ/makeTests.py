from hamiltonian_hb_xxz import get_hamiltonian_sparse


Ls = [3, 4, 6, 7, 10]
Js = [1.4, 0.4, 1.2, 0.0, 1.1]
deltas = [0.0, 0.3, -0.5, 0.2, 0.8]
sz = [0, 2, -1, 4, 3]
ks = [2, 0, 1, 7, 2]

for L, J, delta, sz, k in zip(Ls, Js, deltas, sz, ks):
    rows, cols, data = get_hamiltonian_sparse(L, J, delta, sz, k)
    print(L, rows, cols, data)


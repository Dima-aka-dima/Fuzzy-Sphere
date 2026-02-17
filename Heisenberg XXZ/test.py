from hamiltonian_hb_xxz import get_hamiltonian_sparse
import numpy as np


rows, cols, data = get_hamiltonian_sparse(18, 1.0, 0.5, 0, 2)

print(f"Hamiltonian size: {len(data)}")
# np.set_printoptions(precision=2, suppress=True)
# print(data)
# print("Done!")

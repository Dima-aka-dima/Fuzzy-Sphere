import numpy as np
from typing import Tuple

N = 18
S = 0
K = 2
J = 1.0
D = 0.5

def balanced_word(n, k):
	x = 0
	for i in range(n):
		if ((i + 1) * k // n) - (i * k // n):
			x |= (1 << (n - i - 1))
	return x

def get_site_value(state, site):
	return (state >> site) & 1

def set_site_value(state, site, value):
	site_value = int(value) << site
	return (state ^ site_value) | site_value

def get_first_state():
	n_up = N // 2 + S
	return (1 << n_up) - 1

def get_last_state():
	n_up = N // 2 + S
	return ((1 << n_up) - 1) << (N - n_up)

def next_state(state):
	t = (state | (state - 1)) + 1
	return t | ((((t & -t) // (state & -state)) >> 1) - 1)

def translate(state, translation):
	mask = (1 << N) - 1
	window = (state << N) | state
	return (window >> translation) & mask

def get_representative_and_translation(state):
	mask = (1 << N) - 1
	window = (state << N) | state
	
	r = state
	translation = 0
	
	for t in range(1, N):
		window = window >> 1
		s = window & mask
		if s < r:
			r = s
			translation = t
	
	return r, translation

def get_representative(state):
	mask = (1 << N) - 1
	window = (state << N) | state
	
	r = state
	for t in range(1, N):
		window = window >> 1
		s = window & mask
		r = min(r, s)
	
	return r

if __name__ == "__main__":

    # Build basis
    basis = []
    norms = []

    state = get_first_state()
    last_state = get_last_state()

    while state <= last_state:
        representative = get_representative(state)
        
        if state == representative:
            amplitude = 0j
            
            for t in range(N):
                new_state = translate(state, t)
                if new_state == state:
                    amplitude += np.exp(2.0j * np.pi * t * K / N)
            
            norm = np.sqrt(np.abs(amplitude))
            
            if norm > 1e-12:
                basis.append(state)
                norms.append(norm)
        
        state = next_state(state)

    print(f"Number of representatives: {len(basis)}")

    # Build index lookup
    max_representative = balanced_word(N, N // 2 + S)
    indices = np.zeros(max_representative + 1, dtype=int)
    indices[basis] = np.arange(len(basis))

    # Create bonds
    bonds = [(i, (i + 1) % N) for i in range(N)]

    # Initialize sparse matrix storage
    rows = []
    cols = []
    data = []


    # Diagonal elements
    for state_index in range(len(basis)):
        state = basis[state_index]
        
        diagonal = 0.0
        for bond in bonds:
            if get_site_value(state, bond[0]) == get_site_value(state, bond[1]):
                diagonal += J / 4
            else:
                diagonal -= J / 4
        
        rows.append(state_index)
        cols.append(state_index)
        data.append(diagonal)


    # Off-diagonal elements
    for state_index in range(len(basis)):
        state = basis[state_index]
        
        for bond in bonds:
            if get_site_value(state, bond[0]) == get_site_value(state, bond[1]):
                continue
            
            flipmask = (1 << bond[0]) | (1 << bond[1])
            new_state = state ^ flipmask
            
            representative, translation = get_representative_and_translation(new_state)
            new_state_index = indices[representative]
            
            coefficient = (J + D) / 4
            coefficient *= norms[new_state_index] / norms[state_index]
            coefficient *= np.exp(2.0j * np.pi * translation * K / N)
            
            rows.append(new_state_index)
            cols.append(state_index)
            data.append(coefficient)
            
            rows.append(state_index)
            cols.append(new_state_index)
            data.append(np.conj(coefficient))

    print(f"Hamiltonian size: {len(rows)}")

    # Convert to numpy arrays
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    # print(data)

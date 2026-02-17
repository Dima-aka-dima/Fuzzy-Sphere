import numpy as np


# Functions to manipulate states
def get_site_value(state, site):
    ''' Function to get local value at a given site '''
    return (state >> site) & 1

def set_site_value(state, site, value):
    ''' Function to set local value at a given site '''
    site_val = (value << site)
    return (state ^ site_val) | site_val

def first_state(L, sz):
    ''' Return first state of Hilbert space in lexicographic order '''
    n_upspins = L//2 + sz
    return (1 << n_upspins) - 1

def next_state(state):
    '''
    Return next state of Hilbert space in lexicographic order

    This function implements is a nice trick for spin 1/2 only,
    see http://graphics.stanford.edu/~seander/bithacks.html
    #NextBitPermutation for details
    '''
    t = (state | (state - 1)) + 1
    return t | ((((t & -t) // (state & -state)) >> 1) - 1)

def last_state(L, sz):
    ''' Return last state of Hilbert space in lexicographic order '''
    n_upspins = L//2 + sz
    return ((1 << n_upspins) - 1) << (L - n_upspins)

def translate(L, state, n_translation_sites):
    ''' translates a state by n_translation_sites '''
    new_state = 0
    for site in range(L):
        site_value = get_site_value(state, site)
        new_state = set_site_value(new_state, (site + n_translation_sites)%L, site_value)
    return new_state

def get_representative(L, state):
    ''' finds representative and representative translation for a state '''
    representative = state
    translation = 0
    for n_translation_sites in range(L):
        new_state = translate(L, state, n_translation_sites)
        if new_state < representative:
            representative = new_state
            translation = n_translation_sites
    return representative, translation

L = 4
sz = 0
k = 2

# check if sz is valid
assert (sz <= (L // 2 + L % 2)) and (sz >= -L//2)

# Create list of representatives and norms
basis_states = []
norms = []

state = first_state(L, sz)
# print(format(state, f"0{L}b"))
end_state = last_state(L, sz)
while state <= end_state:
    representative, translation = get_representative(L, state)
    if state == representative:

        # Compute the normalization constant
        amplitude = 0.0
        for n_translation_sites in range(L):
            new_state = translate(L, state, n_translation_sites)
            if new_state == state:
                amplitude += np.exp(1j*2*np.pi*k/L*n_translation_sites)
        norm = np.sqrt(np.abs(amplitude))
        if norm > 1e-12:
            basis_states.append(state)
            norms.append(norm)

    state = next_state(state)
print(len(basis_states))
# print([state for state in basis_states])
# print([format(state, f"0{L}b") for state in basis_states])
# print(norms)

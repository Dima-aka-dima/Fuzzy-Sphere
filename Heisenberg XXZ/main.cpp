#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <bit>
#include "types.hpp"

// using state_t = std::bitset<N>;
using state_t = u32;
using index_t = u32;

const u32 N = 4;
const i32 S = 0;
const index_t K = 2;
const f64 J = 1.0;
const f64 D = 0.5;


constexpr index_t balancedWord(state_t n, index_t k) 
{
	index_t x = 0;
	for (index_t i = 0; i < n; ++i) 
		if (((i + 1) * k / n) - (i * k / n)) x |= (state_t(1) << (n - i - 1));
	return x;
}

bool getSiteValue(state_t state, index_t site)
{
	return (state >> site) & 1;
}

state_t setSiteValue(state_t state, index_t site, bool value)
{
	state_t siteValue = (state_t)value << site;
	return (state ^ siteValue) | siteValue;
}

constexpr state_t getFirstState()
{
	index_t nUp = N / 2 + S;
	return (state_t(1) << nUp) - 1;
}

constexpr state_t getLastState()
{
	index_t nUp = N / 2 + S;
	return ((state_t(1) << nUp) - 1) << (N - nUp);
}

state_t nextState(state_t state)
{
	state_t t = (state | (state - 1)) + 1;
	return t | ((((t & -t) / (state & -state)) >> 1) - 1);
}


// Represent state as two states, so that we only shift instead of rotating [.....][.....]
// Only works for N <= 32

using window_t = u64;
state_t translate(state_t state, index_t translation)
{
	constexpr state_t mask = (state_t(1) << N) - 1;
	window_t window = (window_t(state) << N) | state;
	return (window >> translation) & mask;
}

std::pair<state_t, index_t> getRepresentativeAndTranslation(state_t state)
{
	constexpr state_t mask = (state_t(1) << N) - 1;
	window_t window = (window_t(state) << N) | state;
	
	state_t r = state;
	index_t translation = 0;

	for(index_t t = 1; t < N; ++t)
	{
		window = (window >> 1);
		state = window & mask;
		if(state < r)
		{
			r = state;
			translation = t;
		}
	}

	return {r, translation};
}

state_t getRepresentative(state_t state)
{
	constexpr state_t mask = (state_t(1) << N) - 1;
	window_t window = (window_t(state) << N) | state;

	state_t r = state;
	for(index_t t = 1; t < N; ++t)
	{
		window = (window >> 1);
		state = window & mask;
		r = std::min(r, state);
	}

	return r;
}

// Assumes sorted and existence
template<typename T>
index_t index(std::vector<T>& v, T value)
{
	return std::distance(v.begin(), std::lower_bound(v.begin(), v.end(), value));
}

i32 main()
{
	static_assert(abs(S) <= (N/2 + N % 2));
	static_assert(N <= 32, "Larger N require u128");

	std::vector<state_t> basis; 
	// basis.reserve(binomial(N, N / 2 + S) / N);
	std::vector<f64> norms; 
	// norms.reserve(binomial(N, N / 2 + S) / N);
	
	for(state_t state = getFirstState(); state <= getLastState(); state = nextState(state))
	{
		state_t representative = getRepresentative(state);
		
		if(state == representative)
		{
			std::complex<f64> amplitude = 0;

			for(index_t t = 0; t < N; t++)
			{
				state_t newState = translate(state, t);
				if(newState == state) 
					amplitude += std::exp(2.0*I*pi*f64(t)*f64(K) / f64(N)); // After finding the first one we can extrapolate
			}
			
			f64 norm = std::sqrt(std::abs(amplitude));

			if(norm > 1e-12)
			{
				basis.push_back(state);
				norms.push_back(norm);
			}
		
		}
	}
		
	std::cout << "Number of representatives: " << basis.size() << std::endl;
	// std::cout << basis << std::endl;
	// std::cout << norms << std::endl;
	
	std::array<std::pair<index_t, index_t>, N> bonds;
	for(index_t i = 0; i < N; i++) bonds[i] = {i, (i + 1) % N};
	

	// Making them into std::array doesn't help. 
	// Separating into diag/offdiag doesn't help either.
	// Precomputing count and resizing also doesn't help.
	// std::vector<index_t, PlatformAllocator<index_t, 2> > rows, cols; 
	// std::vector<std::complex<f64>, PlatformAllocator<std::complex<f64>, 8>> data; 
	std::vector<index_t> rows, cols; 
	std::vector<std::complex<f64>> data; 
	rows.resize(basis.size());
	cols.resize(basis.size()); 
	data.resize(basis.size());
	
	index_t count = 0;	
	for(index_t stateIndex = 0; stateIndex < basis.size(); stateIndex++)
	{
		state_t state = basis[stateIndex];

		f64 diagonal = 0;
		for(auto bond: bonds)
		{
			if(getSiteValue(state, bond.first) == getSiteValue(state, bond.second)) {diagonal += J/4;}
			else { diagonal -= J/4; count++;}
		}

		rows[stateIndex] = stateIndex;
		cols[stateIndex] = stateIndex;
		data[stateIndex] = diagonal;

	}

	rows.resize(count*2 + basis.size());
	cols.resize(count*2 + basis.size());
	data.resize(count*2 + basis.size());
	
	constexpr index_t maxRepresentative = balancedWord(N, N/2 + S); // ~O(2^(N - 1))
	// static std::array<index_t, maxRepresentative + 1> indices;
	// static std::vector<index_t, PlatformAllocator<index_t, 16>> indices(maxRepresentative + 1, 0);
	// static index_t indices[maxRepresentative + 1];
	// auto indices = std::make_unique<index_t[]>(maxRepresentative + 1);
	index_t* indices = new index_t[maxRepresentative + 1];
	for(index_t i = 0; i < basis.size(); i++) indices[basis[i]] = i;
	// std::unordered_map<index_t, state_t> indices; for(index_t i = 0; i < basis.size(); i++) indices[basis[i]] = i;
	
	sz index = basis.size();
	for(index_t stateIndex = 0; stateIndex < basis.size(); stateIndex++)
	{
		state_t state = basis[stateIndex];
		for(auto bond: bonds)
		{
			if(getSiteValue(state, bond.first) == getSiteValue(state, bond.second)) continue;
			
			state_t flipmask = (1 << bond.first) | (1 << bond.second);
			state_t newState = state ^ flipmask; // Precomputing flipmasks is slower
			
			auto [representative, translation] = getRepresentativeAndTranslation(newState); // Can be hashed, but will be slower
			index_t newStateIndex = indices[representative]; // Can be replaced with binary search index, same time

			std::complex<f64> coefficient = (J + D) / 4;
			coefficient *= norms[newStateIndex]/norms[stateIndex];
			coefficient *= std::exp(2.0*I*pi*f64(translation)*f64(K) / f64(N));
			
			rows[index] = newStateIndex;
			cols[index] = stateIndex;
			data[index] = coefficient;
			index++;
			
			rows[index] = stateIndex;
			cols[index] = newStateIndex;
			data[index] = std::conj(coefficient);
			index++;
		}
	}

	std::cout << "Hamiltonian size: " << rows.size() << std::endl;
	std::cout << data << std::endl;
	// if(N < 10) std::cout << rows << std::endl << cols << std::endl << data << std::endl;
	// writeCSV("hamiltonian.csv", std::make_tuple(rows, cols, data));
}

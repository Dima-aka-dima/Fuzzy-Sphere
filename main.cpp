#include <iostream>
#include "types.hpp"

#include <bit>
#include <bitset>
#include <array>
#include <vector>
#include <unordered_map>

namespace gsl
{
#include <gsl/gsl_sf_coupling.h>
auto wigner3j = gsl_sf_coupling_3j;
}

#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>


#define push push_back
#define pop pop_back


// Assumes `s` is integer (for now)
const sz N = 5; // |[s, s-1, ..., -s+1, -s]| = 2s + 1 = N, \sum_{m = -s}^s c^\dagger_m c_m = N  -- half-filling
const i32 M = 0; // \sum_{m = -s}^s m c^\dagger_m c_m 
const i32 S = N / 2;

const f64 h = 0.0;
const f64 V0 = 1.0;
const f64 V1 = 4.0;


using state_t = u32; // [s, s-1, ..., -s+1, -s]
using table_t = std::array<std::array<std::array<std::array<f64, N>, N>, N>, N>;


table_t makeCoefficients()
{
	table_t table;
	i32 l;
	f64 C1, C2;
	for(i32 m1 = -S; m1 <= S; m1++)
	for(i32 m2 = -S; m2 <= S; m2++)
	for(i32 m3 = -S; m3 <= S; m3++)
	for(i32 m4 = -S; m4 <= S; m4++)
	{
		if (m1 + m2 != m3 + m4) continue;

		l = 0;
		C1 = gsl::wigner3j(2*S, 2*S, 2*(2*S-l), 2*m1, 2*m2, 2*(-m1-m2)); 
		C2 = gsl::wigner3j(2*S, 2*S, 2*(2*S-l), 2*m4, 2*m3, 2*(-m3-m4)); 
		table[m1+S][m2+S][m3+S][m4+S] += V0*f64(4*S-2*l+1)*C1*C2;

		l = 1;
		C1 = gsl::wigner3j(2*S, 2*S, 2*(2*S-l), 2*m1, 2*m2, 2*(-m1-m2)); 
		C2 = gsl::wigner3j(2*S, 2*S, 2*(2*S-l), 2*m4, 2*m3, 2*(-m3-m4)); 
		table[m1+S][m2+S][m3+S][m4+S] += V1*f64(4*S-2*l+1)*C1*C2;
	}	

	return table;
}

const table_t V = makeCoefficients(); 


bool get(state_t state, sz index){ return (state >> index) & 1; }
state_t set(state_t state, sz index, bool value) { return (state & ~(state_t(1) << index)) | (state_t(value) << index); }


state_t flip(state_t state, sz index){ return state ^ (state_t(1) << index); }


// Comparison is correctly overloaded for std::pair by default
template<typename T>
sz getIndex(std::vector<T>& states, T state)
{
	return std::distance(states.begin(), std::lower_bound(states.begin(), states.end(), state));
}

sz getN(state_t up, state_t down)
{
	return std::popcount(up) + std::popcount(down); 
}

i32 getM(state_t up, state_t down)
{
	i32 res = 0;
	for(i32 m = 0; m < 2*S + 1; m++)
	{
		if ( (up >> m) & 1)   res += (m - S);
		if ( (down >> m) & 1) res += (m - S);
	}

	return res;
}

sz getParity(state_t state, sz m1, sz m2)
{
	sz parity = 0;
	for (sz i = (m1 < m2 ? m1 : m2) + 1; i < (m1 < m2 ? m2 : m1); ++i) if (get(state, i)) parity++;
	return parity;
}

i32 main()
{
	static_assert(2*S + 1 == N);
	static_assert(N <= 15);
	std::cout << "S = " << S << ", N = " << N << ", M = " << M << std::endl;

	std::vector<std::pair<state_t, state_t>> states;
	for(state_t up = 0; up < (state_t(1) << N); up++) for(state_t down = 0; down < (state_t(1) << N); down++)
	{
		if(getN(up, down) == N and getM(up, down) == M) states.push({up, down});
	}

	std::cout << "Number of states: " <<states.size() << std::endl;
	// for(auto [up, down]: states) std::cout << "|" << std::bitset<N>(up) << "|" << std::bitset<N>(down) << "|" << std::endl;

	std::vector<sz> rows, cols;
	std::vector<f64> data;

	for(sz index = 0; index < states.size(); index++)
	{
		auto [up, down] = states[index];

		/*
		for(i32 m = 0; m < 2*S + 1; m++)
		{
			// c^\dagger_{m,\uparrow}c_{m, \downarrow} + c^\dagger_{m,\downarrow} c_{m, \uparrow}
			if(get(up, m) == get(down, m)) continue;

			state_t upNext = flip(up, m); 
			state_t downNext = flip(down, m);
			sz indexNext = getIndex(states, {upNext, downNext});

			rows.push(index); cols.push(indexNext); data.push(-h);
			rows.push(indexNext); cols.push(index); data.push(-h);
		}
		*/

		for(i32 m1 = 0; m1 < 2*S + 1; m1++)
		for(i32 m2 = 0; m2 < 2*S + 1; m2++)
		for(i32 m3 = 0; m3 < 2*S + 1; m3++)
		for(i32 m4 = 0; m4 < 2*S + 1; m4++)
		{
			if (m1 + m2 != m3 + m4) continue;

			f64 coefficient = V[m1][m2][m3][m4];

			if(m1 == m4) // m2 == m3
			{
				// c^\dagger_{m_1,\downarrow} c_{m_1,\downarrow} c^\dagger_{m_2,\uparrow} c_{m_2,\uparrow}
				if(get(up, m2) and get(down, m1))
				{
					rows.push(index); cols.push(index); data.push(coefficient);
				}
				
				// c^\dagger_{m_1,\uparrow} c_{m_1,\uparrow} c^\dagger_{m_2,\downarrow} c_{m_2,\downarrow}
				if(get(down, m2) and get(up, m1))
				{
					rows.push(index); cols.push(index); data.push(coefficient);
				}
				continue;
			}

			// c^\dagger_{m_1,\downarrow} c_{m_4,\downarrow} c^\dagger_{m_2,\uparrow} c_{m_3,\uparrow}
			if(get(up,   m3) and not get(up,   m2) and\
			   get(down, m4) and not get(down, m1))
			{
				state_t downNext = set(set(down, m4, 0), m1, 1);
				state_t upNext   = set(set(up,   m3, 0), m2, 1);

				sz parity = getParity(down, m1, m4) + getParity(up, m2, m3);
				f64 phase = (parity % 2) ? -1.0 : 1.0;

				sz indexNext = getIndex(states, {upNext, downNext});
				
				rows.push(index); cols.push(indexNext); data.push(phase*coefficient);
				rows.push(indexNext); cols.push(index); data.push(phase*coefficient);
			}
			
			// c^\dagger_{m_1,\uparrow} c_{m_4,\uparrow} c^\dagger_{m_2,\downarrow} c_{m_3,\downarrow}
			if(get(down, m3) and not get(down, m2) and\
			   get(up,   m4) and not get(up,   m1))
			{
				state_t downNext = set(set(down, m3, 0), m2, 1);
				state_t upNext   = set(set(up,   m4, 0), m1, 1);

				sz parity = getParity(up, m1, m4) + getParity(down, m2, m3);
				f64 phase = (parity % 2) ? -1.0 : 1.0;

				sz indexNext = getIndex(states, {upNext, downNext});
				
				rows.push(index); cols.push(indexNext); data.push(phase*coefficient);
				rows.push(indexNext); cols.push(index); data.push(phase*coefficient);
			}

		}
		
	}

	std::cout << "Number of computed entries: " << data.size() << std::endl;
	// std::cout << rows << std::endl << cols << std::endl;
	// std::cout << data << std::endl;

	std::unordered_map<sz, f64> values;
	for(sz i = 0; i < rows.size(); i++) 
		values[rows[i]*states.size() + cols[i]] += data[i];

	std::cout << "Number of unique entries in the Hamiltonian: " << values.size() << std::endl;
	std::vector<sz> rowsUnique, colsUnique;
	std::vector<f64> dataUnique;

	for(auto [ind, value]: values)
	{
		rowsUnique.push(ind / states.size());
		colsUnique.push(ind % states.size());
		dataUnique.push(value);
	}


	sz nEigenvalues = 6;
	
	Eigen::SparseMatrix<f64> sparse(states.size(), states.size());
	std::vector<Eigen::Triplet<f64>> triplets;
	for (sz i = 0; i < rowsUnique.size(); ++i) triplets.push_back(Eigen::Triplet<f64>(rowsUnique[i], colsUnique[i], dataUnique[i]));
	sparse.setFromTriplets(triplets.begin(), triplets.end());
	
	Eigen::MatrixXd dense = sparse;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solverDense(dense);
	std::vector<f64> eigenvaluesDense(nEigenvalues);
	for(sz i = 0; i < nEigenvalues; i++) eigenvaluesDense[i] = solverDense.eigenvalues()[i];
	std::cout << "Eigenvalues (Eigen): " << eigenvaluesDense << std::endl;
	return 0;

	Spectra::SparseSymMatProd<f64> op(sparse);
	Spectra::SymEigsSolver<Spectra::SparseSymMatProd<f64>> solver(op, nEigenvalues, 8*nEigenvalues); solver.init();
	solver.compute(Spectra::SortRule::SmallestAlge);
	std::vector<f64> eigenvalues(nEigenvalues); 
	for(sz i = 0; i < nEigenvalues; i++) eigenvalues[nEigenvalues - i - 1] = solver.eigenvalues()[i];	
	std::cout << "Eigenvalues (SPECTRA): " << eigenvalues << std::endl;
}

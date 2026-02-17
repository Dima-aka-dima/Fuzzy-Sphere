#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

int main() {
	std::vector<int> rows = {0, 1, 2, 0, 1, 2};
	std::vector<int> cols = {0, 1, 2, 1, 2, 0};
	std::vector<double> data = {4.0, 5.0, 6.0, 1.0, 2.0, 1.0};
	
	int n = 3;
	
	Eigen::SparseMatrix<double> sparse(n, n);
	std::vector<Eigen::Triplet<double>> triplets;
	for (size_t i = 0; i < rows.size(); ++i) {
		triplets.push_back(Eigen::Triplet<double>(rows[i], cols[i], data[i]));
	}
	sparse.setFromTriplets(triplets.begin(), triplets.end());
	
	Eigen::MatrixXd dense = Eigen::MatrixXd(sparse);
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(dense);
	
	std::cout << "Eigenvalues:\n" << solver.eigenvalues() << "\n\n";
	std::cout << "Eigenvectors:\n" << solver.eigenvectors() << "\n";
	
	return 0;
}

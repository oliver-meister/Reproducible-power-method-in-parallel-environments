#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h> 
#include "matrix.h"

using namespace std;

sparseMatrixCSR* assemble_csr_matrix(std::string filePath){
	//number of rows, columns and non-zeroentries
	int M, N, L;
	std::ifstream fin(filePath);
	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');
	// Read defining parameters:
	fin >> M >> N >> L;
	
	int *row_ptr = malloc(sizeof(int) * (M+1));
	int *column = malloc(sizeof(int) * L);
	double *value = malloc(sizeof(double) * L);

	row_ptr[0] = 0;
	int last_row = 0;

	for (int l = 0; l < L; l++){
		int row, col;
		double data;
		fin >> row >> col >> data;
		row--; col--;
		column[l] = col;
		value[l] = data;

		if (row > last_row) {
			for (int r = last_row + 1; r <= row; ++r) {
				row_ptr[r] = l;
			}
        last_row = row;
    	}	
	}

	for (int r = last_row + 1; r <= M; ++r) {
    	row_ptr[r] = L;
	}

	sparseMatrixCSR *A = malloc(sizeof(sparseMatrixCSR));
	A->row_ptr = row_ptr;
	A->col = column;
	A->val = value;
	A->cols = N;
	A->nnz = L;
	A->rows = M;
	fin.close();
	return A;
}

CSR assemble_simetric_csr_matrix(std::string filePath){

	int M, N, L;
	vector<int> rows, cols;
	vector<double> data;
	CSR matrix;
	std::ifstream fin(filePath);
	// Ignore headers and comments:
	while (fin.peek() == '%') fin.ignore(2048, '\n');
	// Read defining parameters:
	fin >> M >> N >> L;	
	matrix.row_ptr.push_back(0);
	for (int l = 0; l < L; l++){
		int row, col;
		double d;
		fin >> row >> col >> d;
		rows.push_back(row);
		cols.push_back(col);
		data.push_back(d);
	}
	fin.close();
	for (int l = 1; l <= M; l++){
		for (int k = 0; k < L; k++){
			if (cols[k] == l){
				matrix.col_ind.push_back(rows[k]);
				matrix.val.push_back(data[k]);					
			}	
			else if (rows[k] == l){
				matrix.col_ind.push_back(cols[k]);
				matrix.val.push_back(data[k]);				
			}
		}
		matrix.row_ptr.push_back(matrix.col_ind.size());
	}
	
	matrix.row_ptr.push_back(matrix.col_ind.size() + 1);
	
	return matrix;
}



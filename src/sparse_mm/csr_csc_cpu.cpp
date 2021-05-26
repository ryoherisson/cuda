/* Reference
https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
*/

#include <algorithm>
#include <iostream>
#include <vector>

// 新しい型名viの作成
typedef std::vector<int> vi;

typedef std::vector<std::vector<int>> matrix;

// Utility Function to print a Matrix
void printMatrix(const matrix& M) {
    int m = M.size();
    int n = M[0].size();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << M[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Utility Function to print val, col_ind, row_prt vectors
// with some decoration.
void printVector(const vi& V, char* msg) {
    std::cout << msg << "[ ";
    std::for_each(V.begin(), V.end(), [](int a) {
        std::cout << a << " ";
    });
    std::cout << "]" << std::endl;
}

// Generate the three vectors val, col_ind, row_prt
void csr_format(const matrix& M) {
    int m = M.size();
    int n = M[0].size(), i, j;
    vi val;
    vi col_ind;
    vi row_ptr = { 0 };
    int nnz = 0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (M[i][j] != 0) {
                val.push_back(M[i][j]);
                col_ind.push_back(j);

                nnz++;
            }
        }
        row_ptr.push_back(nnz);
    }

    printMatrix(M);
    printVector(val, (char*)"val = ");;
    printVector(col_ind, (char*)"col_ind = ");
    printVector(row_ptr, (char*)"row_ptr = ");
}

// Generate the three vectors val, row_ind, col_ptr
void csc_format(const matrix& M) {
    int m = M.size();
    int n = M[0].size(), i, j;
    vi val;
    vi row_ind;
    vi col_ptr = { 0 };
    int nnz = 0;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (M[i][j] != 0) {
                val.push_back(M[i][j]);
                row_ind.push_back(i);

                nnz++;
            }
        }
        col_ptr.push_back(nnz);
    }

    printMatrix(M);
    printVector(val, (char*)"val = ");;
    printVector(row_ind, (char*)"row_ind = ");
    printVector(col_ptr, (char*)"col_ptr = ");
}

int main(int argc, char* argv[]) {
    matrix M = {
        { 0, 0, 0, 0, 1 },
        { 5, 8, 0, 0, 0 },
        { 0, 0, 3, 0, 0 },
        { 0, 6, 0, 0, 1 },
    };

    std::cout << "CSR format" << std::endl;
    csr_format(M);

    std::cout << std::endl;

    std::cout << "CSC format" << std::endl;
    csc_format(M);

    return 0;
}
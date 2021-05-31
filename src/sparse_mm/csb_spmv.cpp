#include <algorithm>
#include <cmath>
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

void csb_format(const matrix& M) {
    int m = M.size();
    int n = M[0].size(), i, j;
    vi val;
    vi blk_row_ind;
    vi blk_col_ind;
    vi blk_ptr = { 0 };
    int nnz = 0;

    int blk_m = std::sqrt((double)m);
    int blk_n = std::sqrt((double)n);




}


int main(int argc, char* argv[]) {
    matrix M = {
        { 0, 0, 0, 0 },
        { 5, 8, 0, 0 },
        { 0, 0, 3, 0 },
        { 0, 6, 0, 0 },
    };

    std::cout << "CSB format" << std::endl;
    csb_format(M);

    std::cout << std::endl;

    return 0;
}
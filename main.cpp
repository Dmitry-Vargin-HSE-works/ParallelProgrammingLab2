#include "matrix.h"
#include "utils.h"

int main(int argc, char** argv) {

    if (argc != 5) {
        std::cout << "Wrong usage! Example: " << std::endl;
        std::cout << "mpiexec -n <num_of_processes> --oversubscribe <execution_file> "
                     "<num_of_left_rows> <num_of_left_columns_and_right_rows> <num_of_right_columns> "
                     "<split_type (0 or 1)>" << std::endl;
        return -1;
    }

    srand(0);
    MPI_Init(&argc, &argv);

    int left_rows = std::stoi(argv[1]);
    int left_columns_right_rows = std::stoi(argv[2]);
    int right_columns = std::stoi(argv[3]);
    SplitType split_type = SplitType(std::stoi(argv[4]));

    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;

    if (isMasterProcess()) {
        A = genRandomMatrix(A, left_rows, left_columns_right_rows);
        B = genRandomMatrix(B, left_columns_right_rows, right_columns);
        C = genDefaultMatrix(C, left_rows, right_columns, 0);
    }

    switch (split_type) {
        case SplitType::Rows:
            multiplyByRows(A, B, C, left_rows, left_columns_right_rows, right_columns);
            break;
        case SplitType::Columns:
            multiplyByColumns(A, B, C, left_rows, left_columns_right_rows, right_columns);
            break;
        default:
            std::cout << "Value error! SplitTipe must be 0 or 1!" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (isMasterProcess()) {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();
    return 0;
}

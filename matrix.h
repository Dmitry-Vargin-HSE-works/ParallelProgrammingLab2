#ifndef LAB2_MATRIX_H
#define LAB2_MATRIX_H


#include <iostream>
#include <mpi.h>
#include <iomanip>
#include "utils.h"

void transposeMatrix(double* matrix, int rows, int cols) {
    auto* temp = (double*) malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp[j * rows + i] = matrix[i * cols + j];
        }
    }
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            matrix[i * rows + j] = temp[i * rows + j];
        }
    }
    free(temp);
}

void printMatrix(const double* matrix, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
}

template <class T>
T *genRandomMatrix(T *matrix, int n, int m, int remained_of = 10) {
    matrix = (T *)malloc(n * m * sizeof(T));
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            matrix[i * m + j] = rand() % remained_of;
        }
    }
    return matrix;
}

template <class T>
T *genDefaultMatrix(T *matrix, int n, int m, int def = 0) {
    matrix = (T *)malloc(n * m * sizeof(T));
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            matrix[i * m + j] = def;
        }
    }
    return matrix;
}

void multiplyByRows(
        double* A, double* B, double* C,
        int left_rows, int left_columns_right_rows, int right_columns) {

    int rank = getProcessNumber(),
            size = getProcessCount();

    double start_time = MPI_Wtime();

    int rowsA_per_task = left_rows / size;
    int rowsB_per_task = left_columns_right_rows / size;
    int extra_Arows = left_rows % size;
    int extra_Brows = left_columns_right_rows % size;

    int* countsA = (int*)malloc(size * sizeof(int));
    int* displsA = (int*)malloc(size * sizeof(int));

    int* countsB = (int*)malloc(size * sizeof(int));
    int* displsB = (int*)malloc(size * sizeof(int));

    int* countsC = nullptr;
    int* displsC = nullptr;
    
    int i, j, k;

    MPI_Allgather(&rowsA_per_task, 1, MPI_INT, countsA, 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Allgather(&rowsB_per_task, 1, MPI_INT, countsB, 1, MPI_INT, MPI_COMM_WORLD);

    int disp = 0;
    for (i = 0; i < size; ++i) {
        displsA[i] = disp;
        if (extra_Arows > 0) {
            countsA[i]++;
            extra_Arows--;
        }
        countsA[i] *= left_columns_right_rows;
        disp += countsA[i];
    }

    disp = 0;
    for (i = 0; i < size; ++i) {
        displsB[i] = disp;
        if (extra_Brows > 0) {
            countsB[i]++;
            extra_Brows--;
        }
        countsB[i] *= right_columns;
        disp += countsB[i];
    }

    if (isMasterProcess()) {
        countsC = (int*)malloc(size * sizeof(int));
        displsC = (int*)malloc(size * sizeof(int));
        for(i = 0; i < size; i++) {
            countsC[i] = countsA[i] / left_columns_right_rows * right_columns;
        }

        displsC[0] = 0;
        for(i = 1; i < size; i++) {
            displsC[i] = displsC[i - 1] + countsC[i - 1];
        }
    }

    int max_rows_per_proc = 0;
    for(i = 0; i < size; i++) {
        if (countsB[i] > max_rows_per_proc) {
            max_rows_per_proc = countsB[i];
        }
    }

    auto *local_A = (double*)malloc(countsA[rank] * sizeof(double));
    auto *local_B = (double*)malloc(max_rows_per_proc * sizeof(double));
    auto *local_C = (double*)malloc(countsA[rank] / left_columns_right_rows * right_columns * sizeof(double));
    for(i = 0; i < countsA[rank] / left_columns_right_rows * right_columns; i++) {
        local_C[i] = 0;
    }

    MPI_Scatterv(A, countsA, displsA, MPI_DOUBLE, local_A, countsA[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, countsB, displsB, MPI_DOUBLE, local_B, countsB[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rank_data, start_row, row_count, iter;
    for (iter = 0; iter < size; ++iter) {
        rank_data = (rank + size - iter) % size;
        start_row = displsB[rank_data] / right_columns;
        row_count = countsB[rank_data] / right_columns;
        for (i = 0; i < countsA[rank] / left_columns_right_rows; ++i) {
            for (j = 0; j < row_count; ++j) {
                for (k = 0; k < right_columns; ++k) {
                    local_C[i * right_columns + k] += local_A[i * left_columns_right_rows + (start_row + j) % left_columns_right_rows] * local_B[j * right_columns + k];
                }
            }
        }
        MPI_Sendrecv_replace(local_B, max_rows_per_proc, MPI_DOUBLE, (rank + 1) % size, 0, (rank + size - 1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Gatherv(local_C, countsA[rank] / left_columns_right_rows * right_columns, MPI_DOUBLE, C, countsC, displsC, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (isMasterProcess()) {
        std::cout <<"Runtime = " << end_time - start_time << std::endl;
        free(countsC);
        free(displsC);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    free(countsA);
    free(displsA);
    free(countsB);
    free(displsB);
}

void multiplyByColumns(
        double* A, double* B, double* C,
        int left_rows, int left_columns_right_rows, int right_columns) {

    double start_time = MPI_Wtime();

    int rank = getProcessNumber(),
        size = getProcessCount();

    int rows_per_task = left_rows / size;
    int columns_per_task = right_columns / size;
    int extra_rows = left_rows % size;
    int extra_columns = right_columns % size;

    if (isMasterProcess()) {
        transposeMatrix(B, left_columns_right_rows, right_columns);
    }

    int *countsA = (int*)malloc(size * sizeof(int));
    int *displsA = (int*)malloc(size * sizeof(int));

    int *countsB = (int*)malloc(size * sizeof(int));
    int *displsB = (int*)malloc(size * sizeof(int));

    int *countsC = nullptr;
    int *displsC = nullptr;

    int i, j, k;

    MPI_Allgather(&rows_per_task, 1, MPI_INT, countsA, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&columns_per_task, 1, MPI_INT, countsB, 1, MPI_INT, MPI_COMM_WORLD);

    int disp = 0;
    for (i = 0; i < size; ++i) {
        displsA[i] = disp;
        if (extra_rows > 0) {
            countsA[i]++;
            extra_rows--;
        }
        countsA[i] *= left_columns_right_rows;
        disp += countsA[i];
    }

    disp = 0;
    for (i = 0; i < size; ++i) {
        displsB[i] = disp;
        if (extra_columns > 0) {
            countsB[i]++;
            extra_columns--;
        }
        countsB[i] *= left_columns_right_rows;
        disp += countsB[i];
    }

    if (isMasterProcess()) {
        countsC = (int*)malloc(size * sizeof(int));
        displsC = (int*)malloc(size * sizeof(int));
        for(i = 0; i < size; i++) {
            countsC[i] = countsA[i] / left_columns_right_rows * right_columns;
        }

        displsC[0] = 0;
        for(i = 1; i < size; i++) {
            displsC[i] = displsC[i - 1] + countsC[i - 1];
        }
    }

    int max_rows_per_proc = 0;
    for(i = 0; i < size; i++) {
        if (countsB[i] > max_rows_per_proc) {
            max_rows_per_proc = countsB[i];
        }
    }

    auto *local_A = (double*)malloc(countsA[rank] * sizeof(double));
    auto *local_B = (double*)malloc(max_rows_per_proc * sizeof(double));
    auto *local_C = (double*)malloc(countsA[rank] / left_columns_right_rows * right_columns * sizeof(double));

    MPI_Scatterv(A, countsA, displsA, MPI_DOUBLE, local_A, countsA[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, countsB, displsB, MPI_DOUBLE, local_B, countsB[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int rand_data, start_column, column_count;
    for (int iter = 0; iter < size; iter++) {
        rand_data = (rank + size - iter) % size;
        start_column = displsB[rand_data] / left_columns_right_rows;
        column_count = countsB[rand_data] / left_columns_right_rows;

        for (i = 0; i < countsA[rank] / left_columns_right_rows; ++i) {
            for (j = 0; j < column_count; ++j) {
                local_C[i * right_columns + (start_column + j)] = 0;
                for (k = 0; k < left_columns_right_rows; ++k) {
                    local_C[i * right_columns + (start_column + j)] += (local_A[i * left_columns_right_rows + k] * local_B[j * left_columns_right_rows + k]);
                }
            }
        }
        MPI_Sendrecv_replace(local_B, max_rows_per_proc, MPI_DOUBLE, (rank + 1) % size, 0, (rank + size - 1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Gatherv(local_C, countsA[rank] / left_columns_right_rows * right_columns, MPI_DOUBLE, C, countsC, displsC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    if (isMasterProcess()) {
        std::cout <<"Time = " << end_time - start_time << std::endl;
        free(countsC);
        free(displsC);
    }
    free(local_A);
    free(local_B);
    free(local_C);
    free(countsA);
    free(displsA);
    free(countsB);
    free(displsB);
}


#endif //LAB2_MATRIX_H

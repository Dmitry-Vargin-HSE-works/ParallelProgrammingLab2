#ifndef LAB2_UTILS_H
#define LAB2_UTILS_H

#include <iostream>
#include <mpi.h>

#define MASTER_PROCESS 0

enum SplitType {
    Rows = 0,
    Columns = 1,
};

int getProcessNumber(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int getProcessCount(MPI_Comm comm = MPI_COMM_WORLD) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

bool isMasterProcess(MPI_Comm comm = MPI_COMM_WORLD) {
    return getProcessNumber(comm) == MASTER_PROCESS;
}

#endif //LAB2_UTILS_H

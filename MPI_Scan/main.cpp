#include <iostream>
#include "mpi.h"
#include <cstdlib>  // Для функции rand() и srand()
#include <fstream>  // Для работы с файлами

# define MIN_NUM 1  // Нижний порог для генерации числа в узле
# define MAX_NUM 1000  // Верхний порог для генерации числа в узле
# define MATRIX_SIZE 4  // Размер матрицы 4x4
# define TAG 0  


// Генерация случайного числа
int get_number(int rank) {
    std::srand(rank);
    return std::rand() % MAX_NUM + MIN_NUM;
    // return rank
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPI_Status status;
    MPI_Request req[2];
    MPI_Status st[2];

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != MATRIX_SIZE * MATRIX_SIZE) {
        MPI_Finalize();
        std::cerr << "Error: incorrect number of processes" << std::endl;
        return 1;
    }

    int cur_number = get_number(rank);  // число в узле данного процесса
    int sum_final;  // конечное число в узле

    // пересылка по строке

    int sum_left = cur_number, sum_right, sum;

    if (rank % MATRIX_SIZE == 0) {
        MPI_Sendrecv(&cur_number, 1, MPI_INT, rank + 1, TAG, &sum, 1, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD, &status);
        sum += cur_number;
    }
    else if (rank % MATRIX_SIZE == 3) {
        MPI_Sendrecv(&cur_number, 1, MPI_INT, rank - 1, TAG, &sum_left, 1, MPI_INT, rank - 1, TAG, MPI_COMM_WORLD, &status);
        sum_left += cur_number;
        sum = sum_left;
    }
    else if (rank % MATRIX_SIZE == 1) {
        MPI_Recv(&sum_left, 1, MPI_INT, rank - 1, TAG, MPI_COMM_WORLD, &status);

        sum_left += cur_number;
        MPI_Isend(&sum_left, 1, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&sum_right, 1, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, st);
        
        sum_right += cur_number;
        MPI_Send(&sum_right, 1, MPI_INT, rank - 1, TAG, MPI_COMM_WORLD);
        sum = sum_left + sum_right - cur_number;
    }
    else {
        MPI_Recv(&sum_right, 1, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD, &status);

        sum_right += cur_number;
        MPI_Isend(&sum_right, 1, MPI_INT, rank - 1, TAG, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&sum_left, 1, MPI_INT, rank - 1, TAG, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, st);

        sum_left += cur_number;
        MPI_Send(&sum_left, 1, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD);
        sum = sum_left + sum_right - cur_number;
    }

    sum_final = sum_left;

    // пересылка по столбцу

    if (rank / MATRIX_SIZE != 0) {
        MPI_Recv(&sum_left, 1, MPI_INT, rank - MATRIX_SIZE, TAG, MPI_COMM_WORLD, &status);
        sum += sum_left;
        sum_final += sum_left;
    }

    if (rank / MATRIX_SIZE != 3) {
        MPI_Send(&sum, 1, MPI_INT, rank + MATRIX_SIZE, TAG, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //std::cout << rank << ' ' << sum_final;
    MPI_Finalize();
    return 0;
}

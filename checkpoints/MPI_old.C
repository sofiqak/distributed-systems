#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N 1500
#define TAG_PASS_FIRST 0xA
#define TAG_PASS_LAST 0xB

double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;

double eps, s=0.;
double A [N][N];

void relax();
void init();
void verify();

void pass_first_row();
void pass_last_row();
void wait_all();

int size, rank, fst_r, lst_r, cnt_r;

MPI_Request req_buf[4];
MPI_Status stat_buf[4];

int main(int an, char **as)
{
        int it;

        MPI_Init(&an, &as);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        fst_r = (N - 2) / size * rank + 1;
    lst_r = (N - 2) / size * (rank + 1) + 1;
    cnt_r = lst_r - fst_r;

    double time_start, time_end;
    time_start = MPI_Wtime();

        init();

        for(it=1; it<=itmax; it++)
        {
                eps = 0.;
                relax();
                if (eps < maxeps) break;
        }

    MPI_Gather(A[fst_r], cnt_r * N, MPI_DOUBLE, A[1], cnt_r * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      verify();

    time_end = MPI_Wtime();

    if (!rank)
        printf("Total time: %lf.\n", time_end - time_start);

    MPI_Finalize();

        return 0;
}


void init()
{
        for (i = fst_r; i < lst_r; i++)
        for(j=0; j<=N-1; j++)
        {
                if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
                else A[i][j]= ( 1. + i + j ) ;
        }
}

void relax()
{
    pass_last_row();
    pass_first_row();
    wait_all();

        for (i = fst_r; i < lst_r; i++)
        for(j=1; j<=N-2; j++)
        {
                A[i][j] = (A[i-1][j]+A[i+1][j])/2.;
        }

        double eps_local = 0.;

    pass_last_row();
    pass_first_row();
    wait_all();

    for (i = fst_r; i < lst_r; i++)
        for(j=1; j<=N-2; j++)
        {
                double e;
                e=A[i][j];
                A[i][j] =(A[i][j-1]+A[i][j+1])/2.;
                eps_local=Max(eps_local,fabs(e-A[i][j]));
        }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&eps_local, &eps, 1 , MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void verify()
{
        double s_local = 0.;

    for (i = fst_r; i < lst_r; i++)
        for(j=0; j<=N-1; j++)
        {
                s_local=s_local+A[i][j]*(i+1)*(j+1)/(N*N);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&s_local, &s, 1 , MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("  S = %f\n", s);
    }
}

void pass_last_row() {
    if (rank)
        MPI_Irecv(A[fst_r - 1], N, MPI_DOUBLE, rank - 1, TAG_PASS_LAST, MPI_COMM_WORLD, req_buf);
    if (rank != size - 1)
        MPI_Isend(A[lst_r - 1], N, MPI_DOUBLE, rank + 1, TAG_PASS_LAST, MPI_COMM_WORLD, req_buf + 2);
}

void pass_first_row() {
    if (rank != size - 1)
        MPI_Irecv(A[lst_r], N, MPI_DOUBLE, rank + 1, TAG_PASS_FIRST, MPI_COMM_WORLD, req_buf + 3);
    if (rank)
        MPI_Isend(A[fst_r], N, MPI_DOUBLE, rank - 1, TAG_PASS_FIRST, MPI_COMM_WORLD, req_buf + 1);
}

void wait_all() {
    int count = 4, shift = 0;
    if (!rank) {
        count -= 2;
        shift = 2;
    }
    if (rank == size - 1) {
        count -= 2;
    }

    MPI_Waitall(count, req_buf + shift, stat_buf);
}

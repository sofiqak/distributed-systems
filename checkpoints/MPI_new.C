#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <signal.h>
#include <mpi-ext.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N 1500
#define TAG_PASS_FIRST 0xA
#define TAG_PASS_LAST 0xB
#define SAVE_FILE "save_data.txt"

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;

double eps, s = 0.;
double (*A)[N];

void relax();
void init();
void verify();

void pass_first_row();
void pass_last_row();
void wait_all();

int it, size, rank, fst_r, lst_r, cnt_r;

MPI_Request req_buf[4];
MPI_Status stat_buf[4];

int kill_rank = 1, kill_iteration = 10;
MPI_Comm main_comm;
MPI_Errhandler errh;

void save_checkpoint();
void load_checkpoint();
void error_handler(MPI_Comm *pcomm, int *error_code, ...);

int main(int an, char **as)
{
    if (an >= 2) {
        kill_rank = strtol(as[1], NULL, 10);  // rank of killed process
    }
    if (an >= 3) {
        kill_iteration = strtol(as[2], NULL, 10);  // number iteration, when process kill_rank will be killed
    }

    MPI_Init(&an, &as);

    main_comm = MPI_COMM_WORLD;
    MPI_Comm_size(main_comm, &size);
    MPI_Comm_rank(main_comm, &rank);
    MPI_Comm_create_errhandler(error_handler, &errh);
    MPI_Comm_set_errhandler(main_comm, errh);

    fst_r = (N - 2) / size * rank + 1;
    lst_r = (N - 2) / size * (rank + 1) + 1;
    cnt_r = lst_r - fst_r;
    A = (double (*)[N]) malloc((cnt_r + 2) * sizeof(double[N]));

    double time_start, time_end;
    time_start = MPI_Wtime();

    init();

    for(it = 1; it <= itmax; it ++)
    {
	MPI_Barrier(main_comm);
	if (it != kill_iteration) {
	    save_checkpoint();
	}
	eps = 0.;
        relax();
        if (eps < maxeps) {
            break;
        }
        if (rank == kill_rank && it == kill_iteration) {
	    printf("Rank %d / %d: I died!\n", rank, size);
	    raise(SIGKILL);
        }
    }

    verify();

    time_end = MPI_Wtime();

    if (!rank) {
        printf("Total time: %lf.\n", time_end - time_start);
    }

    free(A);
    MPI_Finalize();

    return 0;
}


void init()
{
    for (i = 1; i <= cnt_r; i ++)
    for (j = 0; j <= N - 1; j ++)
    {
        if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
            A[i][j] = 0.;
        } else {
            A[i][j] = (fst_r + i + j);
        }
    }
}

void relax()
{
    pass_last_row();
    pass_first_row();
    wait_all();

    for (i = 1; i <= cnt_r; i ++)
    for (j = 1; j <= N - 2; j ++)
    {
        A[i][j] = (A[i - 1][j] + A[i + 1][j]) / 2.;
    }

    double eps_local = 0.;

    pass_last_row();
    pass_first_row();
    wait_all();

    for (i = 1; i <= cnt_r; i ++)
    for (j = 1; j <= N - 2; j ++)
    {
        double e;
        e = A[i][j];
        A[i][j] = (A[i][j - 1] + A[i][j + 1]) / 2.;
        eps_local = Max(eps_local, fabs(e - A[i][j]));
    }

    MPI_Barrier(main_comm);
    MPI_Reduce(&eps_local, &eps, 1 , MPI_DOUBLE, MPI_MAX, 0, main_comm);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, main_comm);
}

void verify()
{
    double s_local = 0.;

    for (i = 1; i <= cnt_r; i ++)
    for (j = 0; j <= N - 1; j ++)
    {
        s_local = s_local + A[i][j] * (fst_r + i)*(j + 1) / (N * N);
    }

    MPI_Barrier(main_comm);
    MPI_Reduce(&s_local, &s, 1 , MPI_DOUBLE, MPI_SUM, 0, main_comm);
    if (!rank) {
        printf("  S = %f\n", s);
    }
}

void pass_last_row() {
    if (rank) {
        MPI_Irecv(A[0], N, MPI_DOUBLE, rank - 1, TAG_PASS_LAST, main_comm, req_buf);
    }
    if (rank != size - 1) {
        MPI_Isend(A[cnt_r], N, MPI_DOUBLE, rank + 1, TAG_PASS_LAST, main_comm, req_buf + 2);
    }
}

void pass_first_row() {
    if (rank != size - 1) {
        MPI_Irecv(A[cnt_r + 1], N, MPI_DOUBLE, rank + 1, TAG_PASS_FIRST, main_comm, req_buf + 3);
    }
    if (rank) {
        MPI_Isend(A[1], N, MPI_DOUBLE, rank - 1, TAG_PASS_FIRST, main_comm, req_buf + 1);
    }
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

void save_checkpoint() {
    MPI_File fh;
    MPI_Offset offset;
    MPI_File_open(main_comm, SAVE_FILE, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (rank != size) {
       offset = fst_r * N * sizeof(double);
       MPI_File_write_at(fh, offset, &A[1], cnt_r, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);
}

void error_handler(MPI_Comm* pcomm, int* error_code, ...) {
    int len;
    char error_string[MPI_MAX_ERROR_STRING];
    MPI_Error_string(*error_code, error_string, &len);
    printf("Rank %d / %d: Notified of error %s\n", rank, size, error_string);

    MPIX_Comm_shrink(*pcomm, &main_comm);
    MPI_Comm_set_errhandler(main_comm, errh);

    MPI_Comm_rank(main_comm, &rank);
    MPI_Comm_size(main_comm, &size);

    fst_r = (N - 2) / size * rank + 1;
    lst_r = (N - 2) / size * (rank + 1) + 1;
    cnt_r = lst_r - fst_r;

    load_checkpoint();
}

void load_checkpoint() {
    free(A);
    A = (double (*)[N]) malloc((cnt_r + 2) * sizeof(double[N]));
    MPI_File fh;
    MPI_Offset offset;
    MPI_File_open(main_comm, SAVE_FILE, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    offset = fst_r * N * sizeof(double);
    MPI_File_read_at(fh, offset, &A[1], cnt_r * N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

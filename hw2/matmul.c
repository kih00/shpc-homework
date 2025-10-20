#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct thread_arg {
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int num_threads;
  int rank; /* id of this thread */
} args[256];
static pthread_t threads[256];

static void *matmul_kernel(void *arg) {
  struct thread_arg *input = (struct thread_arg *)arg;
  const float *A = (*input).A;
  const float *B = (*input).B;
  float *C = (*input).C;
  int M = (*input).M;
  int N = (*input).N;
  int K = (*input).K;
  int num_threads = (*input).num_threads;
  int rank = (*input).rank;

  /*
  TODO: FILL IN HERE
  (M, K) * (K, N) => (M, N)
  */
  int rows_per_thread = M / num_threads;
  int rem = M % num_threads;
  const int BLOCK_SIZE = 128;

  int row_start, row_end;
  if (rank < rem) {
    row_start = (rows_per_thread + 1) * rank;
    row_end = row_start + rows_per_thread + 1;
  } else {
    row_start = rows_per_thread * rank + rem;
    row_end = row_start + rows_per_thread;
  }

  for (int its = row_start; its < row_end; its += BLOCK_SIZE) {
    for (int kts = 0; kts < K; kts += BLOCK_SIZE) {
      for (int jts = 0; jts < N; jts += BLOCK_SIZE) {
        int ite = (its + BLOCK_SIZE < row_end) ? its + BLOCK_SIZE : row_end;
        int kte = (kts + BLOCK_SIZE < K) ? kts + BLOCK_SIZE : K;
        int jte = (jts + BLOCK_SIZE < N) ? jts + BLOCK_SIZE : N;

        for (int i = its; i < ite; ++i) {
          for (int k = kts; k < kte; ++k) {
            float aik = A[i * K + k];
            for (int j = jts; j < jte; ++j) {
              C[i * N + j] += aik * B[k * N + j];
            }
          }
        }
      }
    }
  }

  // for (int i = row_start; i < row_end; ++i) {
  //   for (int k = 0; k < K; ++k) {
  //     float aik = A[i * K + k];
  //     for (int j = 0; j < N; ++j) {
  //       C[i * N + j] += aik * B[k * N + j];
  //     }
  //   }
  // }

  return NULL;
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads) {

  if (num_threads > 256) {
    fprintf(stderr, "num_threads must be <= 256\n");
    exit(EXIT_FAILURE);
  }

  int err;
  for (int t = 0; t < num_threads; ++t) {
    args[t].A = A, args[t].B = B, args[t].C = C, args[t].M = M, args[t].N = N,
    args[t].K = K, args[t].num_threads = num_threads, args[t].rank = t;
    err = pthread_create(&threads[t], NULL, matmul_kernel, (void *)&args[t]);
    if (err) {
      printf("pthread_create(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }

  for (int t = 0; t < num_threads; ++t) {
    err = pthread_join(threads[t], NULL);
    if (err) {
      printf("pthread_join(%d) failed with err %d\n", t, err);
      exit(EXIT_FAILURE);
    }
  }
}

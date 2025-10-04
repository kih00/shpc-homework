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
  */

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

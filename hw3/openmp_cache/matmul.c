#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(float *A, float *B, float *C, int M, int N, int K,
            int num_threads, int block_size) {

  const int BLOCK_SIZE = block_size;

  #pragma omp parallel for num_threads(num_threads) schedule(guided)
  for (int its = 0; its < M; its += BLOCK_SIZE) {
    for (int kts = 0; kts < K; kts += BLOCK_SIZE) {
      for (int jts = 0; jts < N; jts += BLOCK_SIZE) {
        int ite = (its + BLOCK_SIZE < M) ? its + BLOCK_SIZE : M;
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
}

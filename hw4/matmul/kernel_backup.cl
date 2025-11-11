__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  // TODO: FILL_IN_HERE
  #define TS 32 // must be as same as TS in matmul.c
  
  // calculate global index(i, j) and local index(ii, jj)
  int i = get_global_id(1);
  int j = get_global_id(0);
  int ii = get_local_id(1);
  int jj = get_local_id(0);

  // __local memory
  __local float A_local[TS][TS];
  __local float B_local[TS][TS];
  // variable for saving C[i][j]
  float sum = 0.0f;

  // loop by tile (tile size: TS)
  for (int t = 0; t < K; t += TS) {
      
    // A_local[ii][jj] = A[i][t + jj]
    int ii_start_A = i;
    int jj_start_A = t + jj;
    if (ii_start_A < M && jj_start_A < K) {
      A_local[ii][jj] = A[ii_start_A * K + jj_start_A];
    } else {
      A_local[ii][jj] = 0.0f;
    }

    // B_local[ii][jj] = B[t + ii][j]
    int ii_start_B = t + ii;
    int jj_start_B = j;
    if (ii_start_B < K && jj_start_B < N) {
      B_local[ii][jj] = B[ii_start_B * N + jj_start_B];
    } else {
      B_local[ii][jj] = 0.0f;
    }

    // wait until all local memory complete read from A, B
    barrier(CLK_LOCAL_MEM_FENCE);

    // calculate matrix multiplication
    for (int k = 0; k < TS; ++k) {
      sum += A_local[ii][k] * B_local[k][jj];
    }

    // watil unil all local memory complete write to sum
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // C[i][j] = sum
  if (i < M && j < N) {
    C[i * N + j] = sum;
  }
}

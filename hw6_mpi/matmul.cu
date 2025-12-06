#include "matmul.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at [%s:%d] %d %s\n", __FILE__, __LINE__,     \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#define TILE 16
#define CHUNK 8
#define NUM_STAGE 2
#define MAX_NUM_GPU 4
#define MAX_MPI_NODES 4
#define ROW_CHUNK_TARGET (TILE * CHUNK * 16)
#define TAG_STRIDE 4096
#define TAG_A_BASE 1000
#define TAG_C_BASE 5000

int num_devices = 0;
static int mpi_rank = 0;
static int mpi_world_size = 1;
static bool use_multi_node = true;
static int rank_row_count = 0;
static int row_counts[MAX_MPI_NODES] = {0};
static int row_offsets[MAX_MPI_NODES] = {0};
static int chunk_caps[MAX_MPI_NODES] = {0};
static float *stage_A[NUM_STAGE] = {nullptr};
static float *stage_C[NUM_STAGE] = {nullptr};
static int stage_rows[NUM_STAGE] = {0};
static int local_chunk_cap = 0;
static bool stage_allocated = false;
static bool b_synced = false;
static size_t last_b_elems = 0;
static bool b_uploaded[MAX_NUM_GPU] = {false};
static int current_B_dims[2] = {0, 0};

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N,
                              int K) {
  __shared__ float A_tile[NUM_STAGE][TILE*CHUNK][CHUNK];
  __shared__ float B_tile[NUM_STAGE][CHUNK][TILE*CHUNK];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int block_row = blockIdx.y * TILE * CHUNK;
  int block_col = blockIdx.x * TILE * CHUNK;

  int row = blockIdx.y * TILE * CHUNK + ty * CHUNK;
  int col = blockIdx.x * TILE * CHUNK + tx * CHUNK;

  float C_val[CHUNK][CHUNK];
#pragma unroll
  for (int i = 0; i < CHUNK; ++i) {
#pragma unroll
    for (int j = 0; j < CHUNK; ++j) {
      C_val[i][j] = 0.0f;
    }
  }

  int num_chunks = (K + CHUNK - 1) / CHUNK;
  if (num_chunks == 0) return;

  int tid = ty * blockDim.x + tx;
  int total_threads = blockDim.x * blockDim.y;

  int curr = 0;
  int next = 1;
  for (int i=tid; i<TILE*CHUNK*CHUNK; i+=total_threads) {
    int local_row_A = i / CHUNK;
    int local_col_A = i % CHUNK;
    int local_row_B = i / (TILE * CHUNK);
    int local_col_B = i % (TILE * CHUNK);

    int global_row_A = block_row + local_row_A;
    int global_col_A = local_col_A;
    int global_row_B = local_row_B;
    int global_col_B = block_col + local_col_B;

    float val_A = 0.0f, val_B = 0.0f;
    if (global_row_A < M && global_col_A < K)
      val_A = A[global_row_A * K + global_col_A];

    if (global_row_B < K && global_col_B < N)
      val_B = B[global_row_B * N + global_col_B];

    A_tile[curr][local_row_A][local_col_A] = val_A;
    B_tile[curr][local_row_B][local_col_B] = val_B;
  }

  __syncthreads();

  // double buffering
  for (int t = 0; t < num_chunks; ++t) {
    // load next tile
    if (t + 1 < num_chunks) {
      int next_k = (t + 1) * CHUNK;

      for (int i=tid; i<TILE*CHUNK*CHUNK; i+=total_threads) {
        int local_row_A = i / CHUNK;
        int local_col_A = i % CHUNK;
        int local_row_B = i / (TILE * CHUNK);
        int local_col_B = i % (TILE * CHUNK);

        int global_row_A = block_row + local_row_A;
        int global_col_A = next_k + local_col_A;
        int global_row_B = next_k + local_row_B;
        int global_col_B = block_col + local_col_B;

        float val_A = 0.0f, val_B = 0.0f;
        if (global_row_A < M && global_col_A < K)
          val_A = A[global_row_A * K + global_col_A];

        if (global_row_B < K && global_col_B < N)
          val_B = B[global_row_B * N + global_col_B];

        A_tile[next][local_row_A][local_col_A] = val_A;
        B_tile[next][local_row_B][local_col_B] = val_B;
      }
    }

    float A_reg[CHUNK];
#pragma unroll
    for (int k = 0; k < CHUNK; ++k) {
#pragma unroll
      for (int i = 0; i < CHUNK; ++i) {
        A_reg[i] = A_tile[curr][ty * CHUNK + i][k];
      }
#pragma unroll
      for (int j = 0; j < CHUNK; ++j) {
        float B_val = B_tile[curr][k][tx * CHUNK + j];
#pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
          C_val[i][j] += A_reg[i] * B_val;
        }
      }
    }

    __syncthreads();
    curr ^= 1;
    next ^= 1;
  }

  for (int i = 0; i < CHUNK; ++i) {
    int global_row = row + i;
    if (global_row >= M) continue;
    for (int j = 0; j < CHUNK; ++j) {
      int global_col = col + j;
      if (global_col < N) {
        C[global_row * N + global_col] = C_val[i][j];
      }
    }
  }
}

// Array of device (GPU) pointers
static float *a_d[MAX_NUM_GPU];
static float *b_d[MAX_NUM_GPU];
static float *c_d[MAX_NUM_GPU];
static int Mbegin[MAX_NUM_GPU], Mend[MAX_NUM_GPU];
static cudaStream_t streams[MAX_NUM_GPU];

static inline int div_up(int x, int y) { return (x + y - 1) / y; }

static inline int rows_for_chunk(int chunk_idx, int chunk_rows, int total_rows) {
  int start = chunk_idx * chunk_rows;
  int remain = total_rows - start;
  if (remain <= 0)
    return 0;
  return remain < chunk_rows ? remain : chunk_rows;
}

static inline int make_tag(int base, int rank, int chunk) {
  return base + rank * TAG_STRIDE + chunk;
}

static void reset_device_b_state() {
  for (int i = 0; i < MAX_NUM_GPU; ++i) {
    b_uploaded[i] = false;
  }
}

static void compute_chunk(const float *chunk_A, float *chunk_C, int rows,
                          int N, int K, const float *B) {
  if (rows <= 0 || num_devices <= 0)
    return;

  int active_devices = num_devices;
  if (rows < active_devices)
    active_devices = rows;

  int rows_base = rows / active_devices;
  int rows_extra = rows % active_devices;
  int offset = 0;

  for (int i = 0; i < active_devices; ++i) {
    int rows_i = rows_base + (i < rows_extra ? 1 : 0);
    if (rows_i == 0)
      continue;

    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMemcpyAsync(a_d[i], chunk_A + offset * K,
                              rows_i * K * sizeof(float),
                              cudaMemcpyHostToDevice, streams[i]));

    if (!b_uploaded[i] || current_B_dims[0] != K || current_B_dims[1] != N) {
      CUDA_CALL(cudaMemcpyAsync(b_d[i], B, K * N * sizeof(float),
                                cudaMemcpyHostToDevice, streams[i]));
      b_uploaded[i] = true;
    }

    dim3 blockDim(TILE, TILE, 1);
    dim3 gridDim((N + (TILE * CHUNK) - 1) / (TILE * CHUNK),
                 (rows_i + TILE * CHUNK - 1) / (TILE * CHUNK), 1);

    matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(a_d[i], b_d[i], c_d[i],
                                                        rows_i, N, K);

    CUDA_CALL(cudaMemcpyAsync(chunk_C + offset * N, c_d[i],
                              rows_i * N * sizeof(float),
                              cudaMemcpyDeviceToHost, streams[i]));

    offset += rows_i;
  }

  for (int i = 0; i < active_devices; ++i) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaStreamSynchronize(streams[i]));
  }
}

static void process_local_rows(const float *A, float *C, int rows, int N, int K,
                               const float *B, int chunk_cap) {
  if (rows <= 0)
    return;

  int chunk_rows = chunk_cap > 0 ? chunk_cap : rows;
  int num_chunks = div_up(rows, chunk_rows);
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int rows_this = rows_for_chunk(chunk, chunk_rows, rows);
    int row_start = chunk * chunk_rows;
    compute_chunk(A + row_start * K, C + row_start * N, rows_this, N, K, B);
  }
}

static void run_root_pipeline(const float *A, const float *B, float *C, int N,
                              int K) {
  // Post asynchronous sends/receives for all worker ranks.
  MPI_Request send_reqs[MAX_MPI_NODES][NUM_STAGE];
  MPI_Request recv_reqs[MAX_MPI_NODES][NUM_STAGE];
  bool send_active[MAX_MPI_NODES][NUM_STAGE] = {{false}};
  bool recv_active[MAX_MPI_NODES][NUM_STAGE] = {{false}};

  for (int rank = 1; rank < mpi_world_size; ++rank) {
    int rows = row_counts[rank];
    if (rows == 0)
      continue;
    int chunk_cap = chunk_caps[rank];
    if (chunk_cap == 0)
      chunk_cap = rows;
    int num_chunks = div_up(rows, chunk_cap);
    int row_offset = row_offsets[rank];
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      int stage = chunk % NUM_STAGE;
      if (send_active[rank][stage]) {
        MPI_Wait(&send_reqs[rank][stage], MPI_STATUS_IGNORE);
        send_active[rank][stage] = false;
      }
      if (recv_active[rank][stage]) {
        MPI_Wait(&recv_reqs[rank][stage], MPI_STATUS_IGNORE);
        recv_active[rank][stage] = false;
      }

      int rows_this = rows_for_chunk(chunk, chunk_cap, rows);
      int row_start = row_offset + chunk * chunk_cap;

      MPI_Isend(const_cast<float *>(A) + row_start * K, rows_this * K,
                MPI_FLOAT, rank, make_tag(TAG_A_BASE, rank, chunk),
                MPI_COMM_WORLD, &send_reqs[rank][stage]);
      send_active[rank][stage] = true;

      MPI_Irecv(C + row_start * N, rows_this * N, MPI_FLOAT, rank,
                make_tag(TAG_C_BASE, rank, chunk), MPI_COMM_WORLD,
                &recv_reqs[rank][stage]);
      recv_active[rank][stage] = true;
    }
  }

  // Process local rows on rank 0 while workers stream results back.
  int local_rows = row_counts[0];
  int local_offset = row_offsets[0];
  process_local_rows(A + local_offset * K, C + local_offset * N, local_rows,
                     N, K, B, chunk_caps[0]);

  // Ensure all outstanding sends and receives complete.
  for (int rank = 1; rank < mpi_world_size; ++rank) {
    for (int stage = 0; stage < NUM_STAGE; ++stage) {
      if (send_active[rank][stage]) {
        MPI_Wait(&send_reqs[rank][stage], MPI_STATUS_IGNORE);
        send_active[rank][stage] = false;
      }
      if (recv_active[rank][stage]) {
        MPI_Wait(&recv_reqs[rank][stage], MPI_STATUS_IGNORE);
        recv_active[rank][stage] = false;
      }
    }
  }
}

static void run_worker_pipeline(const float *B, int N, int K) {
  if (rank_row_count == 0)
    return;

  int chunk_cap = local_chunk_cap > 0 ? local_chunk_cap : rank_row_count;
  int num_chunks = div_up(rank_row_count, chunk_cap);

  MPI_Request recv_reqs[NUM_STAGE];
  MPI_Request send_reqs[NUM_STAGE];
  bool recv_active[NUM_STAGE] = {false};
  bool send_active[NUM_STAGE] = {false};

  auto post_recv = [&](int chunk_id) {
    if (chunk_id >= num_chunks)
      return;
    int stage = chunk_id % NUM_STAGE;
    int rows_this = rows_for_chunk(chunk_id, chunk_cap, rank_row_count);
    stage_rows[stage] = rows_this;
    MPI_Irecv(stage_A[stage], rows_this * K, MPI_FLOAT, 0,
              make_tag(TAG_A_BASE, mpi_rank, chunk_id), MPI_COMM_WORLD,
              &recv_reqs[stage]);
    recv_active[stage] = true;
  };

  int preload = num_chunks < NUM_STAGE ? num_chunks : NUM_STAGE;
  for (int i = 0; i < preload; ++i) {
    post_recv(i);
  }

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int stage = chunk % NUM_STAGE;

    if (send_active[stage]) {
      MPI_Wait(&send_reqs[stage], MPI_STATUS_IGNORE);
      send_active[stage] = false;
    }

    if (recv_active[stage]) {
      MPI_Wait(&recv_reqs[stage], MPI_STATUS_IGNORE);
      recv_active[stage] = false;
    }

    int rows_this = stage_rows[stage];
    compute_chunk(stage_A[stage], stage_C[stage], rows_this, N, K, B);

    MPI_Isend(stage_C[stage], rows_this * N, MPI_FLOAT, 0,
              make_tag(TAG_C_BASE, mpi_rank, chunk), MPI_COMM_WORLD,
              &send_reqs[stage]);
    send_active[stage] = true;

    post_recv(chunk + NUM_STAGE);
  }

  for (int stage = 0; stage < NUM_STAGE; ++stage) {
    if (send_active[stage]) {
      MPI_Wait(&send_reqs[stage], MPI_STATUS_IGNORE);
      send_active[stage] = false;
    }
    if (recv_active[stage]) {
      MPI_Wait(&recv_reqs[stage], MPI_STATUS_IGNORE);
      recv_active[stage] = false;
    }
  }
}

void matmul(const float *A, const float *B, float *C, int M, int N, int K) {

  if (!use_multi_node) {
    reset_device_b_state();
    current_B_dims[0] = K;
    current_B_dims[1] = N;
    process_local_rows(A, C, M, N, K, B, M);
    return;
  }

  size_t elems_B = (size_t)K * N;
  if (last_b_elems != elems_B) {
    b_synced = false;
    last_b_elems = elems_B;
  }

  if (!b_synced) {
    float *mutable_B = const_cast<float *>(B);
    MPI_Bcast(mutable_B, elems_B, MPI_FLOAT, 0, MPI_COMM_WORLD);
    b_synced = true;
    reset_device_b_state();
    current_B_dims[0] = K;
    current_B_dims[1] = N;
  }

  if (mpi_rank == 0) {
    run_root_pipeline(A, B, C, N, K);
  } else {
    run_worker_pipeline(B, N, K);
  }
}

void matmul_initialize(int M, int N, int K) {

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  if (mpi_world_size > MAX_MPI_NODES) {
    if (mpi_rank == 0) {
      printf("This build supports up to %d MPI ranks.\n", MAX_MPI_NODES);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  use_multi_node = mpi_world_size > 1;

  int rows_per_rank = M / mpi_world_size;
  int row_remainder = M % mpi_world_size;
  int prefix = 0;
  for (int r = 0; r < mpi_world_size; ++r) {
    int rows = rows_per_rank + (r < row_remainder ? 1 : 0);
    row_counts[r] = rows;
    row_offsets[r] = prefix;
    chunk_caps[r] = rows == 0
                        ? 0
                        : (rows < ROW_CHUNK_TARGET ? rows : ROW_CHUNK_TARGET);
    if (r == mpi_rank) {
      rank_row_count = rows;
      local_chunk_cap = chunk_caps[r];
    }
    prefix += rows;
  }

  CUDA_CALL(cudaGetDeviceCount(&num_devices));
  if (num_devices > MAX_NUM_GPU) {
    num_devices = MAX_NUM_GPU;
  }
  reset_device_b_state();

  if (mpi_rank == 0) {
    printf("Total MPI ranks: %d\n", mpi_world_size);
  }
  printf("[rank %d] Using %d devices\n", mpi_rank, num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, i));

    // Try printing more detailed information here
    printf("[rank %d] GPU %d: %s\n", mpi_rank, i, prop.name);
  }

  if (num_devices <= 0) {
    printf("No CUDA device found. Aborting\n");
    exit(1);
  }

  if (rank_row_count == 0) {
    num_devices = 0;
    return;
  }

  if (rank_row_count < num_devices) {
    num_devices = rank_row_count;
  }
  if (use_multi_node && rank_row_count > 0 && mpi_rank != 0) {
    int cap = local_chunk_cap > 0 ? local_chunk_cap : rank_row_count;
    size_t a_bytes = (size_t)cap * K * sizeof(float);
    size_t c_bytes = (size_t)cap * N * sizeof(float);
    for (int stage = 0; stage < NUM_STAGE; ++stage) {
      CUDA_CALL(cudaMallocHost((void **)&stage_A[stage], a_bytes));
      CUDA_CALL(cudaMallocHost((void **)&stage_C[stage], c_bytes));
    }
    stage_allocated = true;
  }

  // Setup problem size for each GPU
  int rows_base = rank_row_count / num_devices;
  int rows_extra = rank_row_count % num_devices;
  int offset = 0;
  for (int i = 0; i < num_devices; i++) {
    int rows = rows_base + (i < rows_extra ? 1 : 0);
    Mbegin[i] = offset;
    Mend[i] = offset + rows;
    offset += rows;
  }

  // Allocate device memory for each GPU
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaSetDevice(i));
    CUDA_CALL(cudaMalloc(&a_d[i], (Mend[i] - Mbegin[i]) * K * sizeof(float)));
    CUDA_CALL(cudaMalloc(&b_d[i], K * N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&c_d[i], (Mend[i] - Mbegin[i]) * N * sizeof(float)));
    CUDA_CALL(cudaStreamCreate(&streams[i]));
  }
}

void matmul_finalize() {
  if (stage_allocated) {
    for (int stage = 0; stage < NUM_STAGE; ++stage) {
      if (stage_A[stage]) {
        CUDA_CALL(cudaFreeHost(stage_A[stage]));
        stage_A[stage] = nullptr;
      }
      if (stage_C[stage]) {
        CUDA_CALL(cudaFreeHost(stage_C[stage]));
        stage_C[stage] = nullptr;
      }
    }
    stage_allocated = false;
  }

  // Free all GPU memory
  for (int i = 0; i < num_devices; i++) {
    CUDA_CALL(cudaFree(a_d[i]));
    CUDA_CALL(cudaFree(b_d[i]));
    CUDA_CALL(cudaFree(c_d[i]));
    CUDA_CALL(cudaStreamDestroy(streams[i]));
  }
}

#include <cuda_runtime.h>
#include <cuda_profiler_api.h> // 제출 시 지우기
#include <mpi.h>

#include "model.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <unistd.h>

static int num_samples = 1; // batch_size == num_samples
static bool run_validation = false;
static bool run_warmup = false;
static int num_warmup = 1;

double get_time() {
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);

    return tv.tv_sec + tv.tv_nsec * 1e-9;
}

// Helper function to read int32 from file
int32_t read_int32(std::ifstream& file) {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(int32_t));
    return value;
}

// Helper function to read float from file
float read_float(std::ifstream& file) {
    float value;
    file.read(reinterpret_cast<char*>(&value), sizeof(float));
    return value;
}

// Helper function to write int32 to file
void write_int32(std::ofstream& file, int32_t value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(int32_t));
}

// Helper function to write float to file
void write_float(std::ofstream& file, float value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(float));
}

void print_help() {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        fprintf(stdout,
                " Usage: ./main  [-n 'num_samples'] [-b 'batch_size'] [-v] [-w] [-h]\n");
        fprintf(stdout, " Options:\n");
        fprintf(stdout, "  -n: Number of input samples (default: 1)\n");
        fprintf(stdout, "  -v: Enable validation (default: OFF)\n");
        fprintf(stdout, "  -w: Enable warm-up (default: OFF)\n");
        fprintf(stdout, "  -h: Print manual and options (default: OFF)\n");
    }
}

void parse_args(int argc, char **argv) {
    int args;
    while ((args = getopt(argc, argv, "n:vwh")) != -1) {
    switch (args) {
        case 'n': num_samples = atoi(optarg); break;
        case 'v': run_validation = true; break;
        case 'w': run_warmup = true; break;
        case 'h':
            print_help();
            exit(0);
            break;
        default:
            print_help();
            exit(0);
            break;
        }
    }
  
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        fprintf(stdout, "\n=============================================\n");
        fprintf(stdout, " Model: LFM2-8B-A1B\n");
        fprintf(stdout, "---------------------------------------------\n");
        fprintf(stdout, " Validation: %s\n", run_validation ? "ON" : "OFF");
        fprintf(stdout, " Warm-up: %s\n", run_warmup ? "ON" : "OFF");
        fprintf(stdout, " Number of samples: %d\n", num_samples);
        fprintf(stdout, "=============================================\n\n");
    }
}

int main(int argc, char* argv[]) {
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Set global MPI rank for debug output
    g_mpi_rank = mpi_rank;

    parse_args(argc, argv);

    // Configuration
    std::string model_file = "/mnt/ramdisk/model.bin";
    std::string input_file = "data/inputs.bin";
    std::string output_file = "data/outputs.bin";

    ////////////////////////////////////////////////////////////////////
    // INITIALIZATION                                                 //
    ////////////////////////////////////////////////////////////////////

    cudaProfilerStart(); // 제출 시 지우기

    int *inputs = nullptr;
    float *outputs = nullptr;
    int32_t total_samples = 0;
    int32_t seq_length = 0;

    // Data Parallelism: distribute samples across MPI ranks
    int n_per_rank = num_samples / mpi_size;
    int remainder = num_samples % mpi_size;
    int local_n = n_per_rank + (mpi_rank < remainder ? 1 : 0);

    /* Only MPI process rank 0 has the inputs and outputs */
    if (mpi_rank == 0) fprintf(stdout, "Initializing inputs and outputs...");

    if (mpi_rank == 0) {
        // Read input file to get dimensions and data
        std::ifstream infile(input_file, std::ios::binary);
        if (!infile) {
            fprintf(stderr, "Failed to open input file: %s\n", input_file.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        total_samples = read_int32(infile);
        seq_length = read_int32(infile);

        fprintf(stdout, "Done!\n");
        fprintf(stdout, "Input file info:\n");
        fprintf(stdout, "  Total samples: %d\n", total_samples);
        fprintf(stdout, "  Sequence length: %d\n", seq_length);
        fprintf(stdout, "  Processing samples: %d\n", num_samples);
        fprintf(stdout, "  MPI ranks: %d\n", mpi_size);
        fprintf(stdout, "\n");

        // Allocate pinned memory for all inputs (rank 0)
        CHECK_CUDA(cudaMallocHost(&inputs, num_samples * seq_length * sizeof(int)));

        // Read all input samples into buffer
        for (int i = 0; i < num_samples; i++) {
            std::vector<int32_t> temp_input(seq_length);
            infile.read(reinterpret_cast<char*>(
                temp_input.data()), seq_length * sizeof(int32_t));

            if (!infile && i < num_samples - 1) {
                fprintf(stderr, "Warning: Could only read %d samples\n", i);
                break;
            }

            // Copy to pinned memory buffer
            for (int j = 0; j < seq_length; j++) {
                inputs[i * seq_length + j] = static_cast<int>(temp_input[j]);
            }
        }

        infile.close();

        // Allocate pinned memory for all outputs (rank 0)
        CHECK_CUDA(cudaMallocHost(
            &outputs, num_samples * VOCAB_SIZE * sizeof(float)));
    }

    // Broadcast seq_length to all ranks
    MPI_Bcast(&seq_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local input buffer for each rank
    int *local_inputs = nullptr;
    CHECK_CUDA(cudaMallocHost(
        &local_inputs, local_n * seq_length * sizeof(int)));

    // Scatter inputs to all ranks
    std::vector<int> sendcounts(mpi_size);
    std::vector<int> displs(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
        int r_samples = n_per_rank + (r < remainder ? 1 : 0);
        sendcounts[r] = r_samples * seq_length;
        displs[r] = (r * n_per_rank + std::min(r, remainder)) * seq_length;
    }

    MPI_Scatterv(
        inputs, sendcounts.data(), displs.data(), MPI_INT,
        local_inputs, local_n * seq_length, MPI_INT, 0, MPI_COMM_WORLD);

    // Load model (all ranks)
    if (mpi_rank == 0) fprintf(
        stdout, "Loading model from %s...", model_file.c_str());
    LFM2Model model(model_file);

    /* Warm-up (all ranks) */
    if (run_warmup && local_n > 0) {
        if (mpi_rank == 0) fprintf(stdout, "Warming up...");
        std::vector<int> warmup_input(
            local_inputs, local_inputs + local_n * seq_length);
        Tensor warmup_logits;
        for (int i = 0; i < num_warmup; i++) {
            model.forward(
                warmup_input, local_n, seq_length, warmup_logits);
        }
        if (mpi_rank == 0) fprintf(stdout, "Done!\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ////////////////////////////////////////////////////////////////////
    // MODEL COMPUTATION (Data Parallel)                              //
    ////////////////////////////////////////////////////////////////////

    double st = 0.0, et = 0.0;

    if (mpi_rank == 0) {
        fprintf(stdout, "Generating (Data Parallel with %d ranks)...", mpi_size);
        fflush(stdout);
    }

    for (size_t i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaSetDevice(0));
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) st = get_time();

    /* Call the main computation (all ranks) */
    float *local_outputs = nullptr;
    CHECK_CUDA(cudaMallocHost(
        &local_outputs, local_n * VOCAB_SIZE * sizeof(float)));

    if (local_n > 0) {
        // Get input for this rank's samples
        std::vector<int> input_ids_vec(
            local_inputs, local_inputs + local_n * seq_length);

        // Run forward pass
        Tensor logits;
        model.forward(input_ids_vec, local_n, seq_length, logits);

        logits.to_host();

        // Copy logits to local output buffer
        for (int b = 0; b < local_n; b++) {
            for (size_t i = 0; i < VOCAB_SIZE; i++) {
                local_outputs[b * VOCAB_SIZE + i] = logits.at(b, i);
            }
        }
    }

    for (size_t i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaSetDevice(0));
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather outputs to rank 0
    std::vector<int> recvcounts(mpi_size);
    std::vector<int> recvdispls(mpi_size);
    for (int r = 0; r < mpi_size; r++) {
        int r_samples = n_per_rank + (r < remainder ? 1 : 0);
        recvcounts[r] = r_samples * VOCAB_SIZE;
        recvdispls[r] = (r * n_per_rank + std::min(r, remainder)) * VOCAB_SIZE;
    }

    MPI_Gatherv(
        local_outputs, local_n * VOCAB_SIZE, MPI_FLOAT,
        outputs, recvcounts.data(), recvdispls.data(), MPI_FLOAT,
        0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        et = get_time();
        /* Print the result */
        fprintf(stdout, "Done!\n");
        fprintf(stdout, "Elapsed time: %lf (sec)\n", et - st);
        fprintf(stdout, "Throughput: %lf (samples/sec)\n\n",
                num_samples / (et - st));
    }

    // Free local buffers
    CHECK_CUDA(cudaFreeHost(local_inputs));
    CHECK_CUDA(cudaFreeHost(local_outputs));

    cudaProfilerStop(); // 제출 시 지우기

    ////////////////////////////////////////////////////////////////////
    // FINALIZATION                                                   //
    ////////////////////////////////////////////////////////////////////

    if (mpi_rank == 0) {
        /* Save outputs */
        fprintf(stdout, "Saving outputs to %s...", output_file.c_str());
        std::ofstream outfile(output_file, std::ios::binary);
        write_int32(outfile, num_samples);
        write_int32(outfile, VOCAB_SIZE);
        outfile.write(
            reinterpret_cast<const char*>(outputs),
            num_samples * VOCAB_SIZE * sizeof(float));
        outfile.close();
        fprintf(stdout, "Done!\n");

        if (run_validation) {
            std::string answer_file = "data/answers.bin";
            std::ifstream ansfile(answer_file, std::ios::binary);

            std::cout << "=" << std::string(58, '=') << std::endl;
            std::cout << "Validating against reference answers..." << std::endl;
            std::cout << "=" << std::string(58, '=') << std::endl;
            std::cout << std::endl;

            // Read answer file header
            int32_t ans_num_samples = read_int32(ansfile);
            int32_t ans_vocab_size = read_int32(ansfile);
        
            // Reopen outputs.bin to read for comparison
            std::ifstream outfile_read(output_file, std::ios::binary);
            int32_t out_num_samples = read_int32(outfile_read);
            read_int32(outfile_read);  // skip vocab_size

            int num_compare = std::min(
                num_samples, std::min(ans_num_samples, out_num_samples));
            std::cout << "Comparing " << num_compare << " samples..." << std::endl;
            std::cout << "Threshold: 1e-3" << std::endl;

            const float THRESHOLD = 1e-3f;
            int total_values = 0;
            int mismatches = 0;

            int top1_matches = 0;
            int first_mismatch_idx = -1;
            float first_mismatch_output = 0.0f;
            float first_mismatch_answer = 0.0f;

            for (int sample_idx = 0; sample_idx < num_compare; sample_idx++) {
                std::vector<float> output_logits(VOCAB_SIZE);
                std::vector<float> answer_logits(VOCAB_SIZE);

                // Read logits from both files
                for (size_t i = 0; i < VOCAB_SIZE; i++) {
                    output_logits[i] = read_float(outfile_read);
                }

                for (int32_t i = 0; i < ans_vocab_size; i++) {
                    if (i < static_cast<int32_t>(VOCAB_SIZE)) {
                        answer_logits[i] = read_float(ansfile);
                    } else {
                        read_float(ansfile); // Skip extra values
                    }
                }

                // Compare values
                for (size_t i = 0; i < VOCAB_SIZE; i++) {
                    float diff = std::abs(output_logits[i] - answer_logits[i]);
                    total_values++;

                    if (diff > THRESHOLD) {
                        if (first_mismatch_idx == -1) {
                            first_mismatch_idx = sample_idx * VOCAB_SIZE + i;
                            first_mismatch_output = output_logits[i];
                            first_mismatch_answer = answer_logits[i];
                        }
                        mismatches++;
                    }
                }

                // Check top-1 prediction
                int top1_output =
                    std::max_element(output_logits.begin(), output_logits.end())
                    - output_logits.begin();
                int top1_answer =
                    std::max_element(answer_logits.begin(), answer_logits.end())
                    - answer_logits.begin();

                if (top1_output == top1_answer) {
                    top1_matches++;
                }
            }

            outfile_read.close();
            ansfile.close();

            std::cout << std::endl;

            // Print top-1 accuracy
            float top1_accuracy = (float)top1_matches / num_compare * 100.0f;
            std::cout << "Top-1 Prediction Accuracy: " << top1_accuracy << "% " 
                      << "(" << top1_matches << "/" << num_compare << ")"
                      << std::endl;

            // Final verdict
            if (mismatches == 0) {
                fprintf(stdout, "VALID\n");
            } else {
                fprintf(stdout, "INVALID\n");
                if (first_mismatch_idx != -1) {
                    int sample_num = first_mismatch_idx / VOCAB_SIZE;
                    int vocab_idx = first_mismatch_idx % VOCAB_SIZE;
                    fprintf(stdout, "First mismatch at sample[%d], vocab[%d] "
                            "(output[%d]=%.6f <-> answer[%d]=%.6f)\n",
                            sample_num, vocab_idx, first_mismatch_idx,
                            first_mismatch_output,
                            first_mismatch_idx, first_mismatch_answer);
                }
                fprintf(stdout, "Total mismatches: %d/%d\n", mismatches, total_values);
            }
        }

        // Free pinned memory
        CHECK_CUDA(cudaFreeHost(inputs));
        CHECK_CUDA(cudaFreeHost(outputs));
    }

    /* MPI Finalization */
    MPI_Finalize();
    return 0;
}

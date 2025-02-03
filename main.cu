#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <vector>


#define MAX_MATRIX_LENGTH 7500
#define MAX_FRONTIER_SIZE 256

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

__constant__ int destinations_constant[MAX_MATRIX_LENGTH];

__global__ void BFS_CUDA(const int* rowPointers_d,
                         int* distances_d, int* currentFrontier_d, int* currentFrontierSize_d,
                         int dim, int max_frontier_size) {
    extern __shared__ int shared_frontier[];  // extern because the dimension is defined in the kernel call
    __shared__ int shared_frontier_size;

    // initialize shared frontier size
    if (threadIdx.x == 0) {
        shared_frontier_size = 0;
    }

    __syncthreads();

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < dim) {
        // get current vertex
        int currentVertex = currentFrontier_d[i];
        // printf("Thread %d processing vertex %d\n", i, currentVertex); //debugging function

        // get start and end of the destinations
        int start = rowPointers_d[currentVertex];
        int end = rowPointers_d[currentVertex + 1];

        for (int j = start; j < end; ++j) {
            int neighbor = destinations_constant[j];
            // printf("Thread %d processing edge %d -> %d, j = %d\n", i, currentVertex, neighbor, j); //debugging function

            // check if it has already been visited and update the distance
            if (atomicCAS(&distances_d[neighbor], -1, distances_d[currentVertex] + 1) == -1) {
                // printf("Thread %d updating distance for vertex %d: %d\n", i, neighbor, distances_d[currentVertex] + 1); //debugging function

                // if it is new, add neighbor to current frontier
                int idx = atomicAdd(&shared_frontier_size, 1);
                if (idx < max_frontier_size) {
                    shared_frontier[idx] = neighbor;
                    // printf("Thread %d added vertex %d to shared frontier at index %d\n", i, neighbor, idx); // debugging function
                } else {
                    printf("Thread %d could not add vertex %d to shared frontier (overflow)\n", i, neighbor); // check for overflow
                }
            }
        }
    }

    __syncthreads(); // wait for all the threads to finish computations

    // update global memory frontier
    if (threadIdx.x == 0) {
        int global_idx = atomicExch(currentFrontierSize_d, shared_frontier_size); // updates the current frontier size and returns the previous size
        for (int j = 0; j < shared_frontier_size; ++j) {
            if (global_idx + j < max_frontier_size) {
                currentFrontier_d[j] = shared_frontier[j];
                // printf("Block %d added vertex %d to global frontier at index %d\n", blockIdx.x, shared_frontier[j], j);
            } else {
                printf("Block %d could not add vertex %d to global frontier (overflow)\n", blockIdx.x, shared_frontier[j]); // debugging function
            }
        }
    }
}

void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals);

void insertIntoFrontier(int val, int *frontier, int *frontier_size) {
    frontier[*frontier_size] = val;
    *frontier_size = *frontier_size + 1;
}

inline void swap(int **ptr1, int **ptr2) {
    int *tmp = *ptr1;
    *ptr1    = *ptr2;
    *ptr2    = tmp;
}


void BFS_parallel(const int source, const int* rowPointers, const int* destinations,
                  int* distances, const int num_rows, int num_vals) {
    int *currentFrontier_d, *currentFrontierSize_d;
    int *distances_d, *rowPointers_d;

    // device memory allocation
    CHECK(cudaMalloc(&currentFrontier_d, MAX_FRONTIER_SIZE * sizeof(int)));
    CHECK(cudaMalloc(&currentFrontierSize_d, sizeof(int)));
    CHECK(cudaMalloc(&distances_d, num_rows * sizeof(int)));
    CHECK(cudaMalloc(&rowPointers_d, num_rows * sizeof(int)));
    //CHECK(cudaMalloc(&destinations_d, num_vals * sizeof(int)));

    // initialize
    int initialDistances[num_rows];
    for (int i = 0; i < num_rows; ++i) {
        initialDistances[i] = -1;  // all the distances have to be -1
    }
    initialDistances[source] = 0;  // except the starting value, which has to be 1

    // copy to device memory
    CHECK(cudaMemcpy(distances_d, initialDistances, num_rows * sizeof(int), cudaMemcpyHostToDevice));
    int initialFrontierSize = 1;
    CHECK(cudaMemcpy(currentFrontier_d, &source, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(currentFrontierSize_d, &initialFrontierSize, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(rowPointers_d, rowPointers, num_rows * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(destinations_constant, destinations, num_vals * sizeof(int))); // copy to constant memory


    while (initialFrontierSize > 0) {
        int block_Dim = 256;
        int grid_Dim = (initialFrontierSize + block_Dim - 1) / block_Dim;
        int sharedMemSize = block_Dim * sizeof(int);


        // launch the kernel
        BFS_CUDA<<<grid_Dim, block_Dim, sharedMemSize>>>(rowPointers_d, distances_d,
                                                       currentFrontier_d, currentFrontierSize_d,
                                                       initialFrontierSize, MAX_FRONTIER_SIZE);
        CHECK_KERNELCALL();


        // updates frontier dimensions
        CHECK(cudaMemcpy(&initialFrontierSize, currentFrontierSize_d, sizeof(int), cudaMemcpyDeviceToHost));


        // Reset dimensione frontiera per il prossimo passo
        CHECK(cudaMemset(currentFrontierSize_d, 0, sizeof(int)));
    }

    // Copia risultato su host
    CHECK(cudaMemcpy(distances, distances_d, num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    // Libera memoria
    cudaFree(currentFrontier_d);
    cudaFree(currentFrontierSize_d);
    cudaFree(distances_d);
    cudaFree(rowPointers_d);
    // cudaFree(destinations_d);
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./exec matrix_file source\n");
        return 0;
    } // checks if the number of arguments passed is correct, otherwise returns an error message

    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<float> values;
    int num_rows, num_cols, num_vals;

    const std::string filename{argv[1]}; // argv[1] is the name of the file passed
    // The node starts from 1 but array starts from 0
    const int source = atoi(argv[2]) - 1; // source is the starting point of the algorithm (zero-indexed)

    read_matrix(row_ptr, col_ind, values, filename, num_rows, num_cols, num_vals);

    if (num_vals>MAX_MATRIX_LENGTH){
        printf("Matrix is too big");
        return 0;
    }

    // Initialize dist to -1
    std::vector<int> dist(num_rows); //before it was num_vals
    for (int i = 0; i < num_rows; i++) { dist[i] = -1; }

    clock_t start, end;
    start = clock();

    BFS_parallel(source, row_ptr.data(), col_ind.data(), dist.data(), num_rows, num_vals); // .data() returns a pointer to the first element
    // of the array

    end = clock();

    printf("\nFinal distances:\n");
    for (int i=0; i<num_rows; i++) {
        printf("%d ", dist[i]);
    }

    printf("\nTime elapsed: %f ms", float(end-start)*1000/CLOCKS_PER_SEC);

    return EXIT_SUCCESS;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(std::vector<int> &row_ptr, // row_ptr will get filled with the row indexes of the array value
        // corresponding to the beginning of the new row
                 std::vector<int> &col_ind, // col_ind will get filled with the column indexes of the values
                 std::vector<float> &values, // values will get filled with the non-zero values of the matrix
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "File cannot be opened!\n";
        throw std::runtime_error("File cannot be opened");
    }

    // Get number of rows, columns, and non-zero values
    file >> num_rows >> num_cols >> num_vals;// these values are in the first row of the file

    row_ptr.resize(num_rows + 1); // changing the size of the arrays
    col_ind.resize(num_vals);
    values.resize(num_vals);

    // Collect occurrences of each row for determining the indices of row_ptr
    std::vector<int> row_occurrences(num_rows, 0);

    int row, column;
    float value;
    while (file >> row >> column >> value) {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;

        row_occurrences[row]++;
    }

    // Set row_ptr
    int index = 0;
    for (int i = 0; i < num_rows; i++) {
        row_ptr[i] = index;
        index += row_occurrences[i];
    }
    row_ptr[num_rows] = num_vals;

    // Reset the file stream to read again from the beginning
    file.clear();
    file.seekg(0, std::ios::beg);

    // Read the first line again to skip it
    file >> num_rows >> num_cols >> num_vals;

    std::fill(col_ind.begin(), col_ind.end(), -1);

    int i = 0;
    while (file >> row >> column >> value) {
        row--;
        column--;

        // Find the correct index (i + row_ptr[row]) using both row information and an index i
        while (col_ind[i + row_ptr[row]] != -1) { i++; }
        col_ind[i + row_ptr[row]] = column;
        values[i + row_ptr[row]]  = value;
        i                         = 0;
    }
}
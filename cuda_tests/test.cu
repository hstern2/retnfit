#include <stdio.h>
#include <iostream>
#include <math.h>

#define MAX_NODES 2

using namespace std;

typedef struct test
{
    int n;
    int *arr;
} test;

typedef struct trajectory
{
    int n_node;
    int repetition_start, repetition_end;
    int is_persistent[MAX_NODES];
    int **state; /* max_states x MAX_NODES */
    int steady_state[MAX_NODES];
} *trajectory_t;

typedef struct network
{
    int n_node, n_parent, n_outcome;
    int **parent;
    int **outcome;
} * network_t;

typedef struct experiment
{
    double score[MAX_NODES][3];
    int n_perturbed, perturbed[MAX_NODES];
} * experiment_t;

typedef struct experiment_set
{
    int n_experiment, n_node;
    experiment_t experiment;
} * experiment_set_t;

int **initialize_2d_array(int rows, int cols)
{
    int *data = (int *)calloc(rows * cols, sizeof(double));
    int **arr = (int **)calloc(rows, sizeof(double *));
    for (int i = 0; i < rows; i++)
        arr[i] = &(data[cols * i]);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            arr[i][j] = (i * cols) + j;
        }
    }
    return arr;
}

void print_matrix(int **mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

__global__ void kernel(test *t)
{
    int idx = threadIdx.x;
    printf("Inside kernel %d\n", idx);
    // int n = t->n;
    printf("Size of data %d\n", t->n);
    printf("Data inside array %d\n", t->arr[0]);
    // for (int i=0;i<5;i++) {
    //     printf("Thread %d : %d ", idx, t.arr[i]);
    // }
}

__global__ void networkLoadTest(network_t network)
{
    int idx = threadIdx.x;
    printf("Inside kernel %d\n", idx);
    printf("Number of nodes %d\n", network->n_node);
    printf("Number of parents %d\n", network->n_parent);
    printf("Number of outcomes %d\n", network->n_outcome);
    int n = network->n_parent;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ", network->parent[i][j]);
        }
        printf("\n");
    }
}

void test_without_unified_memory()
{
    // testing code
    test *t;
    t = (test *)malloc(sizeof(t));
    t->arr = (int *)malloc(sizeof(int) * 5);
    t->n = 5;
    t->arr[0] = 1;

    test *d_t;
    int *d_arr;
    int sizelen = 5 * sizeof(int);

    cudaMalloc(&d_t, sizeof(test));
    // cudaMalloc(d_t->n, sizeof(int));
    cudaMalloc(&d_arr, sizelen);

    cudaMemcpy(d_t, t, sizeof(test), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr, t->arr, sizelen, cudaMemcpyHostToDevice);
    std::cout << "Copying n" << std::endl;
    cudaMemcpy(&d_t->n, &t->n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_t->arr), &d_arr, sizeof(int *), cudaMemcpyHostToDevice);

    // std::cout<<t->n;
    kernel<<<1, 1>>>(d_t);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

__global__ void experiment_load_test(experiment_t e) {
    int idx = threadIdx.x;
    printf("Inside kernel for experiment %d\n", idx);
    printf("Number of nodes %d\n", e->n_perturbed);
    printf("Score:\n");
    for (int i=0;i<MAX_NODES;i++) {
        for (int j=0;j<3;j++) {
            printf("%lf ", e->score[i][j]);
        }
        printf("\n");
    }
    printf("Perturbed:\n");
    for (int i=0;i<MAX_NODES;i++) {
        printf("%d ", e->perturbed[i]);
    }
    printf("\n");
}

__global__ void experiment_set_load_test(experiment_set_t eset) {
    int idx = threadIdx.x;
    printf("Inside kernel for experiment set %d\n", idx);
    printf("Number of nodes %d\n", eset->n_node);
    printf("Number of experiments %d\n", eset->n_experiment);

    experiment_t e = &eset->experiment[idx];

    experiment_load_test<<<1,1>>>(e);
}

__global__ void trajectory_new_test(trajectory_t t) {
    int idx = threadIdx.x;
    trajectory_t curr = &t[idx];
    printf("Inside kernel for trajectory %d\n", idx);
    printf("Number of n_node %d\n", curr->n_node);
    printf("Number of trajectory start %d\n", curr->repetition_start);
    printf("Number of trajectory end %d\n", curr->repetition_end);
    printf("is persistent\n");
    for (int i=0;i<MAX_NODES;i++) 
    {
        printf("%d ", curr->is_persistent[i]);
    }
    printf("\n");
    printf("state\n");
    for (int i=0;i<10;i++)
    {
        for (int j=0;j<MAX_NODES;j++) 
        {
            printf("%d ", curr->state[i][j]);
        }
        printf("\n");
    }
    printf("steady state\n");
    for (int i=0;i<MAX_NODES;i++) 
    {
        printf("%d ", curr->steady_state[i]);
    }
    printf("\n");

}

experiment_t load_experiment_to_gpu(experiment_t e) 
{
    experiment_t d_e;

    cudaMallocManaged(&d_e, sizeof(experiment));
    cudaMemcpy(d_e, e, sizeof(experiment), cudaMemcpyHostToDevice);
    return d_e;
}

trajectory_t new_trajectory_gpu(int ntraj, int max_states, int n_node) 
{
    trajectory_t d_t;
    cudaMallocManaged(&d_t, ntraj*sizeof(trajectory));
    for (int i=0;i<ntraj;i++) 
    {
        int *data;
        trajectory_t curr = &d_t[i];
        cudaMallocManaged(&data, max_states*n_node*sizeof(int));
        cudaMallocManaged(&curr->state, max_states * sizeof(int *));
        for (int j=0;j<max_states;j++) 
        {
            curr->state[j] = &(data[j*n_node]);
        }
    }
    return d_t;
}

experiment_set_t load_experiment_set_to_gpu(experiment_set_t eset) 
{
    experiment_set_t d_eset;
    const size_t size = sizeof(experiment_set);
    cout<<"Size of eset "<<size<<endl;

    cudaMallocManaged(&d_eset, size);
    cudaMallocManaged(&d_eset->experiment, eset->n_experiment*sizeof(experiment));
    d_eset->n_node = eset->n_node;
    d_eset->n_experiment = eset->n_experiment;
    cudaMemcpy(d_eset->experiment, eset->experiment, eset->n_experiment*sizeof(experiment), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_eset, eset, size, cudaMemcpyHostToDevice);
    experiment_set_load_test<<<1,2>>>(d_eset);
    return d_eset;
}

network_t load_network_to_gpu(network_t n)
{
    network_t d_n;

    cudaMallocManaged(&d_n, sizeof(network_t));
    d_n->n_node = n->n_node;
    d_n->n_parent = n->n_parent;
    d_n->n_outcome = n->n_outcome;

    int *parent_data;
    int parent_size = n->n_parent;
    cudaMallocManaged(&parent_data, parent_size * parent_size * sizeof(int));
    cudaMallocManaged(&d_n->parent, parent_size * sizeof(int *));

    for (int i=0;i<parent_size;i++) {
        for (int j=0;j<parent_size;j++) {
            parent_data[i*parent_size+j] = n->parent[i][j];
        }
    }

    for (int i=0;i<parent_size;i++) {
        d_n->parent[i] = &(parent_data[i*2]);
    }

    int *outcome_data;
    int outcome_size = n->n_outcome;
    cudaMallocManaged(&outcome_data, outcome_size*outcome_size*sizeof(int));
    cudaMallocManaged(&d_n->outcome, outcome_size * sizeof(int *));

    for (int i=0;i<outcome_size;i++) {
        for (int j=0;j<outcome_size;j++) {
            outcome_data[i*outcome_size+j] = n->outcome[i][j];
        }
    }

    for (int i=0;i<outcome_size;i++) {
        d_n->outcome[i] = &(outcome_data[i*outcome_size]);
    }

    return d_n;
}

void test_trajectory_new() 
{
    trajectory_t t = new_trajectory_gpu(2,10,2);

    trajectory_new_test<<<1,2>>>(t);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

}

void test_network_load() 
{
    network_t network = (network_t)malloc(sizeof(network_t));
    int n = 2;
    network->n_node = 2;
    network->n_parent = 2;
    network->n_outcome = 2;
    network->parent = initialize_2d_array(n, n);
    network->outcome = initialize_2d_array(n, n);
    // std::cout << (sizeof network->parent) * (sizeof network->parent[0]) << std::endl;
    network_t d_n = load_network_to_gpu(network);

    networkLoadTest<<<1, 1>>>(d_n);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

void test_experiment_load() {
    experiment_t e = (experiment_t)malloc(sizeof(struct experiment));
    e->n_perturbed = 2;
    for (int i=0;i<MAX_NODES;i++) {
        for (int j=0;j<3;j++) {
            e->score[i][j] = (i*2)+j;
        }
    }
    for (int i=0;i<MAX_NODES;i++) {
        e->perturbed[i] = i;
    }
    cout<<"Copied data";
    load_experiment_to_gpu(e);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

void test_experiment_set_load() 
{
    experiment_set_t eset;
    eset = (experiment_set_t)malloc(sizeof(experiment_set));
    eset->n_experiment = 2;
    eset->n_node = 1;
    eset->experiment = (experiment_t)malloc(sizeof(experiment)*eset->n_experiment);
    eset->experiment[0].n_perturbed = 2;
    for (int i=0;i<MAX_NODES;i++) {
        for (int j=0;j<3;j++) {
            eset->experiment[0].score[i][j] = (i*2)+j;
        }
    }
    for (int i=0;i<MAX_NODES;i++) {
        eset->experiment[0].perturbed[i] = i;
    }

    eset->experiment[1].n_perturbed = 3;
    for (int i=0;i<MAX_NODES;i++) {
        for (int j=0;j<3;j++) {
            eset->experiment[1].score[i][j] = (i*2)+j;
        }
    }
    for (int i=0;i<MAX_NODES;i++) {
        eset->experiment[1].perturbed[i] = i;
    }

    load_experiment_set_to_gpu(eset);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

int main()
{
    test_trajectory_new();
}
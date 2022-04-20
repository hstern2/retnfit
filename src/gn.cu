#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <Rmath.h>
// #include <iostream>
#include "array.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"
#include "gn.h"

#define UNDEFINED 9
#define LARGE_SCORE 1e9

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
        d_n->parent[i] = &(parent_data[i*parent_size]);
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

experiment_set_t load_experiment_set_to_gpu(experiment_set_t eset) {
    experiment_set_t d_eset;
    const size_t size = sizeof(experiment_set);
    const size_t size_of_experiments = eset->n_experiment*sizeof(experiment)
    cudaMallocManaged(&d_eset, size);
    cudaMallocManaged(&d_eset->experiment, size_of_experiments);
    d_eset->n_node = eset->n_node;
    d_eset->n_experiment = eset->n_experiment;
    cudaMemcpy(d_eset->experiment, eset->experiment, size_of_experiments, cudaMemcpyHostToDevice);
    return d_eset;
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

static int state_from_sym(char c)
{
  switch (c) {
  case '-': 
    return -1;
  case '.':
    return 0;
  case '+':
    return 1;
  case '?':
    return UNDEFINED;
  default:
    die("state_from_sym: unknown character: '%c'.  must be one of - . + ?", c);
  }
  return 0;
}

static char sym_from_state(int s)
{
  switch (s) {
  case -1:
    return '-';
  case 0:
    return '.';
  case 1:
    return '+';
  case UNDEFINED:
    return '?';
  default:
    die("sym_from_state: unknown state: %d", s);
  }
  return 0;
}

static int random_outcome()
{
  return random_int_inclusive(-1,1);
}

static int random_parent(int i, int n_node, int n_parent, const int *p)
{
  int k;
  for (k = 0; k < 10000000; k++) {
    const int pnew = random_int_inclusive(0, n_node-1);
    if (pnew != i) {
      int j;
      for (j = 0; j < n_parent; j++)
        if (pnew == p[j])
          break;
      if (j == n_parent)
        return pnew;
    }
  }
  die("random_parent: whoops");
  return 0;
}

unsigned three_to_the(unsigned n)
{
  unsigned a = 1;
  while (n-- > 0)
    a *= 3;
  return a;
}

void network_init(network_t n, int n_node, int max_parents)
{
  n->n_node = n_node;
  if (max_parents > n_node - 1)
    max_parents = n_node - 1;
  n->n_parent = max_parents;
  n->n_outcome = three_to_the(max_parents);
  n->parent = int_array2D_new(n->n_node, n->n_parent);
  n->outcome = int_array2D_new(n->n_node, n->n_outcome);
}

void network_randomize_parents(network_t n)
{
  int i;
  for (i = 0; i < n->n_node; i++) {
    int j;
    for (j = 0; j < n->n_parent; j++)
      n->parent[i][j] = random_parent(i, n->n_node, j, n->parent[i]);
    qsort(&n->parent[i][0], n->n_parent, sizeof(int), intcmp);
  }
}

void network_set_outcomes_to_null(network_t n)
{
  int i;
  for (i = 0; i < n->n_node; i++) {
    int j;
    for (j = 0; j < n->n_outcome; j++)
      n->outcome[i][j] = state_from_sym('.');
  }
}

void network_randomize_outcomes(network_t n)
{
  int i;
  for (i = 0; i < n->n_node; i++) {
    int j;
    for (j = 0; j < n->n_outcome; j++)
      n->outcome[i][j] = random_outcome();
  }
}

void network_write_to_intp(const network_t n, int *parent, int *outcome)
{
  int i, j;
  for (i = 0; i < n->n_node; i++)
    for (j = 0; j < n->n_parent; j++)
      parent[j*n->n_node+i] = n->parent[i][j];
  for (i = 0; i < n->n_node; i++) {
    int k;
    for (k = 0; k < n->n_outcome; k++)
      outcome[k*n->n_node+i] = n->outcome[i][k];
  }  
}

void network_read_parents_from_intp(network_t n, const int *parents)
{
  int i, j;
  for (i = 0; i < n->n_node; i++)
    for (j = 0; j < n->n_parent; j++)
      n->parent[i][j] = parents[j*n->n_node+i];
}

void network_read_outcomes_from_intp(network_t n, const int *outcomes)
{
  int i, j;
  for (i = 0; i < n->n_node; i++)
    for (j = 0; j < n->n_outcome; j++)
      n->outcome[i][j] = outcomes[j*n->n_node+i];
}

void network_delete(network_t n)
{
  int_array2D_delete(n->parent);
  int_array2D_delete(n->outcome);
}

void network_write_to_file(FILE *f, const network_t n)
{
  int i;
  for (i = 0; i < n->n_node; i++) {
    int ip;
    for (ip = 0; ip < n->n_parent; ip++)
      fprintf(f, "%d ", n->parent[i][ip]);
    if (n->n_parent > 0) {
      for (ip = 0; ip < n->n_outcome; ip++)
	fprintf(f, "%c", sym_from_state(n->outcome[i][ip]));
    }
    fprintf(f, "\n");
  }
}

static int repetition_found(const trajectory_t t)
{
  return t->repetition_end > 0;
}

static void advance(const network_t n, trajectory_t traj, int i_state)
{
  const int n_node = n->n_node;
  /* find new state */
  int *si = &traj->state[i_state][0];
  const int *si1 = &traj->state[i_state - 1][0];
  int i_node;
  for (i_node = 0; i_node < n_node; i_node++) {
    if (traj->is_persistent[i_node] || n->n_parent == 0) {
      si[i_node] = si1[i_node];
    } else {
      int ip, a = 0;
      for (ip = n->n_parent - 1; ip >= 0; ip--) {
	a *= 3;
	a += si1[n->parent[i_node][ip]] + 1;
      }
      si[i_node] = n->outcome[i_node][a];
    }
  }
}

static void check_for_repetition(trajectory_t traj, int i_state)
{
  const int n_node = traj->n_node;
  int i_node, j_state;
  const int *si = &traj->state[i_state][0];
  for (j_state = 0; j_state < i_state; j_state++) {
    const int *sj = &traj->state[j_state][0];
    for (i_node = 0; i_node < n_node; i_node++)
      if (si[i_node] != sj[i_node])
	break;
    if (i_node < n_node)
      continue;
    /* repetition found - create summary */
    traj->repetition_start = j_state;
    traj->repetition_end = i_state;
    for (i_node = 0; i_node < traj->n_node; i_node++) {
      int k, visited_plus = 0, visited_minus = 0;
      for (k = traj->repetition_start; k <= traj->repetition_end; k++)
	if (traj->state[k][i_node] == 1)
	  visited_plus = 1;
	else if (traj->state[k][i_node] == -1)
	  visited_minus = 1;
      if (visited_plus && visited_minus)
	traj->steady_state[i_node] = UNDEFINED;
      else if (visited_plus)
	traj->steady_state[i_node] = 1;
      else if (visited_minus)
	traj->steady_state[i_node] = -1;
      else
	traj->steady_state[i_node] = 0;
    }
    return;
  }
  /* no repetition found */
  traj->repetition_start = 0;
  traj->repetition_end = 0;
  for (i_node = 0; i_node < n_node; i_node++)
    traj->steady_state[i_node] = UNDEFINED;
}

static double score_for_state(const experiment_t e, int i, int s)
{
  return e->score[i][s+1];
}

static void set_score_for_state(const experiment_t e, int i, int s, double val)
{
  e->score[i][s+1] = val;
}

static int most_probable_state(const experiment_t e, int i)
{
  int min_s = -1;
  double min = score_for_state(e,i,-1);
  int s;
  for (s = 0; s <= 1; s++)
    if (score_for_state(e,i,s) < min) {
      min_s = s;
      min = score_for_state(e,i,s);
    }
  return min_s;
}

static double score_for_most_probable_state(const experiment_t e, int j)
{
  return score_for_state(e,j,most_probable_state(e,j));
}

trajectory_t trajectories_new(int ntraj, int max_states, int n_node)
{
  trajectory_t t = (trajectory_t) safe_malloc(ntraj*sizeof(struct trajectory));
  int i;
  for (i = 0; i < ntraj; i++)
    t[i].state = int_array2D_new(max_states, n_node);
  return t;
}

void trajectories_delete(trajectory_t t, int ntraj)
{
  int i;
  for (i = 0; i < ntraj; i++)
    int_array2D_delete(t[i].state);
  free(t);
}

static void init_trajectory(trajectory_t t, const experiment_t e, int n_node)
{
  t->n_node = n_node;
  int i;
  for (i = 0; i < t->n_node; i++) {
    t->is_persistent[i] = 0;
    t->state[0][i] = 0;
  }
  t->repetition_start = t->repetition_end = 0;
  for (i = 0; i < e->n_perturbed; i++) {
    const int j = e->perturbed[i];
    t->is_persistent[j] = 1;
    t->state[0][j] = most_probable_state(e,j);
  }
}

void experiment_set_init(experiment_set_t e, 
			 int n,
			 const int *i_exp, 
			 const int *i_node, 
			 const int *outcome,
			 const double *val,
			 const int *is_perturbation)
{
  e->n_experiment = 0;
  e->n_node = 0;
  int i, j_exp;
  for (i = 0; i < n; i++) {
    if (i_exp[i] >= e->n_experiment)
      e->n_experiment = i_exp[i] + 1;
    if (i_node[i] >= e->n_node)
      e->n_node = i_node[i] + 1;
  }
  e->experiment = (experiment_t) safe_malloc(e->n_experiment * sizeof(struct experiment));
  for (j_exp = 0; j_exp < e->n_experiment; j_exp++)
    e->experiment[j_exp].n_perturbed = 0;
  for (i = 0; i < n; i++) {
    experiment_t en = &e->experiment[i_exp[i]];
    set_score_for_state(en, i_node[i], outcome[i], val[i]);
    if (is_perturbation[i])
      en->perturbed[en->n_perturbed++] = i_node[i];
  }
}

void experiment_set_delete(experiment_set_t e)
{
  free(e->experiment);
}

static int is_node_perturbed(experiment_t e, int i_node)
{
  int i;
  for (i = 0; i < e->n_perturbed; i++)
    if (e->perturbed[i] == i_node)
      return 1;
  return 0;
}

void experiment_set_write(FILE *f, const experiment_set_t e)
{
  int i_exp, i_node, i_state;
  for (i_exp = 0; i_exp < e->n_experiment; i_exp++) {
    const experiment_t ei = &e->experiment[i_exp];
    for (i_node = 0; i_node < e->n_node; i_node++)
      for (i_state = -1; i_state <= 1; i_state++)
	fprintf(f, "%d %d %c %f %d\n", i_exp, i_node, sym_from_state(i_state), score_for_state(ei,i_node,i_state), is_node_perturbed(ei, i_node));
  }
}

static void write_state(FILE *f, const int *state, int n_node)
{
  int i;
  for (i = 0; i < n_node; i++)
    fprintf(f, "%c", sym_from_state(state[i]));
}

static void write_repetition(FILE *f, const trajectory_t t)
{
  int i;
  for (i = 0; i <= t->repetition_end; i++) {
    fprintf(f, "%d: ", i);
    write_state(f, &t->state[i][0], t->n_node);
    fprintf(f, "\n");
  }
  fprintf(f, "s: ");
  write_state(f, t->steady_state, t->n_node);
  fprintf(f, "\n");
}

static void write_most_probable(FILE *f, const experiment_t e, int n_node)
{
  int i;
  for (i = 0; i < n_node; i++)
    fprintf(f, "%c", sym_from_state(most_probable_state(e,i)));
  fprintf(f, "\n");
}

double scale_factor(const experiment_set_t eset)
{
  return 1.0 / (eset->n_experiment * eset->n_node);
}

double lowest_possible_score(const experiment_set_t eset)
{
  double s = 0;
  int i;
  for (i = 0; i < eset->n_experiment; i++) {
    const experiment_t e = &eset->experiment[i];
    int j;
    for (j = 0; j < eset->n_node; j++)
      s += score_for_most_probable_state(e,j);
  }
  return s;
}

void network_advance_until_repetition(const network_t n, const experiment_t e, trajectory_t t, int max_states)
{
  init_trajectory(t, e, n->n_node);
  int i;
  for (i = 1; i < max_states && !repetition_found(t); i++) {
    advance(n,t,i);
    check_for_repetition(t,i);
  }
}

void network_write_response_as_target_data(FILE *f, network_t n, const experiment_set_t e, int max_states)
{
  const int n_node = n->n_node;
  if (n_node != e->n_node)
    die("network_write_response_as_target_data: network has %d nodes, experiment set has %d nodes",
	n_node, e->n_node);
  fprintf(f, "i_exp, i_node, outcome, value, is_perturbation\n");
  trajectory_t trajectories = trajectories_new(e->n_experiment, max_states, n_node);
  int i_exp;
  for (i_exp = 0; i_exp < e->n_experiment; i_exp++) {
    trajectory_t traj = &trajectories[i_exp];
    network_advance_until_repetition(n, &e->experiment[i_exp], traj, max_states);
    int i_node;
    for (i_node = 0; i_node < n_node; i_node++) {
      int i_outcome;
      for (i_outcome = -1; i_outcome <= 1; i_outcome++)
	fprintf(f, "%d, %d, %d, %.1f, %d\n",
		i_exp, i_node, i_outcome,
		fabs((double) traj->steady_state[i_node] - (double) i_outcome),
		traj->is_persistent[i_node] && traj->steady_state[i_node] == i_outcome);
    }
  }
  trajectories_delete(trajectories, e->n_experiment);
}

void network_write_response_from_experiment_set(FILE *f, network_t n, const experiment_set_t e, int max_states)
{
  const int n_node = n->n_node;
  if (n_node != e->n_node)
    die("network_write_response_from_experiment_set: network has %d nodes, experiment set has %d nodes",
	n_node, e->n_node);
  int i;
  trajectory_t trajectories = trajectories_new(e->n_experiment, max_states, n_node);
  for (i = 0; i < e->n_experiment; i++) {
    trajectory_t traj = &trajectories[i];
    fprintf(f, "experiment %d:\n", i);
    network_advance_until_repetition(n, &e->experiment[i], traj, max_states);
    write_repetition(f,traj);
    fprintf(f, "\n");
  }
  fprintf(f, "Lowest possible score: %g\n", lowest_possible_score(e));
  fprintf(f, "Most probable and predicted steady states:\n");
  for (i = 0; i < e->n_experiment; i++) {
    trajectory_t traj = &trajectories[i];
    write_most_probable(f, &e->experiment[i], n_node);
    network_advance_until_repetition(n, &e->experiment[i], traj, max_states);
    write_state(f, traj->steady_state, n_node);
    fprintf(f, "\n\n");
  }
  trajectories_delete(trajectories, e->n_experiment);
}

static double score_for_trajectory(const experiment_t e, const trajectory_t t)
{
  int i_node;
  double s = 0;
  for (i_node = 0; i_node < t->n_node; i_node++) {
    const int si = t->steady_state[i_node];
    if (si == UNDEFINED)
      return LARGE_SCORE;
    s += score_for_state(e, i_node, si);
  }
  return s;
}

__global__ void cuda_init_trajectory(trajectory_t t, const experiment_t e, int n_node) {
  t->n_node = n_node;
  int i;
  for (i = 0; i < t->n_node; i++) {
    t->is_persistent[i] = 0;
    t->state[0][i] = 0;
  }
  t->repetition_start = t->repetition_end = 0;
  for (i = 0; i < e->n_perturbed; i++) {
    const int j = e->perturbed[i];
    t->is_persistent[j] = 1;
    t->state[0][j] = most_probable_state(e,j);
  }
}

__global__ void cuda_score_device(int n, network_t net, const experiment_set_t eset, trajectory_t trajectories, double limit, int max_states, double *s_kernels) {
  // TODO: something 
  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < n) {
    const experiment_t e = &eset->experiment[globalIdx];
    trajectory_t traj = &trajectories[globalIdx];
    // TODO: how call function from within the kernel?
    network_advance_until_repetition(net, e, traj, max_states);
    const double s = repetition_found(traj) ? score_for_trajectory(e, traj) : limit;
    s_kernels[globalIdx] = s;
  }

} 

static double cuda_score_host(network_t n, const experiment_set_t eset, trajectory_t trajectories, double limit, int max_states) {
  double s_tot = 0;
  // initialize memory
  int N = eset->n_experiments;
  // TODO: network_t, experiment_set_t, trajectory_t
  double d_limits;
  int d_max_states;
  double *s_kernels, *d_s_kernels;
  s_kernels = (double*)malloc(N*sizeof(double));
  for (i = 0; i < N; i++) {
    s_kernels[i] = 0.0;
  }
  // copy data
  cudaMalloc(&d_limits, sizeof(double));
  cudaMalloc(&d_max_states, sizeof(int));
  cudaMalloc(&d_s_kernels, N*sizeof(double));

  cudaMemcpy(d_max_states, max_states, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_limits, limits, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_kernels, s_kernels, N*sizeof(double), cudaMemcpyHostToDevice);

  // launch kernel
  // TODO: figure how to sync (s_tot <= limit) check 

  // synchronize and free memomry


  // calculate s_total
  for (i = 0; i < eset->n_experiments; i++) {
    s_tot += s_kernels[i];
  }

  return s_tot;

}

static double score(network_t n, const experiment_set_t eset, trajectory_t trajectories, 
                    double limit, int max_states)
{
  double s_tot = 0;
  int i_exp;
#pragma omp parallel for
  for (i_exp = 0; i_exp < eset->n_experiment; i_exp++)
    if (s_tot <= limit) {
      const experiment_t e = &eset->experiment[i_exp];
      trajectory_t traj = &trajectories[i_exp];
      network_advance_until_repetition(n, e, traj, max_states);
      const double s = repetition_found(traj) ? score_for_trajectory(e, traj) : limit;
#pragma omp atomic
      s_tot += s;
    }
  return s_tot;
}

static void copy_network(network_t to, const network_t from)
{
  memcpy(&to->parent[0][0], &from->parent[0][0], to->n_node * to->n_parent * sizeof(int));
  memcpy(&to->outcome[0][0], &from->outcome[0][0], to->n_node * to->n_outcome * sizeof(int));
} 

#ifdef USE_MPI

static void send_int(int proc, int buf)
{
  MPI_Send(&buf, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
}

static int receive_int(int proc)
{
  int buf;
  MPI_Status s;
  MPI_Recv(&buf, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, &s);
  return buf;
}

static void send_double(int proc, double buf)
{
  MPI_Send(&buf, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
}

static double receive_double(int proc)
{
  double buf;
  MPI_Status s;
  MPI_Recv(&buf, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &s);
  return buf;
}

static void exchange_networks(int proc, network_t n)
{
  MPI_Status s;
  MPI_Sendrecv_replace(&n->parent[0][0], n->n_node * n->n_parent, MPI_INT,
		       proc, 0, proc, 0, MPI_COMM_WORLD, &s);
  MPI_Sendrecv_replace(&n->outcome[0][0], n->n_node * n->n_outcome, MPI_INT,
		       proc, 0, proc, 0, MPI_COMM_WORLD, &s);
}

static void bcast_int(int *x)
{
  MPI_Bcast(x, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

#endif

static double fraction(unsigned long a, unsigned long b)
{
  if (b > 0)
    return (double) a / (double) b;
  return 0;
}

double network_monte_carlo(network_t n, 
			   const experiment_set_t e, 
			   unsigned long n_cycles,
			   unsigned long n_write,
			   double T_lo,
			   double T_hi,
			   FILE *out,
                           int n_thread,
			   double target_score,
                           unsigned long exchange_interval,
                           unsigned long adjust_move_size_interval,
                           int max_states)
{

  const int n_node = n->n_node;
  double T = T_hi;
#ifdef USE_MPI
  int mpi_size = 0;
  int mpi_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  unsigned long exchange_acc = 0, exchange_tries = 0;
  if (mpi_size > 1)
    T = T_lo * pow(T_hi/T_lo, fraction(mpi_rank, mpi_size-1));
#endif
  if (e->n_experiment == 0)
    die("network_monte_carlo: no experiments given");
  if (e->n_node < 2)
    die("network_monte_carlo: must have at least 2 nodes");
  if (n_node != e->n_node)
    die("network_monte_carlo: network has %d nodes, but experiment set has %d nodes", n_node, e->n_node);
  trajectory_t trajectories = trajectories_new(e->n_experiment, max_states, n_node);
  double s = score(n,e,trajectories,HUGE_VAL,max_states), s_best = s;
#ifdef USE_MPI
  fprintf(out, "Process %d of %d\n", mpi_rank, mpi_size);
#endif

#ifdef _OPENMP
  omp_set_num_threads(n_thread);
#pragma omp parallel
  n_thread = omp_get_num_threads();
#endif

  fprintf(out, "Number of threads per process: %d\n", n_thread);
  fprintf(out, "Number of steps: %lu\n", n_cycles);
  fprintf(out, "Initial temperature: %g\n", T);
  fprintf(out, "Target score: %g\n", target_score);
  fprintf(out, "Exchange interval: %lu\n", exchange_interval);
  fprintf(out, "Adjust move size interval: %lu\n", adjust_move_size_interval);
  fprintf(out, "Max states: %d\n", max_states);
  fprintf(out, "Initial score: %g\n", s);
  fprintf(out, "\n");
  fflush(out);
  struct network best;
  network_init(&best, n->n_node, n->n_parent);
  copy_network(&best, n);
  struct network t0;
  network_init(&t0, n->n_node, n->n_parent);
  unsigned long parent_acc = 0, parent_tries = 0, parent_moves = 1;
  unsigned long outcome_acc = 0, outcome_tries = 0, outcome_moves = 1;
  unsigned long i;
  for (i = 1; i <= n_cycles; i++) {
#ifdef USE_MPI
    if (mpi_size == 1)
#endif
    /* if no MPI available or only one process, do simulated annealing */
      T = T_hi * pow(T_lo/T_hi, fraction(i-1, n_cycles-1));
    copy_network(&t0, n);
    unsigned long j;
    const int is_parent_move = (i % 2) && n->n_parent < n->n_node - 1;
    if (is_parent_move) { /* change a parent */
      parent_tries++;
      for (j = 0; j < parent_moves; j++) {
	const int k = random_int_inclusive(0, n_node - 1); /* which node to change */
      	n->parent[k][random_int_inclusive(0, n->n_parent - 1)] = 
          random_parent(k, n->n_node, n->n_parent, n->parent[k]);
      	qsort(&n->parent[k][0], n->n_parent, sizeof(int), intcmp);
      }
    } else { /* change outcomes */
      outcome_tries++;
      const int i_all_parents_unperturbed = (n->n_outcome - 1)/2;
      for (j = 0; j < outcome_moves; j++) {
	const int k = random_int_inclusive(0, n_node - 1);
      	/* change outcomes */
      	if (n->n_parent > 0) {
      	  int i_outcome;
          do
	    i_outcome = random_int_inclusive(0, n->n_outcome - 1);
      	  while (i_outcome == i_all_parents_unperturbed);
      	  n->outcome[k][i_outcome] = random_outcome();
      	}
      }
    }
    const double limit = s - T*log(uniform_random_from_0_to_1_exclusive());
    const double s_new = score(n, e, trajectories, limit, max_states);
    if (s_new < 0.9*LARGE_SCORE && s_new < limit) { 
      /* accepted */
      if (is_parent_move)
	      parent_acc++;
      else
	      outcome_acc++;
      s = s_new;
      if (s < s_best) {
	      s_best = s;
	      copy_network(&best, n);
      }
    } else {
      /* rejected */
      copy_network(n, &t0);
    }
#ifdef USE_MPI
    const int try_exchange = (mpi_size > 1) && (i % exchange_interval == 0);
    if (try_exchange) {
      if ((mpi_rank + i/exchange_interval) % 2) {
	if (mpi_rank < mpi_size - 1) {
	  /* Try an exchange with higher-T neighbor */
	  const int r1 = mpi_rank + 1;
	  const double T1 = receive_double(r1);
	  const double s1 = receive_double(r1);
	  const double ds = s1 - s;
	  if (ds < 0 || (T > 0 && uniform_random_from_0_to_1_exclusive() < exp(-ds*(1/T - 1/T1)))) {
	    send_int(r1, 1); /* accepted - tell neighbor */
	    exchange_networks(r1,n);
	    s = s1;
	    if (s < s_best) {
	      s_best = s;
	      copy_network(&best, n);
	    }
	  } else {
	    send_int(r1, 0); /* rejected - tell neighbor */
	  }
	}
      } else {
	if (mpi_rank > 0) {
	  /* lower-T neighbor is trying an exchange with us */
	  exchange_tries++;
	  const int r0 = mpi_rank - 1;
	  send_double(r0, T);
	  send_double(r0, s);
	  if (receive_int(r0)) { /* neighbor says the exchange was accepted */
	    exchange_networks(r0,n);
	    exchange_acc++;
	  }
	}
      }
    } /* try_exchange */
#endif
    int stop = 0;
#ifdef USE_MPI
    if (try_exchange) {
      if (mpi_rank == 0 && s_best <= target_score)
	stop = 1;
      bcast_int(&stop);
    }
#else
    if (s_best <= target_score)
      stop = 1;
#endif
    if (out && (stop || (n_write > 0 && n_cycles > n_write && i % (n_cycles/n_write) == 0))) {
      fprintf(out, "Ran %lu steps.\n", i);
      fprintf(out, "Temperature: %g\n", T);
      fprintf(out, "Parent move acceptances since last adjust: %lu\n", parent_acc);
      fprintf(out, "Parent move tries since last adjust: %lu\n", parent_tries);
      fprintf(out, "Fraction of parent moves accepted since last adjust: %g\n", fraction(parent_acc, parent_tries));
      fprintf(out, "Outcome move acceptances since last adjust: %lu\n", outcome_acc);
      fprintf(out, "Outcome move tries since last adjust: %lu\n", outcome_tries);
      fprintf(out, "Fraction of outcome moves accepted since last adjust: %g\n", fraction(outcome_acc, outcome_tries));
      fprintf(out, "Number of parent moves per cycle: %lu\n", parent_moves);
      fprintf(out, "Number of outcome moves per cycle: %lu\n", outcome_moves);
#ifdef USE_MPI
      fprintf(out, "Process %d of %d\n", mpi_rank, mpi_size);
      if (exchange_tries > 0)
	fprintf(out, "Fraction of exchanges with lower-T neighbor accepted: %g\n", fraction(exchange_acc, exchange_tries));
#endif
      fprintf(out, "Best score: %g\n", s_best);
      fprintf(out, "Best score (normalized): %g\n", s_best * scale_factor(e));
      fprintf(out, "\n");
      fflush(out);
    }
    if (stop)
      break;
    /* adjust number of moves */
    if (parent_tries == adjust_move_size_interval) { 
      const double f = fraction(parent_acc, parent_tries);
      if (f > 0.5 && parent_moves < n_node)
	parent_moves++;
      else if (f < 0.5 && parent_moves > 1)
	parent_moves--;
      parent_tries = 0;
      parent_acc = 0;
    }
    if (outcome_tries == adjust_move_size_interval) {
      const double f = fraction(outcome_acc, outcome_tries);
      if (f > 0.5)
	outcome_moves++;
      else if (f < 0.5 && outcome_moves > 1)
	outcome_moves--;
      outcome_tries = 0;
      outcome_acc = 0;
    }
  }
  copy_network(n, &best);
  network_delete(&best);
  network_delete(&t0);
  trajectories_delete(trajectories, e->n_experiment);

  return s_best;
}

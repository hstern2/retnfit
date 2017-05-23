#include "gn_Rwrap.h"
#include <mpi.h>
#include <assert.h>
#define MATHLIB_STANDALONE 1
#include <Rmath.h>
#include "gn.h"
#include "util.h"

static SEXP R_int(int n)
{
  SEXP R_n = NEW_INTEGER(1);
  *(INTEGER_POINTER(R_n)) = n;
  return R_n;
}

SEXP max_nodes_Rwrap() { 
  return R_int(MAX_NODES); 
}

SEXP max_experiments_Rwrap() { 
  return R_int(MAX_EXPERIMENTS);
}

SEXP network_monte_carlo_Rwrap(SEXP R_n,
			       SEXP R_n_node,
			       SEXP R_i_exp, 
			       SEXP R_i_node, 
			       SEXP R_outcome, 
			       SEXP R_val, 
			       SEXP R_is_perturbation, 
			       SEXP R_max_parents,
			       SEXP R_n_cycles,
			       SEXP R_n_write,
			       SEXP R_T_lo,
			       SEXP R_T_hi,
			       SEXP R_target_score,
			       SEXP R_outfile)
{

  const int n = INTEGER_VALUE(R_n);
  const int n_node = INTEGER_VALUE(R_n_node);
  const int *i_exp = INTEGER_POINTER(R_i_exp);
  const int *i_node = INTEGER_POINTER(R_i_node);
  const int *outcome = INTEGER_POINTER(R_outcome);

  const double *val = NUMERIC_POINTER(R_val);

  const int max_parents = INTEGER_VALUE(R_max_parents);

  const int *is_perturbation = INTEGER_POINTER(R_is_perturbation);
  const unsigned long n_cycles = NUMERIC_VALUE(R_n_cycles);
  const int n_write = INTEGER_VALUE(R_n_write);
  const double T_lo = NUMERIC_VALUE(R_T_lo);
  const double T_hi = NUMERIC_VALUE(R_T_hi);
  const double target_score = NUMERIC_VALUE(R_target_score);
  const char *outfile = STRING_VALUE(R_outfile);

  struct experiment_set e;
  experiment_set_init(&e, n, i_exp, i_node, outcome, val, is_perturbation);

  assert(e.n_node == n_node);
  assert(T_lo > 0);
  assert(T_hi > 0);

  struct network net;
  network_init(&net, e.n_node, max_parents);

  char fname[1024];
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  sprintf(fname, "%s.%d", outfile, rank);
  FILE *f = safe_fopen(fname, "w");

  SEXP R_unnormalized_score = PROTECT(NEW_NUMERIC(1));
  double *unnormalized_score = NUMERIC_POINTER(R_unnormalized_score);
  
  *unnormalized_score = network_monte_carlo(&net,
					    &e,
					    n_cycles,
					    n_write,
					    T_lo,
					    T_hi,
					    f, target_score);
  
  SEXP R_normalized_score = PROTECT(NEW_NUMERIC(1));
  double *normalized_score = NUMERIC_POINTER(R_normalized_score);
  *normalized_score = *unnormalized_score * scale_factor(&e);
  
  network_write_response_from_experiment_set(f, &net, &e);
  fprintf(f, "\n");
  fprintf(f, "unnormalized score: %g\n", *unnormalized_score);
  fprintf(f, "lowest possible unnormalized score: %g\n", lowest_possible_score(&e));
  fprintf(f, "difference: %g\n", *unnormalized_score - lowest_possible_score(&e));
  fprintf(f, "normalized score: %g\n", *normalized_score);
  fprintf(f, "\n");
  fprintf(f, "network:\n");
  network_write(f, &net);
  fclose(f);

  SEXP R_parents = PROTECT(NEW_INTEGER(n_node*max_parents));
  int *parents = INTEGER_POINTER(R_parents);

  int i, j;
  for (i = 0; i < n_node; i++)
    for (j = 0; j < max_parents; j++)
      parents[j*n_node+i] = net.parent[i][j];

  SEXP R_outcomes = PROTECT(NEW_INTEGER(n_node*three_to_the(max_parents)));
  int *outcomes = INTEGER_POINTER(R_outcomes);

  for (i = 0; i < n_node; i++) {
    int k;
    for (k = 0; k < net.n_outcome; k++)
      outcomes[k*n_node+i] = net.outcome[i][k];
  }

  SEXP R_trajectories = PROTECT(NEW_LIST(e.n_experiment));
  struct trajectory t;
  for (i = 0; i < e.n_experiment; i++) {
    network_advance_until_repetition(&net, &e.experiment[i], &t);
    const int n_rep = t.repetition_end + 1;
    SEXP R_traj = PROTECT(allocMatrix(INTSXP, n_rep, n_node));
    int j, k;
    for (j = 0; j < n_rep; j++)
      for (k = 0; k < n_node; k++)
	INTEGER_POINTER(R_traj)[k*n_rep+j] = t.state[j][k];
    SET_VECTOR_ELT(R_trajectories, i, R_traj);
  }

  SEXP results = PROTECT(NEW_LIST(5));
  SET_VECTOR_ELT(results, 0, R_unnormalized_score);
  SET_VECTOR_ELT(results, 1, R_normalized_score);
  SET_VECTOR_ELT(results, 2, R_parents);
  SET_VECTOR_ELT(results, 3, R_outcomes);
  SET_VECTOR_ELT(results, 4, R_trajectories);

  UNPROTECT(e.n_experiment + 6);

  network_delete(&net);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  return(results);
}

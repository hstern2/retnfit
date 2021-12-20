#include "gn_Rwrap.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <assert.h>
#define MATHLIB_STANDALONE 1
#include <Rmath.h>
#include "gn.h"
#include "util.h"

SEXP is_MPI_available()
{
#ifdef USE_MPI
  return Rf_ScalarLogical(1);
#else
  return Rf_ScalarLogical(0);
#endif
}

SEXP max_nodes_Rwrap() { 
  return Rf_ScalarInteger(MAX_NODES); 
}

static int SEXP_to_int(SEXP x) { return Rf_asInteger(x); }
static int *SEXP_to_intp(SEXP x) { return INTEGER(x); }
static double SEXP_to_double(SEXP x) { return Rf_asReal(x); }
static double *SEXP_to_doublep(SEXP x) { return REAL(x); }
static const char *SEXP_to_const_charp(SEXP x) { return CHAR(Rf_asChar(x)); }

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
			       SEXP R_outfile,
                               SEXP R_n_thread,
                               SEXP R_init_parents,
                               SEXP R_init_outcomes,
                               SEXP R_exchange_interval,
                               SEXP R_adjust_move_size_interval,
                               SEXP R_max_states)
{

  const int n = SEXP_to_int(R_n);
  const int n_node = SEXP_to_int(R_n_node);
  const int *i_exp = SEXP_to_intp(R_i_exp);
  const int *i_node = SEXP_to_intp(R_i_node);
  const int *outcome = SEXP_to_intp(R_outcome);

  const double *val = SEXP_to_doublep(R_val);

  const int max_parents = SEXP_to_int(R_max_parents);

  const int *is_perturbation = SEXP_to_intp(R_is_perturbation);
  const unsigned long n_cycles = SEXP_to_double(R_n_cycles);
  const int n_write = SEXP_to_int(R_n_write);
  const double T_lo = SEXP_to_double(R_T_lo);
  const double T_hi = SEXP_to_double(R_T_hi);
  const double target_score = SEXP_to_double(R_target_score);
  const int exchange_interval = SEXP_to_int(R_exchange_interval);
  const int adjust_move_size_interval = SEXP_to_int(R_adjust_move_size_interval);
  const int max_states = SEXP_to_int(R_max_states);
  const char *outfile = SEXP_to_const_charp(R_outfile);
  const int n_thread = SEXP_to_int(R_n_thread);

  struct experiment_set e;
  experiment_set_init(&e, n, i_exp, i_node, outcome, val, is_perturbation);

  assert(e.n_node == n_node);
  assert(T_lo > 0);
  assert(T_hi > 0);

  struct network net;
  network_init(&net, e.n_node, max_parents);
  if (!Rf_isNull(R_init_parents))
    network_read_parents_from_intp(&net, SEXP_to_intp(R_init_parents));
  else
    network_randomize_parents(&net);
  if (!Rf_isNull(R_init_outcomes))
    network_read_outcomes_from_intp(&net, SEXP_to_intp(R_init_outcomes));
  else
    network_set_outcomes_to_null(&net);

  char fname[1024];

#ifdef USE_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  sprintf(fname, "%s.%d", outfile, rank);
#else
  sprintf(fname, "%s", outfile);
#endif

  FILE *f = safe_fopen(fname, "w");

  SEXP R_unnormalized_score = PROTECT(NEW_NUMERIC(1));
  double *unnormalized_score = SEXP_to_doublep(R_unnormalized_score);
  
  *unnormalized_score = network_monte_carlo(&net,
					    &e,
					    n_cycles,
					    n_write,
					    T_lo,
					    T_hi,
					    f,
              n_thread,
              target_score,
              exchange_interval,
              adjust_move_size_interval,
              max_states);
  
  SEXP R_normalized_score = PROTECT(NEW_NUMERIC(1));
  double *normalized_score = SEXP_to_doublep(R_normalized_score);
  *normalized_score = *unnormalized_score * scale_factor(&e);
  
  network_write_response_from_experiment_set(f, &net, &e, max_states);
  fprintf(f, "\n");
  fprintf(f, "unnormalized score: %g\n", *unnormalized_score);
  fprintf(f, "lowest possible unnormalized score: %g\n", lowest_possible_score(&e));
  fprintf(f, "difference: %g\n", *unnormalized_score - lowest_possible_score(&e));
  fprintf(f, "normalized score: %g\n", *normalized_score);
  fprintf(f, "\n");
  fprintf(f, "network:\n");
  network_write_to_file(f, &net);
  fclose(f);

  SEXP R_parents = PROTECT(NEW_INTEGER(n_node*max_parents));
  SEXP R_outcomes = PROTECT(NEW_INTEGER(n_node*three_to_the(max_parents)));
  network_write_to_intp(&net, SEXP_to_intp(R_parents), SEXP_to_intp(R_outcomes));

  SEXP R_trajectories = PROTECT(NEW_LIST(e.n_experiment));
  trajectory_t t = trajectories_new(e.n_experiment, max_states, n_node);
  int i;
  for (i = 0; i < e.n_experiment; i++) {
    network_advance_until_repetition(&net, &e.experiment[i], &t[i], max_states);
    const int n_rep = t[i].repetition_end + 1;
    SEXP R_traj = PROTECT(allocMatrix(INTSXP, n_rep, n_node));
    int j, k;
    for (j = 0; j < n_rep; j++)
      for (k = 0; k < n_node; k++)
	      SEXP_to_intp(R_traj)[k*n_rep+j] = t[i].state[j][k];
    SET_VECTOR_ELT(R_trajectories, i, R_traj);
  }
  trajectories_delete(t, e.n_experiment);
  SEXP results = PROTECT(NEW_LIST(5));
  SET_VECTOR_ELT(results, 0, R_unnormalized_score);
  SET_VECTOR_ELT(results, 1, R_normalized_score);
  SET_VECTOR_ELT(results, 2, R_parents);
  SET_VECTOR_ELT(results, 3, R_outcomes);
  SET_VECTOR_ELT(results, 4, R_trajectories);

  UNPROTECT(e.n_experiment + 6);

  network_delete(&net);
  experiment_set_delete(&e);
  
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif  

  return(results);
}

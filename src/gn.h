#ifndef GN_H
#define GN_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NODES 10000
#define MAX_EXPERIMENTS 10000
#define MAX_STATES_LIMIT 200

  typedef struct network 
  {
    int n_node, n_parent, n_outcome;
    int **parent; /* n_node x n_parent */
    int **outcome; /* n_node x n_outcome */
  } *network_t;
 
  void network_init(network_t, int n_node, int max_parents);
  void network_randomize_parents(network_t);
  void network_randomize_outcomes(network_t);
  void network_set_outcomes_to_null(network_t);
  void network_write_to_file(FILE *, const network_t);
  void network_write_to_intp(const network_t, int *parent, int *outcome);
  void network_read_parents_from_intp(network_t, const int *parents);
  void network_read_outcomes_from_intp(network_t, const int *outcomes);
  void network_delete(network_t);

  typedef struct experiment
  {
    double score[MAX_NODES][3];
    int n_perturbed, perturbed[MAX_NODES];
  } *experiment_t;
  
  typedef struct experiment_set
  {
    int n_experiment, n_node;
    struct experiment experiment[MAX_EXPERIMENTS];
  } *experiment_set_t;

  typedef struct trajectory
  {
    int n_node;
    int repetition_start, repetition_end;
    int is_persistent[MAX_NODES];
    int state[MAX_STATES_LIMIT][MAX_NODES];
    int steady_state[MAX_NODES];
  } *trajectory_t;

  double scale_factor(const experiment_set_t eset);
  
  void experiment_set_read_from_csv(FILE *, experiment_set_t);
  void experiment_set_write_as_csv(FILE *, const experiment_set_t);
  void experiment_set_init(experiment_set_t,
			   int n,
			   const int *i_exp, 
			   const int *i_node, 
			   const int *outcome,
			   const double *val,
			   const int *is_perturbation);

  void network_write_response_from_experiment_set(FILE *, network_t, const experiment_set_t, int max_states);
  void network_write_response_as_target_data(FILE *, network_t, const experiment_set_t, int max_states);
  void network_advance_until_repetition(const network_t, const experiment_t, trajectory_t t, int max_states);
  
  double lowest_possible_score(const experiment_set_t);

  double network_monte_carlo(network_t, 
			     const experiment_set_t, 
			     unsigned long n_cycles,
			     unsigned long n_write,
			     double T_lo,
			     double T_hi,
			     FILE *out,
                             int n_thread,
			     double target_score,
                             unsigned long exchange_interval,
                             unsigned long adjust_move_size_interval,
                             int max_states);

  unsigned three_to_the(unsigned n);

#ifdef __cplusplus
}
#endif

#endif /* GN_H */

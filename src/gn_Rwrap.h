#ifndef GN_RWRAP_H
#define GN_RWRAP_H

#include <R.h>
#include <Rdefines.h>

SEXP max_nodes_Rwrap();

SEXP max_experiments_Rwrap();

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
			       SEXP R_outfile);


#endif /* GN_RWRAP_H */

#include <R_ext/Rdynload.h>
#include "gn_Rwrap.h"

#define CALLDEF(name, n) { #name, (DL_FUNC) &name, n }

const static R_CallMethodDef R_CallDef[] = {
  CALLDEF(is_MPI_available, 0),
  CALLDEF(max_nodes_Rwrap, 0),
  CALLDEF(max_experiments_Rwrap, 0),
  CALLDEF(max_states_limit_Rwrap, 0),
  CALLDEF(network_monte_carlo_Rwrap, 19),
  {0, 0, 0}
};

void R_init_retnfit(DllInfo *dll)
{
  R_registerRoutines(dll, 0, R_CallDef, 0, 0);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}

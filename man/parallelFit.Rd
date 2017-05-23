\name{parallelFit}
\alias{parallelFit}
\title{Fit ternary network models using parallel tempering}
\description{Fit ternary network models using parallel tempering}
\usage{
    parallelFit(experiment_set, 
        max_parents, 
        n_cycles, 
        n_write, 
        T_lo, 
        T_hi,
        target_score,
        n_proc,
        logfile,
        seed)
}
\arguments{
    \item{experiment_set}{data frame containing five columns: 
        i_exp (experiment index), i_node (node index), outcome (-1/0/1), 
        value (cost for that outcome), is_perturbation (0 or 1)}
    \item{max_parents}{maximum number of parents allowed for each node}
    \item{n_cycles}{maximum number of Monte Carlo cycles}
    \item{n_write}{number of times to write output during the run}
    \item{T_lo}{T for lowest-temperature replica}
    \item{T_hi}{T for highest-temperature replica}
    \item{target_score}{target_score - run will terminate 
                        if this is reached}
    \item{n_proc}{number of replicas}
    \item{logfile}{filename for log file}
    \item{seed}{seed for random number generator}
}
\value{
    return value is a list with an element for each replica.  Each
    element is itself a list of the best unnormalized score, normalized
    score (unnormalized score divided by product of number of nodes and
    number of experiments), list of parents for each node, and array
    describing transition rule, giving the outcome of a node for each
    configuration of parent node.
}
\author{Harry A. Stern and Matthew N. McCall}
\examples{
??retnfit
\donttest{
# need to be run with mpirun
results <- parallelFit(read.csv('sampledata.csv'),
                        max_parents=4, 
                        n_cycles=10000,
                        n_write=10,
                        T_lo=0.001,
                        T_hi=1.0,
                        target_score=0,
                        n_proc=12,
                        logfile='a.log',
                        seed=525108)

lowest_temp_results <- results[[1]]

print('Unnormalized score:')
print(lowest_temp_results$unnormalized_score)

print('Normalized score:')
print(lowest_temp_results$normalized_score)

print('Parents:')
print(lowest_temp_results$parents)

print('Outcomes:')
print(lowest_temp_results$outcomes)
}
}

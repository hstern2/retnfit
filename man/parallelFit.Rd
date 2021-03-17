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
        n_thread,
        init_parents,
        init_outcomes,
        exchange_interval,
        adjust_move_size_interval,
        max_states)
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
    \item{n_thread}{number of openMP threads to run per process; default=1}
    \item{init_parents}{initial parents; randomized if null}
    \item{init_outcomes}{initial outcomes; set to '.' if null}
    \item{exchange_interval}{steps between exchanges; default=1000}
    \item{adjust_move_size_interval}{steps between move size adjustment;
            default=7001}
    \item{max_states}{max states to propagate when testing for repetition;
            default=10}
}
\value{
    The return value is a list with an element for each replica.  Each
    element is itself a list of the best unnormalized score, normalized
    score (unnormalized score divided by product of number of nodes and
    number of experiments), list of parents for each node, and array
    describing the transition rule, giving the outcome of a node for each
    possible configuration of parent nodes.
}
\author{Harry A. Stern and Matthew N. McCall}
\examples{

i_exp <- as.integer(c(0,0,0, 0,0,0, 0,0,0, 0,0,0,
                        1,1,1, 1,1,1, 1,1,1, 1,1,1,
                        2,2,2, 2,2,2, 2,2,2, 2,2,2,
                        3,3,3, 3,3,3, 3,3,3, 3,3,3,
                        4,4,4, 4,4,4, 4,4,4, 4,4,4,
                        5,5,5, 5,5,5, 5,5,5, 5,5,5,
                        6,6,6, 6,6,6, 6,6,6, 6,6,6,
                        7,7,7, 7,7,7, 7,7,7, 7,7,7))

i_node <- as.integer(c(0,0,0, 1,1,1, 2,2,2, 3,3,3,
                        0,0,0, 1,1,1, 2,2,2, 3,3,3,
                        0,0,0, 1,1,1, 2,2,2, 3,3,3,
                        0,0,0, 1,1,1, 2,2,2, 3,3,3,
                        0,0,0, 1,1,1, 2,2,2, 3,3,3,
                        0,0,0, 1,1,1, 2,2,2, 3,3,3,
                        0,0,0, 1,1,1, 2,2,2, 3,3,3,
                        0,0,0, 1,1,1, 2,2,2, 3,3,3))

outcome <- as.integer(c(-1,0,1, -1,0,1, -1,0,1, -1,0,1,
                        -1,0,1, -1,0,1, -1,0,1, -1,0,1,
                        -1,0,1, -1,0,1, -1,0,1, -1,0,1,
                        -1,0,1, -1,0,1, -1,0,1, -1,0,1,
                        -1,0,1, -1,0,1, -1,0,1, -1,0,1,
                        -1,0,1, -1,0,1, -1,0,1, -1,0,1,
                        -1,0,1, -1,0,1, -1,0,1, -1,0,1,
                        -1,0,1, -1,0,1, -1,0,1, -1,0,1))

value <- c(0,1,2, 0,1,2, 0,1,2, 0,1,2,
            2,1,0, 0,1,2, 0,1,2, 0,1,2,
            2,1,0, 2,1,0, 0,1,2, 0,1,2,
            2,1,0, 2,1,0, 2,1,0, 0,1,2,
            2,1,0, 2,1,0, 2,1,0, 2,1,0,
            0,1,2, 2,1,0, 2,1,0, 2,1,0,
            0,1,2, 0,1,2, 2,1,0, 2,1,0,
            0,1,2, 0,1,2, 0,1,2, 2,1,0)

is_perturbation <- 
c(TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, FALSE,FALSE,FALSE, FALSE,FALSE,FALSE,
    FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE, FALSE,FALSE,FALSE,
    FALSE,FALSE,FALSE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
    FALSE,FALSE,FALSE,  FALSE,FALSE,FALSE, FALSE,FALSE,FALSE, TRUE,TRUE,TRUE,
    TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, FALSE,FALSE,FALSE, FALSE,FALSE,FALSE,
    FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE, FALSE,FALSE,FALSE,
    FALSE,FALSE,FALSE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
    FALSE,FALSE,FALSE,  FALSE,FALSE,FALSE, FALSE,FALSE,FALSE, TRUE,TRUE,TRUE)

indata <- data.frame(i_exp,i_node,outcome,value,is_perturbation)

results <- parallelFit(indata,
                        max_parents=1,
                        n_cycles=100000,
                        n_write=10,
                        T_lo=0.001,
                        T_hi=2.0,
                        target_score=0,
                        n_proc=1,
                        logfile='try.log')

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

callNetworkMonteCarloRwrap <- function(n,
    n_node,
    experiment_set,
    max_parents,
    n_cycles,
    n_write,
    T_lo,
    T_hi,
    target_score,
    logfile,
    seed)
{
    set.seed(seed + mpi.comm.rank())

    results <- .Call("network_monte_carlo_Rwrap",
        n,
        n_node,
        experiment_set$i_exp,
        experiment_set$i_node,
        experiment_set$outcome,
        experiment_set$val,
        experiment_set$is_perturbation,
        max_parents,
        n_cycles,
        n_write,
        T_lo,
        T_hi,
        target_score,
        logfile)

    names(results) <- c("unnormalized_score",
        "normalized_score",
        "parents",
        "outcomes",
        "trajectories")

    dim(results$parents) <- c(n_node, max_parents)
    dim(results$outcomes) <- c(n_node, rep(3,max_parents))

    results
}

parallelFit <- function(experiment_set,
    max_parents,
    n_cycles,
    n_write,
    T_lo,
    T_hi,
    target_score,
    n_proc,
    logfile,
    seed)
{
    if (n_proc < 2) stop("n_proc must be at least 2")
    if (T_lo <= 0) stop("T_lo must be greater than 0")
    if (T_hi <= T_lo) stop("T_hi must be greater than T_lo")
    if (max_parents < 1) stop("max_parents must be at least 1")

    n <- nrow(experiment_set)
    n_node <- max(experiment_set$i_node)+1

    cap = .Call("max_nodes_Rwrap")
    if (n_node > cap) stop(paste("number of nodes must be at most", cap))

    cap = .Call("max_experiments_Rwrap")
    if (max(experiment_set$i_exp)+1 > cap) {
        stop(paste("number of experiments must be at most", cap))
    }

    cl <- makeCluster(n_proc, type="MPI")

    results <- clusterCall(cl, 
        callNetworkMonteCarloRwrap,
        n,
        n_node,
        experiment_set,
        max_parents, 
        n_cycles, 
        n_write, 
        T_lo,
        T_hi,
        target_score, 
        logfile,
        seed)

    stopCluster(cl)
    mpi.finalize()

    results

}

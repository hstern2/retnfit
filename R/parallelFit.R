callNetworkMonteCarloRwrap <- function(i, n,
    n_node,
    i_exp,
    i_node,
    outcome,
    value,
    is_perturbation,
    max_parents,
    n_cycles,
    n_write,
    T_lo,
    T_hi,
    target_score,
    logfile,
    seed,
    n_intermediates)
{

    set.seed(seed + i)


    results <- .Call("network_monte_carlo_Rwrap",
        n,
        n_node,
        i_exp,
        i_node,
        outcome,
        value,
        is_perturbation,
        max_parents,
        n_cycles,
        n_write,
        T_lo,
        T_hi,
        target_score,
        logfile,
        n_intermediates)

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
    seed,
    n_intermediates)
{
    i_exp <- experiment_set$i_exp
    i_node <- experiment_set$i_node
    outcome <- experiment_set$outcome
    value <- experiment_set$value
    is_perturbation <- experiment_set$is_perturbation
    if (is.logical(is_perturbation)) {
        is_perturbation <- as.integer(is_perturbation)
    }

    stopifnot(is.integer(i_exp))
    stopifnot(is.integer(i_node))
    stopifnot(is.integer(outcome))
    stopifnot(is.numeric(value))
    stopifnot(is.integer(is_perturbation))
    stopifnot(T_lo > 0)
    stopifnot(T_hi > T_lo)
    stopifnot(max_parents >= 1)
    stopifnot(n_intermediates >= 1)

    n <- nrow(experiment_set)
    n_node <- max(i_node)+1

    max_nodes = .Call("max_nodes_Rwrap")
    stopifnot(n_node <= max_nodes)

    max_experiments = .Call("max_experiments_Rwrap")
    number_of_experiments = max(i_exp)+1
    stopifnot(number_of_experiments <= max_experiments)

    if (.Call("is_MPI_available") &&
        requireNamespace("BiocParallel") && 
        requireNamespace("Rmpi") && 
        requireNamespace("snow")) {

        stopifnot(n_proc >= 2)
        using_MPI = TRUE
        bp_param <- BiocParallel::SnowParam(n_proc, type="MPI")

    } else {

        using_MPI = FALSE   
        bp_param <- BiocParallel::SerialParam()

    }
        
    results <- BiocParallel::bplapply(seq_len(n_proc), 
        callNetworkMonteCarloRwrap,
        BPPARAM=bp_param,
        n,
        n_node,
        i_exp,
        i_node,
        outcome,
        value,
        is_perturbation,
        max_parents, 
        n_cycles, 
        n_write, 
        T_lo,
        T_hi,
        target_score, 
        logfile,
        seed,
        n_intermediates)

    if (using_MPI) {
        Rmpi::mpi.finalize()
    }

    results

}

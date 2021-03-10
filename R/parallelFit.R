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
    init_parents,
    init_outcomes,
    exchange_interval,
    adjust_move_size_interval,
    max_states)
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
        init_parents,
        init_outcomes,
        exchange_interval,
        adjust_move_size_interval,
        max_states)

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
    seed=NULL,
    init_parents=NULL,
    init_outcomes=NULL,
    exchange_interval=1000,
    adjust_move_size_interval=7001,
    max_states=10)
{
    i_exp <- experiment_set$i_exp
    i_node <- experiment_set$i_node
    outcome <- experiment_set$outcome
    value <- experiment_set$value
    is_perturbation <- experiment_set$is_perturbation
    
    if (is.logical(is_perturbation)) 
        is_perturbation <- as.integer(is_perturbation)

    stopifnot(is.integer(i_exp))
    stopifnot(is.integer(i_node))
    stopifnot(is.integer(outcome))
    stopifnot(is.numeric(value))
    stopifnot(is.integer(is_perturbation))
    stopifnot(T_lo > 0)
    stopifnot(T_hi > T_lo)
    stopifnot(max_parents >= 1)
    stopifnot(exchange_interval >= 1)
    stopifnot(adjust_move_size_interval >= 1)
    stopifnot(max_states >= 1)
    stopifnot(is.null(init_parents) || is.integer(init_parents));
    stopifnot(is.null(init_outcomes) || is.integer(init_outcomes));

    if (is.null(seed)) 
        seed <- as.numeric(Sys.time())

    n <- nrow(experiment_set)
    n_node <- max(i_node)+1
    max_nodes = .Call("max_nodes_Rwrap")
    stopifnot(n_node <= max_nodes)

    max_experiments = .Call("max_experiments_Rwrap")
    number_of_experiments = max(i_exp)+1
    stopifnot(number_of_experiments <= max_experiments)
 
    max_states_limit = .Call("max_states_limit_Rwrap")
    stopifnot(max_states <= max_states_limit)

    if (.Call("is_MPI_available") &&
        requireNamespace("BiocParallel") && 
        requireNamespace("Rmpi") && 
        requireNamespace("snow")) {

        using_MPI = TRUE
        bp_param <- BiocParallel::SnowParam(n_proc, type="MPI")

    } else {

        if (n_proc > 1) stop("Rmpi not available, but n_proc > 1")
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
        init_parents,
        init_outcomes,
        exchange_interval,
        adjust_move_size_interval,
        max_states)

    if (using_MPI) {
        Rmpi::mpi.comm.disconnect()
        Rmpi::mpi.finalize()
    }

    results

}

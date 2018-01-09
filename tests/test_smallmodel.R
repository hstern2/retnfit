library(testthat)
library(retnfit) 

smallmodel_score <- function() {

    i_exp <- as.integer(c(0,0,0,0,0,0,0,0,0,1,1,1,
                            1,1,1,1,1,1,1,1,1,1,1,
                            1,1,1,1))
    i_node <- as.integer(c(0,0,0,1,1,1,2,2,2,0,0,0,
                            1,1,1,2,2,2,0,0,0,1,1,1,
                            2,2,2))
    outcome <- as.integer(c(-1,0,1,-1,0,1,-1,0,1,-1,0,1,
                            -1,0,1,-1,0,1,-1,0,1,-1,0,1,
                            -1,0,1))
    value <- c(2.0,1.0,0.0,2.0,1.0,0.0,2.0,1.0,0.0,
                2.0,1.0,0.0,2.0,1.0,0.0,2.0,1.0,0.0,
                2.0,1.0,0.0,2.0,1.0,0.0,2.0,1.0,0.0)
    is_perturbation <- as.integer(c(0,0,1,0,0,0,0,0,0,0,0,
                                    0,0,0,1,0,0,0,0,0,0,0,
                                    0,0,0,0,1))
    indata <- data.frame(i_exp,i_node,outcome,value,is_perturbation)
    results <- parallelFit(indata,
                            max_parents=1,
                            n_cycles=10000,
                            n_write=10,
                            T_lo=0.001,
                            T_hi=1.0,
                            target_score=0,
                            n_proc=3,
                            logfile='test.log',
                            seed=12345)
    
    lowest_temp_results <- results[[1]]

    lowest_temp_results$unnormalized_score

}
   
test_that("smallmodel", {
    expect_true(smallmodel_score() < 3)
})
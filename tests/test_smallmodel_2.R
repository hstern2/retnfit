library(testthat)
library(retnfit) 

smallmodel_2_score <- function() {

    library(retnfit)

    i_exp <- as.integer(c(0,0,0, 0,0,0, 0,0,0,
                        1,1,1, 1,1,1, 1,1,1,
                        2,2,2, 2,2,2, 2,2,2,
                        3,3,3, 3,3,3, 3,3,3,
                        4,4,4, 4,4,4, 4,4,4,
                        5,5,5, 5,5,5, 5,5,5,
                        6,6,6, 6,6,6, 6,6,6,
                        7,7,7, 7,7,7, 7,7,7,
                        8,8,8, 8,8,8, 8,8,8,
                        9,9,9, 9,9,9, 9,9,9,
                        10,10,10, 10,10,10, 10,10,10,
                        11,11,11, 11,11,11, 11,11,11,
                        12,12,12, 12,12,12, 12,12,12,
                        13,13,13, 13,13,13, 13,13,13,
                        14,14,14, 14,14,14, 14,14,14,
                        15,15,15, 15,15,15, 15,15,15,
                        16,16,16, 16,16,16, 16,16,16,
                        17,17,17, 17,17,17, 17,17,17))

    i_node <- as.integer(c(0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2,
                        0,0,0, 1,1,1, 2,2,2))

    outcome <- as.integer(c(-1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1,
                            -1,0,1, -1,0,1, -1,0,1))

    value <- c(2,1,0, 1,0,1, 2,1,0,
            1,0,1, 2,1,0, 0,1,2,
            1,0,1, 1,0,1, 2,1,0,
            0,1,2, 1,0,1, 0,1,2,
            1,0,1, 0,1,2, 2,1,0,
            1,0,1, 1,0,1, 0,1,2,
            2,1,0, 1,0,1, 2,1,0,
            0,1,2, 1,0,1, 0,1,2,
            1,0,1, 2,1,0, 0,1,2,
            1,0,1, 0,1,2, 2,1,0,
            2,1,0, 2,1,0, 1,0,1,
            0,1,2, 0,1,2, 1,0,1,
            2,1,0, 0,1,2, 2,1,0,
            0,1,2, 2,1,0, 0,1,2,
            2,1,0, 1,0,1, 0,1,2,
            0,1,2, 1,0,1, 2,1,0,
            1,0,1, 2,1,0, 2,1,0,
            1,0,1, 0,1,2, 0,1,2)

    is_perturbation <- c(TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, FALSE,FALSE,FALSE,
                        FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
                        FALSE,FALSE,FALSE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE,
                        TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, FALSE,FALSE,FALSE,
                        FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
                        FALSE,FALSE,FALSE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE,
                        TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE,
                        TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE,
                        FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, TRUE,TRUE,TRUE,
                        FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, TRUE,TRUE,TRUE,
                        TRUE,TRUE,TRUE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
                        TRUE,TRUE,TRUE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
                        TRUE,TRUE,TRUE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
                        TRUE,TRUE,TRUE,  TRUE,TRUE,TRUE, FALSE,FALSE,FALSE,
                        TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE,
                        TRUE,TRUE,TRUE,  FALSE,FALSE,FALSE, TRUE,TRUE,TRUE,
                        FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, TRUE,TRUE,TRUE,
                        FALSE,FALSE,FALSE,  TRUE,TRUE,TRUE, TRUE,TRUE,TRUE)

    indata <- data.frame(i_exp,i_node,outcome,value,is_perturbation)

    results <- parallelFit(indata,
                            max_parents=2,
                            n_cycles=100000,
                            n_write=10,
                            T_lo=0.001,
                            T_hi=2.0,
                            target_score=0,
                            n_proc=1,
                            logfile='try2.log',
                            seed=1234)
    
    lowest_temp_results <- results[[1]]

    lowest_temp_results$unnormalized_score

}
   
test_that("smallmodel_2", {
    expect_true(smallmodel_2_score() == 0)
})

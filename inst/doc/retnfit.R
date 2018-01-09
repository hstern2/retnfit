### R code from vignette source 'retnfit.Rnw'

###################################################
### code chunk number 1: retnfit.Rnw:57-100
###################################################
library('retnfit')

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

print('Unnormalized score:')
print(lowest_temp_results$unnormalized_score)

print('Normalized score:')
print(lowest_temp_results$normalized_score)

print('Parents:')
print(lowest_temp_results$parents)

print('Outcomes:')
print(lowest_temp_results$outcomes)




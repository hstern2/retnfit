### R code from vignette source 'retnfit.Rnw'

###################################################
### code chunk number 1: retnfit.Rnw:57-83
###################################################
library('retnfit')
results <- parallelFit(read.csv('sampledata.csv'),
                        max_parents=4,
                        n_cycles=100000,
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




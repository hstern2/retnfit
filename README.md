# retnfit

## Fitting ternary network models of gene regulatory networks by replica exchange Monte Carlo

#### references:

* H.R. McMurray, A. Ambeskovic, L.A. Newman, J. Aldersley, V. Balakrishnan, B. Smith, H. A. Stern, H. Land, and M. N. McCall, "Gene network modeling via TopNet reveals functional dependencies between diverse tumor-critical mediator genes," *Cell Reports* **37**, 110136 (2021) [doi.org/10.1016/j.celrep.2021.110136](https://doi.org/10.1016/j.celrep.2021.110136)


To install parallel (openMP and MPI) version, run 
```
autoconf
```
in retnfit directory before
```
R CMD build retnfit
```

For MPI version, run
```
R CMD INSTALL --clean retnfit_0.99.17.tar.gz --configure-args='--with-mpi=/path/to/mpi-1.x  --with-Rmpi-type=OPENMPI'
```

For CUDA version
> - Update the ```driver.c``` file with the experiment data
> - Go to ```src/``` and run ```make```
> - This generates an executable ```retnfit_cuda```
> - To run : ```./retnfit_cuda```
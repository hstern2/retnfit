# retnfit

## Fitting ternary network models of gene regulatory networks by replica exchange Monte Carlo

<img width="559" alt="image" src="https://user-images.githubusercontent.com/19351218/206610097-480272b7-6fa9-4c9c-847e-6dcbd9f90b93.png">

#### references:

* H. R. McMurray, A. Ambeskovic, L.A. Newman, J. Aldersley, V. Balakrishnan, B. Smith, H. A. Stern, H. Land, and M. N. McCall, "Gene network modeling via TopNet reveals functional dependencies between diverse tumor-critical mediator genes," *Cell Reports* **37**, 110136 (2021) [doi.org/10.1016/j.celrep.2021.110136](https://doi.org/10.1016/j.celrep.2021.110136)

* H. R. McMurray, H. A. Stern, A. Ambeskovic, H. Land, and M. N. McCall, “Protocol to use TopNet for gene regulatory network modeling using gene expression data from perturbation experiments,” *STAR protocols* **3**, 101737 (2022) [doi.org/10.1016/j.xpro.2022.101737](https://doi.org/10.1016/j.xpro.2022.101737)

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
> - Go to ```cuda/``` and run ```make```
> - This generates an executable ```driver```
> - To run : ```./driver```

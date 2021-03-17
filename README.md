# retnfit

Fitting ternary network models of gene regulatory networks by replica exchange Monte Carlo

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

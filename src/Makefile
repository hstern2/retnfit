# Path to NVCC compiler, since its likely not on $PATH
NVCC=/usr/local/cuda/bin/nvcc

# Xcompiler flag passes comma separated list of flag to the
# GCC/G++ compiler. Optimization flags can be pass directly
# to NVCC. O3 enables vectorization on for-loops.
NVCC_FLAGS=-arch=sm_35 -rdc=true -std=c++11  -Xcompiler -Wall -g
FILES=driver.c gn.cu array.c util.c
INCLUDE=/usr/include/R
all:
	$(NVCC) $(NVCC_FLAGS) -I$(INCLUDE) $(FILES) -lRmath -o retnfit_cuda

clean:
	rm retnfit_cuda

c_driver:
	
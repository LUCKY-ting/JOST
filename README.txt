
MATLAB Code for the paper: Joint optimization of scoring and thresholding models for online multi-label classification £¨to be appear in Pattern Recognition journal£©

## guideline for how to use this code

1. "run_FALT.m" is a demo for the linear FALT algorithm proposed in this paper, which relies on  "FALT_train_sparse.c" to run.
You should first input "mex -largeArrayDims FALT_train_sparse.c" in the command window of matlab in order to build a executable mex-file.
Then run "run_FALT.m".

2. "run_kernel_FALT.m" is a demo for the kernelized FALT algorithm.
Before running this program, the kernel matrix has been precalculated for accelerating the computing. 
So if you want to change the dataset, please follow the steps below to run the program:
(1) run "precalculate_kernelMatrix.m" to create the kernel matrix
(2) run "run_kernel_FALT.m".

3. "run_SALT.m" is a demo for the linear SALT algorithm.

4. "run_FFLT.m" is a demo for the fixed label threshlding version of FALT.
   "run_kernel_FFLT.m" is a demo for the kernelized FFLT algorithm.
   "run_SFLT.m" is a demo for the fixed label threshlding version of SALT.

ATTN: 
- This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Tingting ZHAI (zhtt@yzu.edu.cn).
- This package was developed by Tingting ZHAI (zhtt@yzu.edu.cn). For any problem concerning the code, please feel free to contact Mrs.ZHAI.
#include "mex.h"
#include "math.h"
#include "matrix.h"
#include "stdlib.h"
#include "float.h"
#include "time.h"


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

double maxfun(double a, double b)
{
    if (a >= b) return a;
    else return b;
}

double minfun(double a, double b)
{
    if (a <= b) return a;
    else return b;
}

double uniform(double a, double b)
{
    return ((double) rand())/ RAND_MAX * (b -a) + a;
}

int binornd(double p)
{
    int x;
    double u;
    u = uniform(0.0, 1.0);
    x = (u <= p)? 1:0;
    return(x);
}


int getRandInt(int lowerLimit, int upperLimit){ // get an randomized interger in [lowerLimit, upperLimit]
    return lowerLimit + rand() % (upperLimit - lowerLimit + 1);
}

void randPerm(double *index, int N){
    int i, r1, r2, tmp;
    for(i=0; i < N; i++){
        r1 = getRandInt(0, N-1);
        r2 = getRandInt(0, N-1);
        if (r1!=r2){
            tmp =  index[r1];
            index[r1]= index[r2];
            index[r2] = tmp;
        }
    }
}

double squareNorm(double *x, int len){
    int i;
    double sum = 0;
    for(i = 0;i < len; i++){
        sum = sum + x[i] * x[i];
    }
    return sum;
}

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 6) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "6 inputs required.");
    }
    if(mxIsSparse(prhs[0])==0 && mxIsSparse(prhs[1])==0){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "data/label matrix is not sparse!");
    }
    
    
    double *data, *labels, *w, *index, *x, *y, eta,t, a_t, b_t, coeff;
    int i,j,k,p,N,d,L,low,high,nonzerosNum,low1,high1,epoch,o,Y_t_size,n_Y_t_size,iter,maxIterNum;
    mwIndex *ir, *jc, *ir1, *jc1;
    int * idx;
    /*Read Input Data*/
    data = mxGetPr(prhs[0]);  // use the mxGetPr function to point to the input matrix data.
    labels = mxGetPr(prhs[1]);
    index = mxGetPr(prhs[2]);
    epoch = mxGetScalar(prhs[3]);
    eta = mxGetScalar(prhs[4]);
    maxIterNum = mxGetScalar(prhs[5]);
            
    // a column is an instance
    d = (int)mxGetM(prhs[0]); //get Number of rows in array
    N = (int)mxGetN(prhs[0]); //get Number of columns in array
    L = (int)mxGetM(prhs[1]); //the dimension of each label vector
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    ir1 = mxGetIr(prhs[1]);
    jc1 = mxGetJc(prhs[1]);
    
    /* preparing outputs */
    plhs[0] = mxCreateDoubleMatrix(d, L+1, mxREAL);
    w = mxGetPr(plhs[0]);
    
    // plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    // errNum = mxGetPr(plhs[1]);
    
    double * pred_v = Malloc(double,L+1);
    srand(0);
    
    t = 0;
    for (o = 1; o <= epoch; o++){
        if (o > 1) randPerm(index, N);
        /* start loop */
        for(i = 0; i < N; i++)
        {
            j = index[i] - 1;
            // get each instance
            low = jc[j]; high = jc[j+1];
            nonzerosNum = high - low;
            x = Malloc(double,nonzerosNum);
            idx = Malloc(int,nonzerosNum); // the indices of the non-zero values in x
            for (k = low; k < high; k++){
                x[k-low] = data[k];
                idx[k-low] = ir[k];
            }
            
            // get each label vector
            y = Malloc(double,L);
            for (k = 0; k < L; k++){
                y[k] = -1;
            }
            low1 = jc1[j]; high1 = jc1[j+1];
            for (k = low1; k < high1; k++){
                y[ir1[k]] = labels[k];
            }
            Y_t_size = high1 - low1; // the number of relevant labels
            n_Y_t_size = L - Y_t_size; // the number of irrelevant labels
            
            // compute each predicted value
            for (k = 0; k <= L; k++){
                pred_v[k] = 0;
                for (p = 0; p < nonzerosNum; p++){
                    pred_v[k] += w[k*d + idx[p]] * x[p];
                }
            }
            
            for (iter = 0; iter < maxIterNum; iter++){
                a_t = 0;
                b_t = 0;
                for (k = 0; k < L; k++){  // update the predictive models
                    if (y[k] == 1 && pred_v[k] - pred_v[L] < 1){
                        a_t++;
                        pred_v[k] = 0;
                        for (p = 0; p < nonzerosNum; p++){
                            w[k*d + idx[p]] = w[k*d + idx[p]] + eta/Y_t_size * x[p];
                            w[L*d + idx[p]] = w[L*d + idx[p]] - eta/Y_t_size * x[p];
                            pred_v[k] += w[k*d + idx[p]] * x[p];
                        }
                    }else if (y[k]==-1 && pred_v[L] - pred_v[k] < 1){
                        b_t++;
                        pred_v[k] = 0;
                        for (p = 0; p < nonzerosNum; p++){
                            w[k*d + idx[p]] = w[k*d + idx[p]] - eta/n_Y_t_size * x[p];
                            w[L*d + idx[p]] = w[L*d + idx[p]] + eta/n_Y_t_size * x[p];
                            pred_v[k] += w[k*d + idx[p]] * x[p];
                        }
                    }
                }
                if (a_t == 0 && b_t == 0)    break;
                pred_v[L] = 0;
                for (p = 0; p < nonzerosNum; p++){
                    pred_v[L] += w[L*d + idx[p]] * x[p];
                }
            }
            free(x);
            free(idx);
            free(y);
        }
    }
    free(pred_v);
}


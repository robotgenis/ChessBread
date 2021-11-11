from numba import vectorize
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer 
import cmath

# normal function to run on cpu 
def func(a):                                 
    for i in range(n): 
        a[i] = cmath.cos(a[i]) 

# function optimized to run on gpu 
@vectorize(['complex128(complex128)'], target ="cuda")                         
def func2(x): 
    return cmath.cos(x)

if __name__=="__main__": 
    # n =  10000000 # without GPU: 2477.0648029 with GPU: 5.064376299999822
    n = 50000000                      
    a = np.array([float(i) / float(n) for i in range(n)], dtype="complex128")


    print(n)

    start = timer() 
    func(a) 
    print("without GPU:", timer()-start)     

    start = timer() 
    func2(a) 
    print("with GPU:", timer()-start) 
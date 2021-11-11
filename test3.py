from numba import vectorize
import numpy as np
from timeit import default_timer as timer

@vectorize(['float64(float64)'], target ="cuda")  
def func2(x):
    return x + 1

if __name__=="__main__":
    n = 10000000                   
    a = np.ones(n, dtype="float64")
    
    start = timer()
    a = func2(a)
    print(a[:10])
    print(len(a))
    print("Time:", timer()-start)
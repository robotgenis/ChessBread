from numba import vectorize
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer 

def totCPU(n:int) -> int:
    l = n
    for i in range(len(l)):
        n = l[i]
        res = n; p = 2 
        while p ** 2 <= n:
            if not n % p:
                while not n % p:
                    n //= p
                res *= 1 - (1 / p)
            p += 1

        if n > 1: res *= 1 - (1 / n)

        l[i] = res

@vectorize(['float64(float64)'], target ="cuda")    
def totGPU(n:int) -> int:
    res = n; p = 2 
    while p ** 2 <= n:
        if not n % p:
            while not n % p:
                n //= p
            res *= 1 - (1 / p)
        p += 1

    if n > 1: res *= 1 - (1 / n)

    return int(res)

# normal function to run on cpu 
def func(a):                                 
    for i in range(n): 
        a[i]+= 1    

# function optimized to run on gpu 
@vectorize(['float64(float64)'], target ="cuda")                         
def func2(x): 
    return x + 1

if __name__=="__main__": 
    # n =  10000000 # without GPU: 2477.0648029 with GPU: 5.064376299999822
    n = 80000000                      
    a = np.array([i for i in range(n)], dtype="float64")

    print(n)

    #start = timer() 
    #totCPU(a) 
    #print("without GPU:", timer()-start)     

    start = timer() 
    totGPU(a) 
    print("with GPU:", timer()-start) 
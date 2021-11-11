from numba import guvectorize, float64
from numba.np.ufunc.decorators import vectorize
import numpy as np
from timeit import default_timer as timer

# @vectorize(['int64(int64, int64)'], target ="cuda")
# def f(x):
#     return x + 


@vectorize(['float64(float64)'], target ="cuda")   
def f(n):
    res = n
    p = 2 
    while p ** 2 <= n:
        if not n % p:
            while not n % p:
                n //= p
            res *= 1 - (1 / p)
        p += 1

    if n > 1: res *= 1 - (1 / n)

    return int(res)


@guvectorize([(float64[:], float64[:])], '(n)->(n)')
def g(x, res):
    for i in range(x.shape[0]):
        n = x[i]
        res = n; p = 2 
        while p ** 2 <= n:
            if not n % p:
                while not n % p:
                    n //= p
                res *= 1 - (1 / p)
            p += 1

        if n > 1: res *= 1 - (1 / n)

        x[i] = res


n = 1000000
a = np.array([x for x in range(n)], dtype='float64')
b = np.array([x for x in range(n)], dtype='float64')

for i in range(10):
	start = timer()
	g(a)
	print("Time Guvectorize:", timer()-start)



for i in range(10):
	start = timer()
	f(b)
	print("Time Vectorize:", timer()-start)


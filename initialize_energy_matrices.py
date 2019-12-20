import numpy as np
from numba import jit




def Kinetic(N, BC, t, mu):
    k = np.zeros((N,N))
    k = np.add(np.dot(-t,np.eye(N,k=1)),(np.dot(-t,np.eye(N,k=-1))))
    k = np.add(k , np.dot(-mu,np.eye(N)))
    k [0,-1] = k[-1 , 0 ] = BC * (-t)
    return k

# @jit(nopython=True, parallel=True)
def Interaction(N, L, lamda):
    v = np.zeros((L,N))
    for i in range (L):
        v[i,:] = np.dot(lamda,np.power(-1,np.random.randint(2,size=N) ))
    return v

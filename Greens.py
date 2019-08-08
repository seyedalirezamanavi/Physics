import numpy as np
from numba import jit
from scipy import linalg as LA


# @jit(nopython=True, parallel=True)
def initialize_green_function(k,v,N):
    M_i = np.matmul(LA.expm(k),LA.expm(np.diag(v[0])))
    for i in range(1,N):
        [Q_ip,R_ip] = LA.qr(M_i)
        B_i = np.matmul(LA.expm(k),LA.expm(np.diag(v[i])))
        D_ip = np.diag(np.diag(R_ip))
        M_i = B_i.dot(Q_ip).dot(D_ip)
    [Q_L,R_L] = LA.qr(M_i)
    D_L = np.diag(np.diag(R_L))
    T_L = np.dot(np.diag(np.divide(1,np.diag(R_L))),R_L)
    M_G =np.add(LA.inv(Q_L).dot(LA.inv(T_L)),D_L)
    [Q_G,R_G] = LA.qr(M_G)
    D_G = np.diag(np.diag(R_G))
    T_G = np.dot(np.diag(np.divide(1,np.diag(R_G))),R_G)
    Q_0 = Q_L.dot(Q_G)
    T_0 = T_G.dot(T_L)
    R_0 = R_G.dot(R_L)
    D_0 = np.diag(np.diag(R_0))
    G_1 = LA.inv(Q_0).dot(np.diag(np.divide(1,np.diag(D_0)))).dot(LA.inv(T_0))
    return G_1

# @jit(nopython=True, parallel=True)
def update(k,v,n,l,N):
    Gu = initialize_green_function(k,v,N)
    Gd = initialize_green_function(k,v.dot(-1),N)
    s_new = - v[l,n]
    du = 1+(1-Gu[n][n])*(np.exp(-2*s_new))
    dd = 1+(1-Gu[n][n])*(np.exp(2*s_new))
    d = dd * du
    r = np.random.random()

    if d > r :
        print("%f accepted"%d)
        cu = np.dot(-(np.exp(-2*s_new)-1),Gu[n,:])
        cu[n] += np.exp(-2*s_new)-1
        cd = np.dot(-(np.exp(2*s_new)-1),Gu[n,:])
        cd[n] += np.exp(2*s_new)-1
        bu = np.divide(Gu[:,n],np.add(1,cu))
        bd = np.divide(Gd[:,n],np.add(1,cd))

        bucu = np.zeros((N,N))
        for i in range(len(bu)):
            for j in range(len(cu)):
                bucu[i,j] = bu[i]*cu[j]
        Gu = np.subtract(Gu,bucu)
        bdcd = np.zeros((N,N))
        for i in range(len(bd)):
            for j in range(len(cd)):
                bucu[i,j] = bd[i]*cd[j]
        Gd = np.subtract(Gd,bdcd)
        v[l,n] = s_new
    else:
        print("%f rejected"%d)
    return Gu, Gd, v

# @jit(nopython=True, parallel=True)
def wrapping(k,v,G,l):
    return LA.expm(k).dot(LA.expm(np.diag(v[l,:]))).dot(G).dot(LA.inv(np.exp(k).dot(np.exp(np.diag(v[l,:])))))
